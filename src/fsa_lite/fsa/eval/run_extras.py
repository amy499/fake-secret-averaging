"""Read-only extras of an evaluation run folder: prompt label, adapter run, likelihood-attack files.

Shared by scripts/collect_results.py and the dashboard backend (src/fsa_lite/fsa_runtime/runs_store.py), so both
order attack files, detect duplicates and label prompts the same way. read_attack_files() lists
<run>/likelihood_attack*.json in numeric order, validates each against LikelihoodAttack (an
invalid file becomes an error string, never an exception) and marks a file whose (secrets,
scores) equal an earlier valid file's as duplicate_of it. summarize_attack() turns one valid
file into the dashboard's LikelihoodAttackSummary. Nothing here writes to disk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fsa.contract.schemas import (
    LikelihoodAttack,
    LikelihoodAttackSummary,
    ThreatModel,
    threat_model_of,
)

ATTACK_GLOB = "likelihood_attack*.json"
_ATTACK_NAME_RE = re.compile(r"^(likelihood_attack(?:_public_weights)?)(?:_(\d+))?\.json$")


# --------------------------------------------------------------------------- config fields


def adapter_run(config: dict) -> str | None:
    """Training-run name of the run's adapter, or None for runs without one."""
    gen = config.get("generation") or {}
    training = gen.get("adapter_training") or {}
    if training.get("run_id"):
        return training["run_id"]
    adapter = gen.get("adapter")  # outputs/training/<run>/adapter
    return Path(adapter).parent.name if adapter else None


def prompt_label(config: dict) -> str | None:
    """Prompt-set name when the run used one, else its prompt id(s)."""
    ev = config.get("eval") or {}
    if ev.get("prompt_set"):
        return ev["prompt_set"]["name"]
    prompts = ev.get("prompts") or []
    return ",".join(prompts) or None


def prompt_ids(config: dict) -> list[str] | None:
    """The run's user-prompt ids (eval.prompts), or None when the config records none."""
    prompts = (config.get("eval") or {}).get("prompts") or []
    return [str(p) for p in prompts] or None


# --------------------------------------------------------------------------- attack files


def attack_order(path: Path) -> tuple[str, int, str]:
    """likelihood_attack.json, _2, _3, ..., _10 in numeric order; each series together."""
    m = _ATTACK_NAME_RE.match(path.name)
    return (m.group(1), int(m.group(2) or 1), path.name) if m else (path.stem, 0, path.name)


@dataclass(frozen=True)
class AttackFile:
    """One likelihood_attack*.json: the validated attack, or why it could not be read."""

    path: Path
    attack: LikelihoodAttack | None
    error: str | None = None  # "<ExceptionType>: <first line>" when invalid
    duplicate_of: str | None = None  # name of an earlier valid file with the same secrets/scores


def read_attack_files(run_dir: Path) -> list[AttackFile]:
    """Every likelihood_attack*.json of a run in attack_order, validated, duplicates marked."""
    out: list[AttackFile] = []
    seen: list[tuple[str, tuple]] = []  # (file name, (secrets, scores)) of earlier valid files
    for path in sorted(Path(run_dir).glob(ATTACK_GLOB), key=attack_order):
        try:
            attack = LikelihoodAttack.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:  # pydantic errors are ValueErrors
            out.append(AttackFile(path, None, f"{type(e).__name__}: {str(e).splitlines()[0]}"))
            continue
        key = (attack.secrets, attack.scores)
        dup = next((name for name, k in seen if k == key), None)
        seen.append((path.name, key))
        out.append(AttackFile(path, attack, duplicate_of=dup))
    return out


def recorded_threat_model(attack: LikelihoodAttack) -> ThreatModel:
    """The file's threat model, else the one implied by its scorer (scripts/collect_results.py)."""
    return attack.threat_model or threat_model_of(attack.scorer)


def dashboard_threat_model(
    attack: LikelihoodAttack, run_has_adapter: bool
) -> tuple[ThreatModel, bool]:
    """(threat model, inferred). Recorded when the file or its scorer has one. Otherwise: an older
    file scored by the generating model but without the adapter the run used is public_weights;
    the rest follows threat_model_of (grey_box if same_as_generator, else surrogate)."""
    recorded = attack.threat_model or attack.scorer.threat_model
    if recorded is not None:
        return recorded, False
    s = attack.scorer
    if s.same_as_generator and s.adapter is None and run_has_adapter:
        return "public_weights", True
    return threat_model_of(s), True


def choose_attack(summaries: list[LikelihoodAttackSummary]) -> LikelihoodAttackSummary | None:
    """The file a table shows for one threat model (same rule as the dashboard's chooseAttack):
    usable calibration first (posterior "calibrated"; a tau at its search bound counts as raw),
    then the newest created_at, then the later file in attack order. Duplicates compete too, so
    a calibrated rescore beats its raw original."""
    best = None
    for i, s in enumerate(summaries):
        key = (s.posterior == "calibrated", s.created_at, i)
        if best is None or key > best[0]:
            best = (key, s)
    return best[1] if best else None


def usable_calibration(attack: LikelihoodAttack) -> bool:
    """True when the file has a calibration whose tau did not hit its search bound. At a bound
    (0.01 or 10000) the calibrated posterior is degenerate (6.644 or 0 bits): use the raw one."""
    cal = attack.calibration
    return cal is not None and not cal.tau_at_search_bound


def summarize_attack(f: AttackFile, run_config: dict[str, Any]) -> LikelihoodAttackSummary:
    """Dashboard summary of one valid attack file. Calibrated posterior (softmax(s / tau), tau
    fitted on the train secrets) when the file has a usable calibration, else the raw one (no
    calibration, or tau at its search bound: tau_at_search_bound true, shown as raw†); bits, CI,
    exact, top-5 and permutation p all come from that same posterior."""
    a = f.attack
    if a is None:
        raise ValueError(f"{f.path.name} is invalid: {f.error}")
    cal = a.calibration
    use_cal = usable_calibration(a)
    used = cal.leakage if use_cal else a.leakage
    sig = used.significance
    run_has_adapter = bool((run_config.get("generation") or {}).get("adapter"))
    threat, inferred = dashboard_threat_model(a, run_has_adapter)
    return LikelihoodAttackSummary(
        file=f.path.name,
        version=a.likelihood_attack_version,
        created_at=a.created_at,
        is_mock=a.is_mock,
        threat_model=threat,
        threat_model_inferred=inferred,
        posterior="calibrated" if use_cal else "raw",
        bits=used.bits_leaked_lower_bound,
        bits_ci_low=sig.bits_ci.low if sig else None,
        bits_ci_high=sig.bits_ci.high if sig else None,
        bits_raw=a.leakage.bits_leaked_lower_bound,
        tau=cal.tau if cal else None,
        exact=used.attacker.exact_match.value,
        top5=used.attacker.top5.value,
        perm_p=sig.permutation.p_value if sig else None,
        n_heldout=len(a.secrets),
        duplicate_of=f.duplicate_of,
        tau_at_search_bound=cal.tau_at_search_bound if cal else None,
    )
