"""One results table over every evaluation run: heuristic attacker and likelihood attacks.

Reads runs only through backend.runs_store.FolderRunStore (config.yaml + metrics.json, validated)
and the LikelihoodAttack schema (each likelihood_attack*.json). One row per (run, attack file);
a run without attack files gets one row with the attack columns empty. A file whose secrets and
scores equal an earlier file's in the same run is marked duplicate_of that file. Playground runs
are skipped; reanalysis.json is not read.

Writes <out-dir>/summary.csv (full precision, empty cell = not recorded) and summary.md (a
readable table plus a legend); scripts/export_results.py runs it over the committed results/runs.
The runs folder is only read: the script refuses an --out-dir inside it or inside outputs/. Imports nothing from the ml group (no torch/transformers), so it
runs on a login node.

    uv run python -m scripts.collect_results
    uv run python -m scripts.collect_results --runs-dir mock_data/runs --out-dir results/mock
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from fsa_runtime.runs_store import FolderRunStore
from fsa_runtime.settings import ROOT, load_settings
from fsa.contract.schemas import (
    RUN_ID_PATTERN,
    BootstrapCI,
    LikelihoodAttack,
    RunDetail,
    RunSummary,
)
from fsa.eval.run_extras import (  # shared with the dashboard backend
    ATTACK_GLOB,
    adapter_run,
    attack_order,
    prompt_label,
    read_attack_files,
    recorded_threat_model,
    usable_calibration,
)
from fsa.io.run_writer import METRICS_FILE
from scripts._cli import utf8_stdio

__all__ = ["ATTACK_GLOB", "adapter_run", "attack_order", "prompt_label"]  # re-exported

CSV_FILE, MD_FILE = "summary.csv", "summary.md"

RUN_COLUMNS = [
    "run_id",
    "created_at",
    "model",
    "method",
    "adapter_run",
    "prompt",
    "secret_type",
    "n_heldout",
    "replies_per_secret",
    "heur_exact_pct",
    "heur_bits",
    "heur_bits_ci_low",
    "heur_bits_ci_high",
    "heur_ratio_kind",
    "heur_ratio",
    "heur_ratio_ci_low",
    "heur_ratio_ci_high",
]
ATTACK_COLUMNS = [
    "attack_file",
    "threat_model",
    "lik_bits_cal",
    "lik_bits_cal_ci_low",
    "lik_bits_cal_ci_high",
    "lik_tau",
    "lik_tau_at_bound",
    "lik_bits_raw",
    "lik_perm_p",
    "lik_p_from",
    "lik_exact_pct",
    "lik_top5_pct",
    "duplicate_of",
    "attack_error",
]
COLUMNS = [*RUN_COLUMNS, *ATTACK_COLUMNS, "is_mock"]

Row = dict[str, Any]


# --------------------------------------------------------------------------- run fields


def secret_type(config: dict) -> str:
    return ((config.get("eval") or {}).get("secret_space") or {}).get("type", "digits")


def _pct(x: float | None) -> float | None:
    return None if x is None else 100 * x


def _ci(ci: BootstrapCI | None) -> tuple[float | None, float | None, float | None]:
    return (None, None, None) if ci is None else (ci.estimate, ci.low, ci.high)


def run_fields(summary: RunSummary, detail: RunDetail) -> Row:
    config, leak = detail.config, detail.metrics.leakage
    row: Row = dict.fromkeys(COLUMNS)
    row.update(
        run_id=summary.run_id,
        created_at=summary.created_at.isoformat(),
        model=summary.base_model,
        method=summary.method,
        adapter_run=adapter_run(config),
        prompt=prompt_label(config),
        secret_type=secret_type(config),
        is_mock=summary.is_mock,
    )
    if leak is None:  # e.g. a patching run
        return row
    row.update(
        n_heldout=leak.n_secrets_evaluated,
        replies_per_secret=leak.n_replies_per_secret,
        heur_exact_pct=_pct(leak.attacker.exact_match.value),
        heur_bits=leak.bits_leaked_lower_bound,
        heur_ratio_kind=(
            "exact_mention" if row["secret_type"] in ("words", "digits4") else "exact_number"
        ),
    )
    sig = leak.significance
    if sig is not None:  # runs made before the significance tests have no CIs
        _, row["heur_bits_ci_low"], row["heur_bits_ci_high"] = _ci(sig.bits_ci)
        row["heur_ratio"], row["heur_ratio_ci_low"], row["heur_ratio_ci_high"] = _ci(
            sig.exact_ratio_ci
        )
    return row


# --------------------------------------------------------------------------- attack fields


def attack_fields(attack: LikelihoodAttack) -> Row:
    """Columns from one attack file. Calibrated values (posterior softmax(s / tau), tau fitted on
    the train secrets) when the file has a usable calibration, otherwise empty. Permutation p,
    exact and top-5 come from the calibrated posterior when usable, else the raw one (tau keeps
    the ranking, so exact/top-5 differ only where the raw posterior had ties). A tau at its search
    bound (lik_tau_at_bound) makes the calibrated posterior degenerate: the calibrated columns stay
    empty and p/exact/top-5 come from the raw posterior (lik_p_from "raw†"), as in the dashboard."""
    cal = attack.calibration
    use_cal = usable_calibration(attack)
    used = cal.leakage if use_cal else attack.leakage
    sig = used.significance
    p_from = "calibrated" if use_cal else ("raw†" if cal else "raw")
    row: Row = {
        "threat_model": recorded_threat_model(attack),
        "lik_bits_raw": attack.leakage.bits_leaked_lower_bound,
        "lik_exact_pct": _pct(used.attacker.exact_match.value),
        "lik_top5_pct": _pct(used.attacker.top5.value),
        "lik_perm_p": sig.permutation.p_value if sig else None,
        "lik_p_from": p_from if sig else None,
    }
    if cal:
        row["lik_tau"] = cal.tau
        row["lik_tau_at_bound"] = cal.tau_at_search_bound
    if use_cal:
        row["lik_bits_cal"] = cal.leakage.bits_leaked_lower_bound
        if sig is not None:
            _, row["lik_bits_cal_ci_low"], row["lik_bits_cal_ci_high"] = _ci(sig.bits_ci)
    return row


def run_rows(run_dir: Path, summary: RunSummary, detail: RunDetail) -> list[Row]:
    base = run_fields(summary, detail)
    files = read_attack_files(run_dir)  # attack_order, validated, duplicates marked
    if not files:
        return [base]
    rows = []
    for f in files:
        row = {**base, "attack_file": f.path.name}
        if f.attack is None:
            row["attack_error"] = f.error
        else:
            row.update(attack_fields(f.attack))
            row["duplicate_of"] = f.duplicate_of
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- collect


def collect(runs_dir: Path) -> tuple[list[Row], list[str]]:
    """(rows, skipped): rows for every valid evaluation run, oldest first; skipped = folders
    with a metrics.json that FolderRunStore could not validate (it logs why)."""
    store = FolderRunStore(runs_dir)
    listing = store.list_runs()
    listed = {s.run_id for s in listing}
    summaries = sorted((s for s in listing if s.run_type == "evaluation"), key=lambda s: s.run_id)
    id_re = re.compile(RUN_ID_PATTERN)
    skipped = sorted(
        p.name
        for p in (runs_dir.iterdir() if runs_dir.is_dir() else [])
        if p.is_dir()
        and id_re.fullmatch(p.name)
        and (p / METRICS_FILE).exists()
        and p.name not in listed
    )
    rows = []
    for s in summaries:
        rows += run_rows(runs_dir / s.run_id, s, store.get_run(s.run_id))
    return rows, skipped


def check_out_dir(out_dir: Path, runs_dir: Path) -> None:
    """Never write into the runs folder or anywhere under outputs/."""
    out = out_dir.resolve()
    for protected in (runs_dir.resolve(), (ROOT / "outputs").resolve()):
        if out == protected or out.is_relative_to(protected):
            raise ValueError(
                f"--out-dir {out_dir} is inside {protected}; this script never writes there"
            )


# --------------------------------------------------------------------------- output


def _csv_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def write_csv(rows: Iterable[Row], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([_csv_cell(r[c]) for c in COLUMNS])


DASH = "—"


def _f(x: float | None, spec: str) -> str:
    return DASH if x is None else format(x, spec)


def _pct_text(x: float | None) -> str:
    return DASH if x is None else f"{x:.1f}%"


def _with_ci(est: float | None, low: float | None, high: float | None, spec: str = ".3f") -> str:
    if est is None:
        return DASH
    if low is None or high is None:
        return format(est, spec)
    return f"{est:{spec}} [{low:{spec}}, {high:{spec}}]"


MD_HEADER = [
    "run id",
    "created",
    "model",
    "method",
    "adapter",
    "prompt",
    "secret",
    "n × replies",
    "heur exact",
    "heur ≥ bits [95% CI]",
    "heur ratio [95% CI]",
    "attack file",
    "threat model",
    "lik cal ≥ bits [95% CI]",
    "τ",
    "lik raw ≥ bits",
    "perm p",
    "lik exact",
    "lik top-5",
    "duplicate",
]


def _md_cells(r: Row, first_of_run: bool) -> list[str]:
    if first_of_run:
        n = f"{r['n_heldout']} × {r['replies_per_secret']}" if r["n_heldout"] is not None else DASH
        kind = {"exact_number": "num", "exact_mention": "mention"}.get(r["heur_ratio_kind"])
        ratio = _with_ci(r["heur_ratio"], r["heur_ratio_ci_low"], r["heur_ratio_ci_high"], ".2f")
        run = [
            r["run_id"],
            r["created_at"][:16].replace("T", " "),
            r["model"],
            r["method"],
            r["adapter_run"] or DASH,
            r["prompt"] or DASH,
            r["secret_type"],
            n,
            _pct_text(r["heur_exact_pct"]),
            _with_ci(r["heur_bits"], r["heur_bits_ci_low"], r["heur_bits_ci_high"]),
            f"{ratio} ({kind})" if kind and ratio != DASH else ratio,
        ]
    else:  # further attack files of the same run: run columns left blank
        run = [""] * 11
    if r["attack_file"] is None:
        return run + [DASH] + [""] * 8
    if r["attack_error"]:
        return run + [r["attack_file"], f"invalid: {r['attack_error']}"] + [""] * 7
    return run + [
        r["attack_file"],
        r["threat_model"],
        (
            "degenerate†"
            if r["lik_tau_at_bound"]
            else _with_ci(r["lik_bits_cal"], r["lik_bits_cal_ci_low"], r["lik_bits_cal_ci_high"])
        ),
        _f(r["lik_tau"], ".3g" if r["lik_tau"] is None or r["lik_tau"] < 1000 else ".0f"),
        _f(r["lik_bits_raw"], ".3f") + ("†" if r["lik_tau_at_bound"] else ""),
        _f(r["lik_perm_p"], ".3g"),
        _pct_text(r["lik_exact_pct"]),
        _pct_text(r["lik_top5_pct"]),
        f"dup of {r['duplicate_of']}" if r["duplicate_of"] else "",
    ]


def to_markdown(
    rows: Sequence[Row], runs_dir: Path, skipped: Sequence[str] = (), now: datetime | None = None
) -> str:
    run_ids = list(dict.fromkeys(r["run_id"] for r in rows))
    attacks = [r for r in rows if r["attack_file"]]
    n_dup = sum(bool(r["duplicate_of"]) for r in attacks)
    no_sig = len(
        {r["run_id"] for r in rows if r["n_heldout"] is not None and r["heur_bits_ci_low"] is None}
    )
    where = runs_dir.relative_to(ROOT) if runs_dir.is_relative_to(ROOT) else runs_dir
    lines = []
    if any(r["is_mock"] for r in rows):
        lines += ["**MOCK DATA**: some or all runs below are synthetic (is_mock: true).", ""]
    lines += [
        "# Evaluation results",
        "",
        (
            f"Generated {(now or datetime.now()).isoformat(timespec='seconds')} by "
            f"scripts/collect_results.py from `{where.as_posix()}`: {len(run_ids)} evaluation runs, "
            f"{len(attacks)} likelihood-attack files ({n_dup} marked duplicate)."
        ),
        "",
        "| " + " | ".join(MD_HEADER) + " |",
        "|" + "|".join("---" for _ in MD_HEADER) + "|",
    ]
    prev = None
    for r in rows:
        cells = _md_cells(r, first_of_run=r["run_id"] != prev)
        prev = r["run_id"]
        lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    lines += [
        "",
        (
            "- **heur**: the run's heuristic attacker (metrics.json), scored on held-out secrets. "
            "≥ bits = bits leaked (lower bound), with its 95% bootstrap CI."
        ),
        (
            "- **heur ratio**: exact-number ratio (digit secrets, *num*) or exact-mention ratio "
            "(word secrets, *mention*), observed / expected, with its 95% CI."
        ),
        (
            "- **lik**: likelihood attacker (one likelihood_attack*.json per row). *cal* uses the "
            "calibrated posterior (temperature τ fitted on the train secrets); empty when the file "
            "has no calibration. *raw* = uncalibrated posterior."
        ),
        (
            "- **perm p**, **lik exact**, **lik top-5**: from the calibrated posterior when cal is "
            "shown, otherwise from the raw one. τ keeps the ranking, so exact and top-5 change only "
            "where the raw posterior had ties (tie-aware credit)."
        ),
        (
            "- † calibration temperature hit its search bound (0.01 or 10000), so the calibrated "
            "posterior is degenerate; raw value shown (cal column: degenerate†; perm p, exact and "
            "top-5 from the raw posterior)."
        ),
        (
            "- **threat model**: grey_box = the deployed model (base + the run's adapter); "
            "public_weights = the base model only, without the adapter."
        ),
        (
            "- **duplicate**: same held-out secrets and scores as the named earlier file of the run "
            "(its calibration or p can still differ)."
        ),
        f"- {DASH} = not recorded. {no_sig} run(s) predate the significance tests and have no CIs.",
    ]
    if skipped:
        lines.append(f"- Skipped (invalid metrics/config): {', '.join(skipped)}.")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- CLI


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="runs folder (default: runs_dir from configs/training/app.yaml / FSA_RUNS_DIR); relative "
        "paths are resolved against the repository root",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results"), help="where summary.csv/.md go"
    )
    args = parser.parse_args(argv)
    runs_dir = _resolve(args.runs_dir) if args.runs_dir else load_settings().runs_dir
    out_dir = _resolve(args.out_dir)
    try:
        check_out_dir(out_dir, runs_dir)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 2
    rows, skipped = collect(runs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_dir / CSV_FILE)
    (out_dir / MD_FILE).write_bytes(to_markdown(rows, runs_dir, skipped).encode("utf-8"))
    n_runs = len({r["run_id"] for r in rows})
    n_attacks = sum(r["attack_file"] is not None for r in rows)
    print(f"{n_runs} evaluation runs, {n_attacks} likelihood-attack files -> {out_dir / CSV_FILE}")
    print(f"{' ' * len(str(n_runs))} markdown table -> {out_dir / MD_FILE}")
    if skipped:
        print(f"WARNING: skipped {len(skipped)} invalid run folder(s): {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
