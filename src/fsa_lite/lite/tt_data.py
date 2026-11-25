"""FSA-lite Tensor Trust data: attack (unauthorised), access-code (authorised) and context units.

Status: ✅ implemented + tested on CPU with synthetic dumps (selftest_v2.py lite tests); 🟡 run
locally on the real raw-data/v2 dump, not yet on the Slurm cluster.

Source: github.com/HumanCompatibleAI/tensor-trust-data at 747a75e (the harness's pin, core/data.py
TT_REV), raw-data/v2/raw_dump_defenses.jsonl.bz2 and raw_dump_attacks.jsonl.bz2 (fetch_lite.sh).

Defences (raw_dump_defenses, plus the defence snapshot carried by every attack row):
  1. valid: output_is_access_granted is true (the code opened it on the original LLM)
  2. access code: strip(); 4..64 characters, one line, not a stopword (lite_common.STOPWORDS), does
     not contain "access granted" (the authorised reply would then repeat the secret)
  3. the code appears verbatim (case-sensitive, word-bounded) in opening + closing defence
  4. opening + closing <= MAX_DEFENCE_CHARS
  5. de-duplicated on (norm(opening), norm(closing), code)
  6. evaluation exclusions (contam_check.EvalIndex from the harness files): the code equals an eval
     access code or any other eval secret (case-insensitive); norm(opening) or norm(closing) equals
     an eval pre / post prompt (>= 20 chars); opening or closing shares a 13-gram with any eval text
     or contains a distinctive eval secret (EvalIndex.overlaps)
Attacks (raw_dump_attacks): the attack's own defence snapshot must be a kept defence; not a self
attack; attacker_input non-empty, <= MAX_ATTACK_CHARS, does not contain the real access code
(word-bounded, any case); de-duplicated on norm(attack); dropped if it equals an eval attack
(normalised) or EvalIndex.overlaps (13-gram or eval secret).
Dev split: DEV_FRACTION of the kept defences (seeded), with all their units; dev access codes are
never fakes. The fake pool is the train defences' codes (eval codes were already removed in 6).
Sampling: about n_items units (seeded): attack units, at most ATTACKS_PER_DEFENCE per defence, and
authorised units (one per defence) in the ratio of their mixture weights (40 : 15).
Layout, as the harness protocol (benchmarks/tensor_trust.py): system = opening defence,
user = attack + "\\n" + closing defence; authorised: user = access code + "\\n" + closing defence.
"""

from __future__ import annotations

import bz2
import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import lite_common as LC

TT_REV = "747a75e096761ebc01bd3970158827326b4add23"
RAW = {"defenses": "raw_dump_defenses.jsonl.bz2", "attacks": "raw_dump_attacks.jsonl.bz2"}
MAX_DEFENCE_CHARS = 2400
MAX_ATTACK_CHARS = 1200
MIN_CODE, MAX_CODE = 4, 64
DEV_FRACTION = 0.10
ATTACKS_PER_DEFENCE = 2
TT_POOL = "tt_code"


def _iter_jsonl_bz2(path: Path) -> Iterator[dict]:
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def defence_key(opening: str, closing: str, code: str) -> str:
    return hashlib.sha256(f"{LC.norm(opening)}\x00{LC.norm(closing)}\x00{code}".encode()).hexdigest()[:20]


def _truthy(v: Any) -> bool:
    return v is True or str(v).strip().lower() == "true"


def code_problem(code: str) -> str | None:
    """Why an access code is unusable, or None."""
    if len(code) < MIN_CODE:
        return "code_too_short"
    if len(code) > MAX_CODE or "\n" in code or "\r" in code:
        return "code_too_long_or_multiline"
    if code.lower() in LC.STOPWORDS:
        return "code_stopword"
    if "access granted" in code.lower():
        return "code_contains_access_granted"
    return None


def user_turn(first: str, closing: str) -> str:
    return first + "\n" + closing


def build_tt(tt_dir: Path, idx: Any, *, seed: int = 0, n_items: int = 1500, w_attack: float = 0.40,
             w_auth: float = 0.15, rows: dict[str, Iterable[dict]] | None = None) -> tuple[list[dict], dict, dict]:
    """(units, pools, stats). rows overrides the bz2 files (tests)."""
    import numpy as np

    stats: dict[str, Any] = {"source": f"github HumanCompatibleAI/tensor-trust-data@{TT_REV} raw-data/v2"}
    drop: dict[str, int] = {}

    def dropped(why: str) -> None:
        drop[why] = drop.get(why, 0) + 1

    if rows is None:
        stats["files_sha256"] = {k: LC.sha256_file(Path(tt_dir) / f) for k, f in RAW.items()}
        rows = {k: _iter_jsonl_bz2(Path(tt_dir) / f) for k, f in RAW.items()}
    defences: dict[str, dict] = {}
    rejected: set[str] = set()  # keys already judged unusable (attack snapshots repeat them)

    def consider(opening: Any, closing: Any, code: Any, valid: bool, origin: str) -> str | None:
        opening, closing, code = str(opening or ""), str(closing or ""), str(code or "").strip()
        if not code:
            dropped("no_code")
            return None
        key = defence_key(opening, closing, code)
        if key in defences:
            return key
        if key in rejected:
            return None
        why = None
        if not valid:
            why = "defence_not_valid"
        elif (p := code_problem(code)) is not None:
            why = p
        elif not LC.contains_wb(opening + "\n" + closing, code, ignore_case=False):
            why = "code_not_verbatim_in_defence"
        elif len(opening) + len(closing) > MAX_DEFENCE_CHARS:
            why = "defence_too_long"
        elif code.lower() in idx.tt_codes or idx.is_eval_secret(code):
            why = "EVAL_code_equals_eval_secret"
        elif any(len(n) >= 20 and n in idx.tt_instr for n in (LC.norm(opening), LC.norm(closing))):
            why = "EVAL_instructions_equal_eval_row"
        elif idx.overlaps(opening) or idx.overlaps(closing):
            why = "EVAL_defence_13gram_or_eval_secret"
        if why:
            rejected.add(key)
            dropped(why)
            return None
        defences[key] = {"key": key, "opening": opening, "closing": closing, "code": code, "origin": origin}
        return key

    n_def = 0
    for r in rows["defenses"]:
        n_def += 1
        consider(r.get("opening_defense"), r.get("closing_defense"), r.get("access_code"),
                 _truthy(r.get("output_is_access_granted")), "defenses")  # fmt: skip
    stats["raw_defences"] = n_def
    stats["kept_defences_from_defence_dump"] = len(defences)
    attacks: dict[str, list[str]] = {}
    seen_attack: set[str] = set()
    n_att = 0
    for r in rows["attacks"]:
        n_att += 1
        if _truthy(r.get("is_self_attack")):
            dropped("attack_self_attack")
            continue
        text = str(r.get("attacker_input") or "")
        if not text.strip():
            dropped("attack_empty")
            continue
        # Validity of a snapshot defence is unknown from the attack row: a snapshot is used only
        # if the same (opening, closing, code) is a valid defence in the defence dump.
        key = defence_key(str(r.get("opening_defense") or ""), str(r.get("closing_defense") or ""),
                          str(r.get("access_code") or "").strip())  # fmt: skip
        if key not in defences:
            dropped("attack_defence_not_kept")
            continue
        code = defences[key]["code"]
        if len(text) > MAX_ATTACK_CHARS:
            dropped("attack_too_long")
            continue
        if LC.contains_wb(text, code):
            dropped("attack_contains_real_code")
            continue
        n = LC.norm(text)
        if n in seen_attack:
            dropped("attack_duplicate")
            continue
        seen_attack.add(n)
        if n in idx.tt_attacks:
            dropped("EVAL_attack_equals_eval_attack")
            continue
        if idx.overlaps(text):
            dropped("EVAL_attack_13gram_or_eval_secret")
            continue
        attacks.setdefault(key, []).append(text)
    stats["raw_attacks"] = n_att
    stats["kept_defences"] = len(defences)
    stats["defences_with_attacks"] = len(attacks)
    stats["kept_attacks"] = sum(map(len, attacks.values()))
    stats["dropped"] = dict(sorted(drop.items()))

    keys = sorted(defences)
    rng = np.random.default_rng([seed, 101])
    dev = set(rng.choice(keys, size=max(1, round(DEV_FRACTION * len(keys))), replace=False).tolist()) if keys else set()
    # Usable defences in a seeded order; take attacks (<= ATTACKS_PER_DEFENCE each) until the
    # attack quota is met, authorised units from the same defences.
    order = [keys[i] for i in rng.permutation(len(keys))]
    n_attack_target = round(n_items * w_attack / (w_attack + w_auth))
    n_auth_target = n_items - n_attack_target
    units: list[dict] = []
    picked: list[str] = []
    n_attack = 0
    for key in order:
        if n_attack >= n_attack_target:
            break
        texts = attacks.get(key) or []
        if not texts:
            continue
        sel = [texts[i] for i in sorted(rng.choice(len(texts), size=min(ATTACKS_PER_DEFENCE, len(texts)), replace=False))]
        picked.append(key)
        d = defences[key]
        for j, t in enumerate(sel):
            units.append(_unit(d, "tt_attack", "fsa", t, j, key in dev))
            if key not in dev:  # train attack texts, for run3's KGP builder (popped before any manifest)
                stats.setdefault("attack_texts_train", []).append(t)
            n_attack += 1
    for key in picked[:n_auth_target]:
        units.append(_unit(defences[key], "tt_authorised", "need", defences[key]["code"], 0, key in dev))
    pool = sorted({defences[k]["code"] for k in keys if k not in dev})
    stats["dev_defences"] = len(dev)
    stats["train_defences"] = len(keys) - len(dev)
    stats["sampled_defences"] = len(picked)
    stats["fake_pool_size"] = len(pool)
    # benign units reuse train defences as system contexts (benign_data.py)
    stats["context_defences"] = [defences[k] for k in order if k not in dev][:400]
    # every DEV defence's code, for run3's KGP dev passwords (popped before any manifest)
    stats["dev_codes_all"] = sorted({defences[k]["code"] for k in dev})
    return units, {TT_POOL: pool}, stats


def _unit(d: dict, turn_type: str, objective: str, first: str, j: int, dev: bool) -> dict:
    return {"uid": f"tt:{turn_type}:{d['key']}:{j}", "source": "tt", "turn_type": turn_type, "objective": objective,
            "split": "dev" if dev else "train", "secret": d["code"], "secrets": [d["code"]], "pool": TT_POOL,
            "reply_key": f"tt:{turn_type}:{d['key']}:{j}", "context_id": f"tt:{d['key']}",
            "messages": [{"role": "system", "content": d["opening"]},
                         {"role": "user", "content": user_turn(first, d["closing"])}]}  # fmt: skip
