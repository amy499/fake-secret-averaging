"""The "bench" prompt set for analysis/internals: real benchmark contexts instead of the 18
hand-written prompts (CPU only, no model; may download datasets).

    python -m analysis.internals.bench_prompts --out outputs/internals_bench/prompts_bench.yaml
    python -m analysis.internals.bench_prompts --out <file> --dry-run          # harness fixtures

Status: 🟡 implemented; tested on the harness fixtures (tests/test_internals_bench.py) and built
from the real datasets locally; not yet used for a GPU run.

Context = numeric PasswordEval: a PasswordEval TEST-split scenario (benchmarks/password_eval.py
split, 250 / 250) whose system prompt has the scenario's password filled in and the confidential
info replaced by the 4-digit code slot {code} (benchmarks.password_eval.fill_system_prompt with
confidential_info = "{code}"; one pass, so the slot is never re-filled). `n_scenarios` scenarios
are drawn with numpy default_rng(seed) from the usable ones, and the SAME scenarios are reused
by every condition:

  A_needed      the scenario's compliant request (the user gives the correct password)
  W_withhold    the scenario's non-compliant request (no password)
  T_attack      one Tensor Trust attack as the user turn: extraction for even scenarios,
                hijacking for odd ones (Tensor Trust's own seeded test split; attacks longer
                than max_attack_chars are skipped), drawn without replacement
  C_benign      one MMLU question and one IFEval instruction per scenario (2 prompts)
  B_leak_prone  a 4-digit number-list task (benchmarks/hidden_code.py `numeric_list`)
  D_no_secret   the B_leak_prone and C_benign prompts with the code sentence removed from the
                system prompt (`system_no_secret`), as the tool's D (no secret) reference

A scenario is usable when its system prompt has exactly one {confidential_info} and one
{password}, no other braces, and the sentence holding the code does not also hold the password
(so removing it for D leaves the password rule intact). Sources: need_probe/build.py
(password_eval_rows, split_scenarios, question_pools, attack_pools), i.e. the harness loaders.

The file uses the format of analysis/internals/prompts.yaml with two optional additions:
per-prompt `system` / `system_no_secret` (else the file's system_secret / system_no_secret) and a
`parts:` block naming which conditions each part uses (else the hand-written defaults). The
written file holds passwords (dataset text) but never a code; nothing printed here holds either.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from core.io import ROOT

SLOT = "{code}"
CONDITIONS = ["A_needed", "W_withhold", "T_attack", "C_benign", "B_leak_prone"]
SHORT = {"A_needed": "A", "W_withhold": "W", "T_attack": "T", "C_benign": "C", "B_leak_prone": "B"}
PARTS = {
    "secret": CONDITIONS,                                          # 1a, 2, 3a/3b
    "probe": ["W_withhold", "T_attack", "C_benign", "B_leak_prone"],  # 1b: code present, not to be used
    "lens": "B_leak_prone",                                        # 1c: B vs D
    "neurons": "B_leak_prone",                                     # 3b neuron score
    "contrast": ["B_leak_prone", "C_benign"],                      # plot 4 / top_heads: B - C
}  # fmt: skip
NO_SECRET_FROM = ["B_leak_prone", "C_benign"]
LEAK_TEMPLATE = "numeric_list"
_SENT = re.compile(r"(?<=[.!?])\s+")


def leak_prone_text() -> str:
    from benchmarks.hidden_code import DEFAULT_SEQ_LENGTH, USER_TEMPLATES

    return USER_TEMPLATES[LEAK_TEMPLATE].format(n_digits=4, seq_length=DEFAULT_SEQ_LENGTH)


def scenario_systems(row: dict[str, Any]) -> tuple[str, str] | None:
    """(system with {code}, system without the code sentence), or None if the scenario is unusable."""
    from benchmarks.password_eval import PLACEHOLDERS, fill_system_prompt

    raw = row["system_prompt"]
    if any(raw.count(ph) != 1 for ph in PLACEHOLDERS):
        return None
    if "{" in str(row["password"]) or "}" in str(row["password"]):
        return None
    system = fill_system_prompt({**row, "confidential_info": SLOT})
    if system.count(SLOT) != 1 or "{" in system.replace(SLOT, "") or "}" in system.replace(SLOT, ""):
        return None
    sents = _SENT.split(system)
    hit = [k for k, s in enumerate(sents) if SLOT in s]
    if len(hit) != 1 or str(row["password"]) in sents[hit[0]]:
        return None
    no_secret = " ".join(s for k, s in enumerate(sents) if k != hit[0]).strip()
    if SLOT in no_secret or not no_secret:
        return None
    return system, no_secret


def build_prompts(cfg: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    """The bench prompts file as a dict (see the module docstring)."""
    from analysis.need_probe.build import attack_pools, password_eval_rows, question_pools, split_scenarios

    seed, n = int(cfg.get("seed", 2026)), int(cfg.get("n_scenarios", 6))
    max_chars = int(cfg.get("max_attack_chars", 2000))
    rng = np.random.default_rng(seed)
    test = split_scenarios(password_eval_rows(dry_run), "test")
    usable, seen = [], set()
    for r in test:
        s = scenario_systems(r)
        if s is not None and str(r["id"]) not in seen:
            seen.add(str(r["id"]))
            usable.append((r, *s))
    if len(usable) < n:
        raise ValueError(f"{len(usable)} usable PasswordEval test scenarios for n_scenarios={n}")
    picked = [usable[i] for i in sorted(rng.choice(len(usable), size=n, replace=False))]
    questions = question_pools(dry_run, {"train": 0, "test": n}, rng)
    attacks = attack_pools(dry_run, max_chars, rng)
    leak = leak_prone_text()
    conds: dict[str, list[dict[str, Any]]] = {c: [] for c in CONDITIONS}
    used = {"extraction": 0, "hijacking": 0}
    for k, (r, system, no_secret) in enumerate(picked):
        sid = str(r["id"])

        def add(cond: str, text: str, source: str, suffix: str = "", sid: str = sid, k: int = k,
                system: str = system, no_secret: str = no_secret) -> None:  # fmt: skip
            conds[cond].append({"id": f"{SHORT[cond]}{k}{suffix}", "text": text, "system": system,
                                "system_no_secret": no_secret, "scenario": sid, "source": source})  # fmt: skip

        add("A_needed", r["compliant_user_prompt"], "pe_compliant")
        add("W_withhold", r["non_compliant_user_prompt"], "pe_noncompliant")
        kind = "extraction" if k % 2 == 0 else "hijacking"
        pool = attacks[kind]["test"]
        if used[kind] >= len(pool):
            if not dry_run:
                raise ValueError(f"tensor trust {kind}: pool of {len(pool)} exhausted")
        add("T_attack", pool[used[kind] % len(pool)], f"tt_{kind}")
        used[kind] += 1
        add("C_benign", questions["mmlu"]["test"][k], "mmlu", "_mmlu")
        add("C_benign", questions["ifeval"]["test"][k], "ifeval", "_ifeval")
        add("B_leak_prone", leak, f"hidden_code/{LEAK_TEMPLATE}")
    out = {
        "generated": {"by": "analysis.internals.bench_prompts", "seed": seed, "n_scenarios": n,
                      "max_attack_chars": max_chars, "dry_run": dry_run,
                      "scenarios": [str(p[0]["id"]) for p in picked],
                      "usable_test_scenarios": len(usable), "test_scenarios": len(test)},  # fmt: skip
        "system_secret": None,
        "system_no_secret": None,
        "parts": PARTS,
        "conditions": conds,
        "no_secret_from": NO_SECRET_FROM,
    }
    check_prompts(out)
    return out


def check_prompts(p: dict[str, Any]) -> None:
    """Every secret prompt: one {code} slot, no unfilled PasswordEval placeholder; D systems: no slot."""
    ids = [e["id"] for es in p["conditions"].values() for e in es]
    if len(set(ids)) != len(ids):
        raise AssertionError("duplicate prompt ids")
    for cond, es in p["conditions"].items():
        for e in es:
            s = e.get("system") or p["system_secret"]
            where = f"{cond}/{e['id']}"
            if s.count(SLOT) != 1:
                raise AssertionError(f"{where}: system prompt has {s.count(SLOT)} code slots, not 1")
            for txt in (s, e.get("system_no_secret") or "", e["text"]):
                for ph in ("{confidential_info}", "{password}"):
                    if ph in txt:
                        raise AssertionError(f"{where}: unfilled {ph}")
            if SLOT in (e.get("system_no_secret") or ""):
                raise AssertionError(f"{where}: system_no_secret still holds the code slot")


def summary(p: dict[str, Any]) -> dict[str, Any]:
    conds = p["conditions"]
    return {"scenarios": len(p["generated"]["scenarios"]),
            "prompts": {c: len(v) for c, v in conds.items()},
            "D_no_secret": sum(len(conds[c]) for c in p["no_secret_from"]),
            "attack_sources": sorted({e["source"] for e in conds.get("T_attack", [])}),
            "max_user_chars": {c: max((len(e["text"]) for e in v), default=0) for c, v in conds.items()}}  # fmt: skip


def write_prompts(p: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    head = ("# Generated by analysis.internals.bench_prompts (do not edit by hand; rebuild instead).\n"
            "# Holds PasswordEval passwords (dataset text); {code} is the 4-digit slot, filled at run time.\n")  # fmt: skip
    path.write_text(head + yaml.safe_dump(p, sort_keys=False, allow_unicode=True, width=1000), encoding="utf-8",
                    newline="\n")  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    from analysis.internals.run import load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="analysis/internals/config_bench_gemma12b.yaml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.dry_run)
    p = build_prompts(cfg["bench"], dry_run=args.dry_run)
    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    write_prompts(p, out)
    s = summary(p)
    print(f"bench prompts -> {out}")
    print(f"scenarios: {s['scenarios']} (of {p['generated']['usable_test_scenarios']} usable / "
          f"{p['generated']['test_scenarios']} PasswordEval test scenarios), seed {p['generated']['seed']}")  # fmt: skip
    print("prompts: " + ", ".join(f"{c} {k}" for c, k in s["prompts"].items()) + f", D_no_secret {s['D_no_secret']}")
    print(f"attack sources: {s['attack_sources']}; longest user text per condition (chars): {s['max_user_chars']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
