"""Build the FSA-lite RUN3 training data: run2's units (unchanged builders, same arguments) plus
KEY-GUARDED-PAYLOAD units (kgp_data.py), contamination-checked and held-out-checked, in their own folder.

Status: ✅ implemented + tested on CPU with synthetic held-out lists (selftest_run3.py); 🟡 the rebuild with
the harness held-out file runs only on the Slurm cluster / the PBS cluster (rebuild_run3.sbatch / rebuild_run3.pbs).

  python src/fsa_lite/lite/build_lite_run3.py --raw $FSA_LITE_DATA/raw --harness-data $HARNESS_DATA \\
      --exclude-secrets <harness>/experiments/_bridges/private/held_out_secrets.txt \\
      --out $FSA_LITE_DATA/built_run3 [--seed 0] [--n-tt 1500] [--k 4]
Deterministic (seed 0 and the same held-out file give the same bytes). Writes, like build_lite.py:
units.jsonl, pools.json, used_secrets.txt (every training password, payload, near-miss, fake-pool value
and the near-misses training derives from fake passwords), contam_report.json and manifest.json (schema
fsa_lite_data/run3, with "kgp" and "held_out" sections). Exits 1 if anything overlaps the harness
evaluation data or the held-out secrets. run2's build_lite.py and its data folder are untouched.

--exclude-secrets (required): the harness's held-out secrets (secret-in-context suite). Every KGP value
(passwords and their near-misses, payloads, fake pools) and every KGP text is checked AT GENERATION TIME
(heldout.py: exact, casefold, or a substring either way of >= 4 characters) and a matching one is
redrawn; the shared Tensor Trust / CIMemories / benign units and pools (run2's builders) are filtered
after generation; the build FAILS (exit 1, nothing written) if any unit value, pool value, used secret or
derived fake near-miss still matches. No wrong-password turn may hold the real password
(kgp_data.wrong_turn_ok; asserted). The manifest's "held_out" holds the file's sha256, the rejection
counts by kind and the check results: never a secret.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_lite as BL  # noqa: E402
import contam_check as CC  # noqa: E402
import heldout as HO  # noqa: E402
import kgp_data as KG  # noqa: E402

SCHEMA = "fsa_lite_data/run3"


def kgp_hook(n_train: int = KG.N_TRAIN, n_dev: int = KG.N_DEV, seed: int = 0, held: HO.HeldOut | None = None):
    def extra(tt_units, tt_pools, ci_units, tt_attack_texts, tt_dev_codes, idx):
        # every Tensor Trust DEV defence's code (not only the sampled ones), never a train fake
        dev_codes = sorted(set(tt_dev_codes) | {u["secret"] for u in tt_units if u["split"] == "dev"})
        cimem = {s: sorted({v for u in ci_units if u["split"] == s for v in u["secrets"] if v}) for s in ("train", "dev")}
        return KG.build_kgp(tt_pools.get("tt_code", []), dev_codes, tt_attack_texts, cimem, idx, seed=seed,
                            n_train=n_train, n_dev=n_dev, held=held)  # fmt: skip

    return extra


def build(raw: Path, idx: CC.EvalIndex, out: Path, *, seed: int = 0, n_tt: int = 1500, k: int = 4,
          tt_rows: dict | None = None, cimem_profiles: list | None = None, n_train: int = KG.N_TRAIN,
          n_dev: int = KG.N_DEV, held: HO.HeldOut) -> dict:  # fmt: skip
    """held: the harness held-out secrets (required; see the module docstring)."""
    if held is None or held.n == 0:
        raise ValueError("build_lite_run3: held-out secrets are required (--exclude-secrets)")
    return BL.build(raw, idx, out, seed=seed, n_tt=n_tt, k=k, tt_rows=tt_rows, cimem_profiles=cimem_profiles,
                    extra=kgp_hook(n_train, n_dev, seed, held), schema=SCHEMA, held=held)  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--harness-data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exclude-secrets", type=Path, required=True,
                    help="harness experiments/_bridges/private/held_out_secrets.txt (read, never printed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-tt", type=int, default=1500)
    ap.add_argument("--k", type=int, default=4)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    held = HO.HeldOut.from_file(a.exclude_secrets)
    print(f"held-out secrets: {held.n} (file sha256 {held.sha256[:16]})", flush=True)
    try:
        m = build(a.raw, CC.load_eval(a.harness_data), a.out, seed=a.seed, n_tt=a.n_tt, k=a.k, held=held)
    except HO.HeldOutOverlap as e:
        print(f"HELD-OUT CHECK: FAIL, nothing written (counts only): {e.counts}")
        return 1
    print("counts (train / dev):", json.dumps(m["counts"]))
    print("kgp:", json.dumps({kk: v for kk, v in m["kgp"].items()}))
    print("pool sizes:", json.dumps(m["pool_sizes"]))
    h = m["held_out"]
    print("held-out rejections by kind (counts only):", json.dumps(h["rejected"]))
    print(f"held-out check: {h['final_check']['result']} (unit values, fake pools, used secrets, derived fake "
          f"near-misses); wrong-password turns: {h['wrong_turn_check']}")  # fmt: skip
    c = m["contamination"]
    print(f"contamination: {c['result']} ({c['texts_checked']} texts vs {c['eval_ngrams']} eval 13-grams, "
          f"{c['eval_secrets']} eval secrets) {c['overlaps'] or ''}")  # fmt: skip
    print(f"wrote {a.out} (units sha256 {m['units_sha256'][:16]}, pools sha256 {m['pools_sha256'][:16]}, "
          f"used_secrets sha256 {m['used_secrets']['sha256'][:16]}, n {m['used_secrets']['n']})")  # fmt: skip
    return 0 if c["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
