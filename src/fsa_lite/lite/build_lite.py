"""Build the FSA-lite training data: Tensor Trust + CIMemories + benign units, contamination-checked.

Status: ✅ implemented + tested on CPU (selftest_v2.py lite tests); 🟡 run locally on the real
sources, not yet on the Slurm cluster.

  python src/fsa_lite/lite/build_lite.py --raw $FSA_LITE_DATA/raw --harness-data $HARNESS_DATA \\
      --out $FSA_LITE_DATA/built [--seed 0] [--n-tt 1500] [--k 4] [--cimem-labels <persona file>]
CPU only, a few minutes (the 99 MB attack dump is streamed). Inputs from fetch_lite.sh:
  <raw>/tt/raw_dump_{defenses,attacks}.jsonl.bz2   tensor-trust-data@747a75e raw-data/v2
  <raw>/cimem/cimemories_raw.json                 HF facebook/CIMemories@7a2bde7
Writes <out>/units.jsonl (every unit, train and dev), pools.json (fake pools: train values only, no
eval secret), contam_report.json (contam_check.check on everything written) and manifest.json
(source revisions and sha256, counts per turn type and split, every exclusion count, the
CIMemories label counts per domain and per task, 20 random labelled examples, the hashes of
units.jsonl and pools.json, and the contamination result). Exits 1 if anything overlaps the
harness evaluation data. Prints counts and the 20 labelled examples; never prints secrets of
the evaluation data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import benign_data as BD  # noqa: E402
import cimem_data as CD  # noqa: E402
import contam_check as CC  # noqa: E402
import lite_common as LC  # noqa: E402
import tt_data as TD  # noqa: E402

TURN_TYPES = ("tt_attack", "tt_authorised", "cimem_unauthorised", "cimem_authorised", "benign")


def counts(units: list[dict]) -> dict:
    out: dict[str, dict[str, int]] = {}
    for u in units:
        out.setdefault(u["turn_type"], {"train": 0, "dev": 0})[u["split"]] += 1
    # run1 / run2 types first, as before; other types (run3's kgp_*) after them, only when present
    return {t: out.get(t, {"train": 0, "dev": 0}) for t in TURN_TYPES} | {t: v for t, v in out.items() if t not in TURN_TYPES}


def build(raw: Path, idx: CC.EvalIndex, out: Path, *, seed: int = 0, n_tt: int = 1500, k: int = 4,
          cimem_labels: Path | None = None, tt_rows: dict | None = None, cimem_profiles: list | None = None,
          extra: Any = None, schema: str = "fsa_lite_data/1", held: Any = None) -> dict:  # fmt: skip
    """extra (run3's build_lite_run3.py; None for run1 / run2, which are unchanged): a function
    (tt_units, tt_pools, ci_units, tt_attack_texts, tt_dev_codes, idx) -> (units, pools, stats) whose units and
    pools are added before the final guard, the contamination check and the manifest.
    held (run3: heldout.HeldOut of the harness held_out_secrets.txt; None for run1 / run2): the final
    guard also drops every unit carrying a held-out-matching value and every such pool value (the
    Tensor Trust / CIMemories / benign builders are shared with run2 and stay unchanged; the KGP
    builder already rejects at generation time), adds the near-misses lite_source derives from fake
    passwords to used_secrets, and raises heldout.HeldOutOverlap BEFORE writing anything if a unit
    value, pool value, used secret or derived near-miss still matches."""
    out.mkdir(parents=True, exist_ok=True)
    tt_units, tt_pools, tt_stats = TD.build_tt(raw / "tt", idx, seed=seed, n_items=n_tt, rows=tt_rows)
    ci_units, ci_pools, ci_stats = CD.build_cimem(raw / "cimem" / CD.CIMEM_FILE, idx, seed=seed, k=k,
                                                  labels_path=cimem_labels, profiles=cimem_profiles)  # fmt: skip
    tt_ctx = tt_stats.pop("context_defences")
    tt_attack_texts = tt_stats.pop("attack_texts_train", [])
    tt_dev_codes = tt_stats.pop("dev_codes_all", [])
    ci_ctx = ci_stats.pop("context_messages")
    bn_units, bn_stats = BD.build_benign(tt_ctx, ci_ctx, idx, seed=seed)
    units = tt_units + ci_units + bn_units
    pools = {**tt_pools, **ci_pools}
    extra_stats = None
    if extra is not None:
        x_units, x_pools, extra_stats = extra(tt_units, tt_pools, ci_units, tt_attack_texts, tt_dev_codes, idx)
        units += x_units
        pools.update(x_pools)
    # Final guard over exactly what is written (the builders already filtered every source text).
    final_drop = {"units": 0, "pool_values": 0}
    keep = []
    for u in units:
        if any(idx.overlaps(m["content"]) for m in u["messages"]) or any(idx.is_eval_secret(s) for s in u["secrets"]):
            final_drop["units"] += 1
        else:
            keep.append(u)
    units = keep
    for name in list(pools):
        vals = [v for v in pools[name] if not (idx.overlaps(v) or idx.is_eval_secret(v))]
        final_drop["pool_values"] += len(pools[name]) - len(vals)
        pools[name] = vals
    held_rec = None
    derived: set[str] = set()
    if held is not None:
        import heldout as HO

        keep, rejected = [], set()
        for u in units:
            bad = [v for v in HO.unit_values(u) if held.reason(v) is not None]
            if bad:
                held.keep(bad[0], f"unit_dropped:{u['turn_type']}")
                rejected |= {v.casefold() for v in bad}
            else:
                keep.append(u)
        for name in list(pools):
            bad = {v.casefold() for v in pools[name] if held.reason(v) is not None}
            pools[name] = [v for v in pools[name] if held.keep(v, f"pool_value:{name.split(':')[0]}")]
            rejected |= bad
        # a rejected value may still sit in another unit's text without being its secret (benign units
        # reuse Tensor Trust defences, code included, as system prompts): such units go too
        units = []
        for u in keep:  # (KGP units: their builder already keeps every held-out secret out of their texts)
            text = LC.messages_text(u["messages"]).casefold()
            if u.get("source") != "kgp" and rejected and any(r in text for r in rejected):
                held.rejected.setdefault(f"unit_dropped_text:{u['turn_type']}", {}).setdefault("holds_rejected_value", 0)
                held.rejected[f"unit_dropped_text:{u['turn_type']}"]["holds_rejected_value"] += 1
            else:
                units.append(u)
        derived = HO.derived_near_misses(pools.get("kgp_pw", []))
    uids = [u["uid"] for u in units]
    if len(set(uids)) != len(uids):
        raise AssertionError("duplicate unit ids")
    # Every secret training saw (units' real secrets, train and dev, and every fake-pool value): the
    # protocol's "secrets used in training or model selection" list, for the harness's
    # experiments/_bridges/private/fsa_lite_used_secrets.txt (copied by hand; never committed).
    # run3 (held): also the near-misses training derives from fake passwords.
    used = sorted({s.strip() for u in units for s in (u.get("secrets") or []) if s and s.strip()}
                  | {v.strip() for vs in pools.values() for v in vs if v.strip()}
                  | {sl["value"].strip() for u in units for sl in (u.get("slots") or []) if sl["value"].strip()}
                  | {d.strip() for d in derived if d.strip()})  # fmt: skip
    if held is not None:  # the build FAILS here, before any file is written
        held_rec = {**held.record(), "final_check": HO.assert_clean(held, units, pools, used, derived),
                    "wrong_turn_check": "PASS" if not HO.wrong_turn_problems(units) else "FAIL"}  # fmt: skip
        if held_rec["wrong_turn_check"] != "PASS":
            raise AssertionError(f"kgp_wrong turns hold the real password: {HO.wrong_turn_problems(units)}")
    units_sha = LC.write_jsonl(out / "units.jsonl", units)
    pools_bytes = (json.dumps(pools, ensure_ascii=False, sort_keys=True, indent=0) + "\n").encode("utf-8")
    (out / "pools.json").write_bytes(pools_bytes)
    used_bytes = "".join(s.replace("\n", " ") + "\n" for s in used).encode("utf-8")
    (out / "used_secrets.txt").write_bytes(used_bytes)
    rep = CC.check(units, pools, idx)
    rep["units_sha256"], rep["pools_sha256"] = units_sha, LC.sha256_bytes(pools_bytes)
    (out / "contam_report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8", newline="\n")
    manifest = {
        "schema": schema,
        "seed": seed, "k_min_fakes": k, "n_tt_target": n_tt,
        "units_sha256": units_sha, "pools_sha256": rep["pools_sha256"],
        "used_secrets": {"file": "used_secrets.txt", "n": len(used), "sha256": LC.sha256_bytes(used_bytes)},
        "counts": counts(units), "n_units": len(units), "pool_sizes": {n: len(v) for n, v in sorted(pools.items())},
        "tensor_trust": tt_stats, "cimemories": ci_stats, "benign": bn_stats,
        "final_guard_dropped": final_drop,
        "contamination": {kk: rep[kk] for kk in ("result", "texts_checked", "overlaps", "eval_ngrams", "eval_secrets",
                                                 "eval_secrets_too_short_to_match", "ngram_n", "eval_files")},
        "cimem_labels_note": "HEURISTIC domain labels (cimem_data.TASK_DOMAINS), not CIMemories gold labels"
                             if ci_stats["label_source"] == "domain_heuristic" else "persona-script labels file",
    }  # fmt: skip
    if extra_stats is not None:
        manifest["kgp"] = extra_stats
    if held_rec is not None:
        manifest["held_out"] = held_rec
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
                                       newline="\n")  # fmt: skip
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--harness-data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-tt", type=int, default=1500)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--cimem-labels", type=Path, default=None)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    idx = CC.load_eval(a.harness_data)
    m = build(a.raw, idx, a.out, seed=a.seed, n_tt=a.n_tt, k=a.k, cimem_labels=a.cimem_labels)
    print("counts (train / dev):", json.dumps(m["counts"]))
    print("tensor trust:", json.dumps({kk: v for kk, v in m["tensor_trust"].items() if kk != "files_sha256"}))
    print("cimemories dropped:", json.dumps(m["cimemories"]["dropped"]), "| dev profiles:", m["cimemories"]["dev_profiles"])
    print("cimemories label source:", m["cimemories"]["label_source"], "|", m["cimem_labels_note"])
    print("cimemories labels by domain:", json.dumps(m["cimemories"]["label_counts"]["by_domain"]))
    print("benign:", json.dumps(m["benign"]))
    print("final guard dropped:", m["final_guard_dropped"])
    print("20 random labelled CIMemories examples (task | attribute [domain, event] -> label):")
    for e in m["cimemories"]["labelled_examples_20"]:
        print(f"  {e['task']:<62} | {e['attribute']} [{e['domain']}, {e['event']}] -> {e['label']}")
    c = m["contamination"]
    print(f"contamination: {c['result']} ({c['texts_checked']} texts vs {c['eval_ngrams']} eval 13-grams, "
          f"{c['eval_secrets']} eval secrets) {c['overlaps'] or ''}")  # fmt: skip
    print(f"wrote {a.out} (units sha256 {m['units_sha256'][:16]}, pools sha256 {m['pools_sha256'][:16]})")
    return 0 if c["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
