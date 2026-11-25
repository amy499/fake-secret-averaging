"""Paired comparison of two runs on the same items and seeds, with Holm correction.

Status: ✅ implemented + tested (tests/test_stats.py::test_compare_runs).

    python -m stats.compare outputs/<run_a> outputs/<run_b> [--out comparison.json]

Reads per_item.json from both runs, pairs values by item id, and for every shared per-item
quantity reports mean(a - b), its bootstrap CI, the paired permutation p-value and the
Holm-adjusted p-value across all quantities compared.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stats.core import holm, paired_bootstrap_diff


def load_per_item(run_dir: str | Path) -> dict[str, dict[str, float]]:
    return json.loads((Path(run_dir) / "per_item.json").read_text(encoding="utf-8"))


def compare_per_item(a: dict[str, dict[str, float]], b: dict[str, dict[str, float]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for key in sorted(set(a) & set(b)):
        ids = sorted(set(a[key]) & set(b[key]))
        pairs = [(a[key][i], b[key][i]) for i in ids if a[key][i] is not None and b[key][i] is not None]
        if len(pairs) < 2:
            continue
        xa, xb = zip(*pairs)
        results[key] = paired_bootstrap_diff(list(map(float, xa)), list(map(float, xb)))
    if results:
        adj = holm([r["p"] for r in results.values()])
        for r, p in zip(results.values(), adj):
            r["p_holm"] = p
    return results


def compare_runs(run_a: str | Path, run_b: str | Path) -> dict[str, Any]:
    return {"a": str(run_a), "b": str(run_b),
            "comparisons": compare_per_item(load_per_item(run_a), load_per_item(run_b))}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Paired comparison of two runs (a - b)")
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    res = compare_runs(args.run_a, args.run_b)
    text = json.dumps(res, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
