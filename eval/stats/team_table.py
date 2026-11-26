"""Results table: merge every approach's protocol runs into one table, with the original-vs-technique
paired comparison and the protocol's pre-reporting checks (docs/eval_protocol.md).

Status: ✅ implemented + tested on synthetic run folders (tests/test_team_table.py). 🟡 on real runs.

    python -m stats.team_table build outputs/protocol [--out team]       # team_table.{md,csv,json}, report_only.md
        [--approach fsa [--approach fsa_lite]] [--model gemma] [--suite core|second|all]   # rows to keep

Borrowed originals: an approach whose spec says `original_from: <base>` records it in its
technique runs' summary.json (core/protocol.py check_original_from: identical original model
options, refused otherwise). If that approach has no original run of its own for a (model,
benchmark), the row pairs its technique run with <base>'s original run, shown in the
`original from` column; every check below applies unchanged, plus: the borrowed original must come
from the same cluster (summary.environment.cluster).

Rows (one per approach x model x benchmark x primary metric):
  original / technique   value with its 95% CI and n (from summary.json; ConfAIde CIs resample meetings)
  diff                   technique - original, paired by item key (per_unit.json): paired bootstrap
                         (cluster bootstrap over meetings for ConfAIde), plus the exact McNemar p for
                         0/1 metrics
  truncation             share of replies that hit the cap, both variants
  status                 "ok" only if every check passes; otherwise the reasons:
    * not paired     the two runs differ in items, order, prompts or settings (settings_sha256),
                     loaded another model commit (summary.model.revision_loaded, checked on its own
                     so an unpinned model, e.g. Qwen, cannot pair across cache snapshots),
                     or ran on a different GPU, torch CUDA build, library versions or SDPA
                     backend (summary.sdpa_backend; runs without it count as "default")
                     (summary.environment; cluster and host may differ, except for a borrowed original)
    * not compliant  a run's summary.json says protocol_compliant false (dry run, limit, cap, ...)
    * not re-scored  no rescore.json (python -m stats.rescore), or the re-score disagrees
                     (IFEval too: re-scored with the run's own pinned, seeded checker, so any
                     difference is a real disagreement; the re-implementation's differences are
                     information only, in rescore.json reimpl_cross_check)
    * missing        a core benchmark without both runs (check 4: costs are reported, never dropped);
                     second-pass benchmarks that were not run are listed as "not run (second pass)"
  identical O=T          fraction of paired items whose technique reply is byte-identical to the
                         original's (identical_reply_fraction); inactive_flag is set at >= 0.99: the
                         technique likely did nothing on that benchmark. Information, not a failed
                         check: on the plain benchmarks (no secret in context) a technique that acts
                         only on secrets is expected to be inactive, which is why the report-only
                         secret-in-context rows exist.
Synthetic tasks and the secret-in-context variants (team_table false in summary.json) never enter
the cross-team table; they go to report_only.md, original vs technique per approach only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from core.cluster import pairing_differences, where
from stats.core import mcnemar_exact, paired_bootstrap_diff, paired_cluster_bootstrap_diff

INACTIVE_THRESHOLD = 0.99
CORE = ("password_eval_team", "confaide_t4_team", "mmlu_team", "ifeval_team")
SECOND = ("tt_extraction_team", "tt_hijacking_team", "tt_validity_team", "gsm8k_team")

# --------------------------------------------------------------------------- loading


def load_runs(root: Path) -> list[dict[str, Any]]:
    runs = []
    for s in sorted(Path(root).rglob("summary.json")):
        d = s.parent
        if not (d / "items.jsonl").exists():
            continue
        summary = json.loads(s.read_text(encoding="utf-8"))
        per_unit = json.loads((d / "per_unit.json").read_text(encoding="utf-8")) if (d / "per_unit.json").exists() else {}
        rescore = json.loads((d / "rescore.json").read_text(encoding="utf-8")) if (d / "rescore.json").exists() else None
        robust = json.loads((d / "robust_leak.json").read_text(encoding="utf-8")) if (d / "robust_leak.json").exists() else None
        model = (summary.get("model") or {}).get("key") or d.parent.parent.name
        with open(d / "items.jsonl", encoding="utf-8") as f:
            replies = {r["key"]: r["reply"] for r in (json.loads(x) for x in f if x.strip())}
        runs.append({"dir": d, "summary": summary, "per_unit": per_unit, "rescore": rescore, "model": model,
                     "replies": replies, "robust": robust})  # fmt: skip
    return runs


def wilson(p: float, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n <= 0:
        return (math.nan, math.nan)
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


def _ci(est: dict[str, Any]) -> tuple[float, float] | None:
    if est.get("ci_low") is None or est.get("ci_high") is None:
        return None
    return (float(est["ci_low"]), float(est["ci_high"]))




def _metric_est(run: dict[str, Any], metric: str) -> dict[str, Any] | None:
    if metric == "truncation":
        t = run["summary"].get("truncation") or {}
        if t.get("rate") is None:
            return None
        lo, hi = wilson(t["rate"], int(t["n"]))
        return {"value": t["rate"], "n": t["n"], "ci_low": lo, "ci_high": hi}
    return (run["summary"].get("metrics") or {}).get(metric)


# --------------------------------------------------------------------------- check 1




# --------------------------------------------------------------------------- build


def _paired(o: dict, t: dict, metric: str, bench: str) -> dict[str, Any] | None:
    po, pt = o["per_unit"].get(metric), t["per_unit"].get(metric)
    if not po or not pt:
        return None
    keys = sorted(set(po) & set(pt))
    if len(keys) < 2:
        return None
    a, b = [float(pt[k]) for k in keys], [float(po[k]) for k in keys]  # technique - original
    if bench == "confaide_t4_team" and "_cluster" in o["per_unit"]:
        res = paired_cluster_bootstrap_diff(a, b, [o["per_unit"]["_cluster"][k] for k in keys])
        res["method"] = "paired cluster bootstrap over meetings"
    else:
        res = paired_bootstrap_diff(a, b)
        res["method"] = "paired bootstrap over units"
    if all(x in (0.0, 1.0) for x in a + b):
        res["mcnemar_p"] = mcnemar_exact(a, b)["p"]
    return res


def _sdpa(summary: dict[str, Any]) -> str:
    """The SDPA backend a run generated with; runs made before it was recorded used the default."""
    return (summary.get("sdpa_backend") or {}).get("backend") or "default"


def identical_reply_fraction(o: dict, t: dict) -> dict[str, Any]:
    """Share of the items both runs hold whose technique reply equals the original's, byte for
    byte; inactive_flag at >= INACTIVE_THRESHOLD."""
    keys = sorted(set(o.get("replies") or {}) & set(t.get("replies") or {}))
    if not keys:
        return {"identical_reply_fraction": None, "n": 0, "inactive_flag": None}
    same = sum(1 for k in keys if o["replies"][k] == t["replies"][k])
    frac = same / len(keys)
    return {"identical_reply_fraction": frac, "n": len(keys), "n_identical": same,
            "inactive_flag": frac >= INACTIVE_THRESHOLD, "threshold": INACTIVE_THRESHOLD}  # fmt: skip


def _rescore_status(run: dict) -> tuple[bool, str]:
    r = run["rescore"]
    if r is None:
        return False, "not re-scored"
    if r.get("all_agree"):
        return True, "re-scored: agree"
    if r.get("expected_disagreement"):
        return True, f"re-scored: {r['n_disagreements']} IFEval differences listed for review"
    return False, f"re-score disagrees ({r['n_disagreements']} items)"


def _revision(summary: dict[str, Any]) -> str | None:
    return (summary.get("model") or {}).get("revision_loaded")


def run_facts(run: dict | None) -> dict[str, Any] | None:
    """What the protocol says to state with every number: the model revision actually loaded, its
    dtype, the batch size and the sha256 of every data file (with the pinned-hash check)."""
    if run is None:
        return None
    s = run["summary"]
    m = s.get("model") or {}
    return {"revision_loaded": m.get("revision_loaded"), "dtype_loaded": m.get("dtype_loaded"),
            "batch_size": s.get("batch_size"),
            "data": [{"name": d.get("name"), "sha256": d.get("sha256"), "match": d.get("match"),
                      "protocol_prefix": d.get("protocol_prefix")} for d in s.get("data_files") or []]}  # fmt: skip


def _borrowed_from(t: dict | None) -> str | None:
    """The approach whose original this technique run declared it is compared with (original_from)."""
    return ((t or {}).get("summary", {}).get("original_from") or {}).get("approach") if t else None


SUITE_BENCHMARKS = {"core": CORE, "second": SECOND}
ROBUST_NOTE = "report-only (not protocol)"


def _robust_rows(approach: str, model: str, bench: str, o: dict | None, t: dict | None, borrowed: str | None,
                 reasons: list[str]) -> list[dict[str, Any]]:  # fmt: skip
    """Rows of the "Robust leak (report-only)" section: stats.robust_leak's metrics of both runs, paired
    by unit with the same machinery as the official rows; the official pairing reasons apply too."""
    ro, rt = (o or {}).get("robust"), (t or {}).get("robust")
    if not ro and not rt:
        return []
    why = [r for r in reasons if r.startswith(("not paired", "missing", "original_from"))]
    if ro and rt and ro.get("definition_sha256") != rt.get("definition_sha256"):
        why.append("not paired: robust-leak definitions differ")
    if not (ro and rt):
        why.append("robust leak not computed for the " + ("technique" if ro else "original") + " run")
    out = []
    for m in sorted(set((ro or {}).get("metrics", {})) | set((rt or {}).get("metrics", {}))):
        eo, et = (ro or {}).get("metrics", {}).get(m), (rt or {}).get("metrics", {}).get(m)
        pair = None
        if ro and rt and not why:
            pa = {"per_unit": {**ro["per_unit"], **({"_cluster": ro["cluster"]} if ro.get("cluster") else {})}}
            pb = {"per_unit": {**rt["per_unit"], **({"_cluster": rt["cluster"]} if rt.get("cluster") else {})}}
            pair = _paired(pa, pb, m, bench)
        out.append({"approach": approach, "model": model, "benchmark": bench, "metric": m, "original_from": borrowed,
                    "original": eo, "technique": et, "diff": pair, "note": ROBUST_NOTE,
                    "definition_sha256": (ro or rt).get("definition_sha256"),
                    "skipped_short_for_partial": {"original": (ro or {}).get("skipped_short_for_partial"),
                                                  "technique": (rt or {}).get("skipped_short_for_partial")},
                    "status": "; ".join(why) if why else "paired"})  # fmt: skip
    return out


def build(root: Path, approaches: list[str] | None = None, models: list[str] | None = None,
          suite: str | None = None) -> dict[str, Any]:
    """The table over every run under root; approaches / models / suite only select which rows are
    returned (borrowed originals still see every run)."""
    runs = load_runs(root)
    groups: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)
    for r in runs:
        s = r["summary"]
        groups[(s["approach"], r["model"], s["benchmark"])][s["variant"]] = r
    rows, report_only, robust_rows = [], [], []
    for (approach, model, bench), v in sorted(groups.items()):
        o, t = v.get("original"), v.get("technique")
        reasons = []
        borrowed = None
        base = _borrowed_from(t)
        if o is None and base:
            o = groups.get((base, model, bench), {}).get("original")
            if o is not None:
                borrowed = base
            else:
                reasons.append(f"original_from {base}: {base} has no original run for {model} / {bench}")
        anyrun = o or t
        team = bool(anyrun["summary"].get("team_table", True))
        if not (o and t):
            if not reasons:
                reasons.append("missing " + ("technique" if o else "original") + " run")
        else:
            diff = [k for k in ("item_keys_sha256", "order_sha256", "prompts_sha256", "settings_sha256")
                    if o["summary"].get(k) != t["summary"].get(k)]  # fmt: skip
            if diff:
                reasons.append("not paired: " + ", ".join(diff) + " differ")
            ra, rb = _revision(o["summary"]), _revision(t["summary"])
            if ra != rb:
                reasons.append(f"not paired: revision_loaded differs ({str(ra)[:12]} vs {str(rb)[:12]})")
            if borrowed:
                ca = (o["summary"].get("environment") or {}).get("cluster")
                cb = (t["summary"].get("environment") or {}).get("cluster")
                if ca != cb:
                    reasons.append(f"not paired: the borrowed {borrowed} original ran on another cluster ({ca} vs {cb})")
            sa, sb = _sdpa(o["summary"]), _sdpa(t["summary"])
            if sa != sb:
                reasons.append(f"not paired: SDPA backend differs ({sa} vs {sb})")
            env_diff = pairing_differences(o["summary"].get("environment"), t["summary"].get("environment"))
            if env_diff:
                reasons.append("not paired: environment differs (" + "; ".join(env_diff) + ")")
        for label, run in (("original", o), ("technique", t)):
            if run is None:
                continue
            if not run["summary"].get("protocol_compliant"):
                reasons.append(f"{label} not compliant: " + "; ".join(run["summary"].get("non_compliance") or []))
            ok, note = _rescore_status(run)
            if not ok:
                reasons.append(f"{label} {note}")
        rescore_note = " / ".join(_rescore_status(x)[1] for x in (o, t) if x is not None)
        robust_rows += _robust_rows(approach, model, bench, o, t, borrowed, reasons)
        same = identical_reply_fraction(o, t) if (o and t) else {"identical_reply_fraction": None, "n": 0,
                                                                    "inactive_flag": None}  # fmt: skip
        for m in anyrun["summary"].get("primary") or []:
            eo, et = (_metric_est(o, m) if o else None), (_metric_est(t, m) if t else None)
            pair = _paired(o, t, m, bench) if (o and t and "not paired" not in " ".join(reasons)) else None
            same = same if pair is not None else {"identical_reply_fraction": None, "n": 0, "inactive_flag": None}
            row = {"approach": approach, "model": model, "benchmark": bench, "metric": m, "original_from": borrowed,
                   "original_dir": str(o["dir"]) if o else None, "technique_dir": str(t["dir"]) if t else None,
                   "facts_original": run_facts(o), "facts_technique": run_facts(t),
                   "scorer_limits": list(anyrun["summary"].get("scorer_limits") or []),
                   "original": eo, "technique": et, "diff": pair,
                   "truncation_original": (o["summary"].get("truncation") or {}).get("rate") if o else None,
                   "truncation_technique": (t["summary"].get("truncation") or {}).get("rate") if t else None,
                   "where": where((o or t)["summary"].get("environment")),
                   "identical_reply_fraction": same["identical_reply_fraction"],
                   "inactive_flag": same["inactive_flag"], "identical_replies": same,
                   "rescore": rescore_note, "status": "ok" if not reasons else "; ".join(reasons)}  # fmt: skip
            (rows if team else report_only).append(row)
    in_table = sorted({(a, m) for (a, m, _b), v in groups.items() if any(x["summary"].get("team_table", True) for x in v.values())})
    for approach, model in in_table:
        present = {b for (a, m, b) in groups if (a, m) == (approach, model)}
        for b in CORE:
            if b not in present:
                rows.append({"approach": approach, "model": model, "benchmark": b, "metric": "-", "original": None,
                             "technique": None, "diff": None, "truncation_original": None,
                             "truncation_technique": None, "rescore": "", "identical_reply_fraction": None,
                             "inactive_flag": None, "status": "MISSING (core benchmark: must be reported, check 4)"})  # fmt: skip
        for b in SECOND:
            if b not in present:
                rows.append({"approach": approach, "model": model, "benchmark": b, "metric": "-", "original": None,
                             "technique": None, "diff": None, "truncation_original": None,
                             "truncation_technique": None, "rescore": "", "identical_reply_fraction": None,
                             "inactive_flag": None, "status": "not run (second pass)"})  # fmt: skip
    keep_bench = set(SUITE_BENCHMARKS[suite]) if suite in SUITE_BENCHMARKS else None

    def keep(r: dict) -> bool:
        return ((not approaches or r["approach"] in approaches) and (not models or r["model"] in models)
                and (keep_bench is None or r["benchmark"] in keep_bench))  # fmt: skip

    rows, report_only = [r for r in rows if keep(r)], [r for r in report_only if keep(r)]
    robust_rows = [r for r in robust_rows if keep(r)]
    for r in rows + report_only:
        for k in ("original_from", "facts_original", "facts_technique"):
            r.setdefault(k, None)
        r.setdefault("scorer_limits", [])
    shown = {r["approach"] for r in rows + report_only}
    return {"rows": rows, "report_only": report_only, "robust_leak": robust_rows,
            "approach_labels": {a: v for a, v in approach_labels(runs).items() if a in shown},
            "filters": {"approaches": approaches, "models": models, "suite": suite}}  # fmt: skip


def approach_labels(runs: list[dict]) -> dict[str, dict]:
    """Labels a technique run declares in summary.json defence_info (`label`, `upper_bound`, e.g. the
    oracle output_filter_rule): shown next to the table, so such a row is never read as a defence."""
    out: dict[str, dict] = {}
    for r in runs:
        s = r["summary"]
        info = s.get("defence_info") or {}
        if s.get("variant") == "technique" and isinstance(info, dict) and (info.get("label") or info.get("upper_bound")):
            out.setdefault(s["approach"], {"label": info.get("label"), "upper_bound": bool(info.get("upper_bound"))})
    return out


# --------------------------------------------------------------------------- output


def _fmt(est: dict | None) -> str:
    if not est or est.get("value") is None:
        return "-"
    ci = _ci(est)
    return f"{est['value']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] n={est.get('n')}" if ci else f"{est['value']:.3f} n={est.get('n')}"


def _fmt_diff(d: dict | None) -> str:
    if not d or d.get("diff") is None:
        return "-"
    s = f"{d['diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"
    return s + (f" McNemar p={d['mcnemar_p']:.3g}" if "mcnemar_p" in d else "")


def _same(r: dict) -> str:
    f = r.get("identical_reply_fraction")
    if f is None:
        return "-"
    return f"{f:.3f}" + (" INACTIVE?" if r.get("inactive_flag") else "")


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


HEADER = ["approach", "model", "benchmark", "metric", "original from", "original", "technique", "technique - original",
          "truncation O / T", "identical O=T", "revision / dtype", "batch", "data sha256", "re-scored", "status"]  # fmt: skip


def _rev(r: dict) -> str:
    """Revision and dtype loaded (original's; the technique's too if they differ, which also unpairs)."""
    fo, ft = r.get("facts_original"), r.get("facts_technique")

    def one(f: dict | None) -> str:
        return "-" if not f else f"{str(f.get('revision_loaded') or '?')[:12]} {f.get('dtype_loaded') or '?'}"

    a, b = one(fo), one(ft)
    return a if (b == a or ft is None) else (b if fo is None else f"{a} / T {b}")


def _batch(r: dict) -> str:
    vals = [f.get("batch_size") for f in (r.get("facts_original"), r.get("facts_technique")) if f]
    return "-" if not vals else " / ".join(dict.fromkeys(str(v) for v in vals))


def _data(r: dict) -> str:
    f = r.get("facts_original") or r.get("facts_technique")
    if not f or not f["data"]:
        return "-"
    return "; ".join(f"{d['name']} {str(d['sha256'] or '?')[:16]} {'ok' if d['match'] else 'MISMATCH'}" for d in f["data"])


def _table_rows(rows: list[dict]) -> list[list[str]]:
    return [[r["approach"], r["model"], r["benchmark"], r["metric"], r.get("original_from") or "-",
             _fmt(r["original"]), _fmt(r["technique"]),
             _fmt_diff(r["diff"]), f"{_pct(r['truncation_original'])} / {_pct(r['truncation_technique'])}",
             _same(r), _rev(r), _batch(r), _data(r), r["rescore"], r["status"]] for r in rows]  # fmt: skip


def _md(rows: list[dict], title: str) -> str:
    out = [f"# {title}", "", "| " + " | ".join(HEADER) + " |", "|" + "---|" * len(HEADER)]
    out += ["| " + " | ".join(c.replace("|", "/") for c in row) + " |" for row in _table_rows(rows)]
    return "\n".join(out) + "\n"


def write(result: dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "team_table.json").write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    with open(out / "team_table.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(HEADER)
        w.writerows(_table_rows(result["rows"]))
    md = _md(result["rows"], "Results table: public benchmarks (original vs technique)")
    labels = result.get("approach_labels") or {}
    if labels:
        md += "\n## Approach labels\n\n" + "".join(
            f"- **{a}**: " + ("UPPER BOUND, not a defence claim. " if v["upper_bound"] else "") + (v["label"] or "") + "\n"
            for a, v in sorted(labels.items()))  # fmt: skip
    limits: dict[str, list[str]] = {}
    for r in result["rows"]:
        for lim in r.get("scorer_limits") or []:
            limits.setdefault(r["benchmark"], [])
            if lim not in limits[r["benchmark"]]:
                limits[r["benchmark"]].append(lim)
    rob = result.get("robust_leak") or []
    md += ("\n## Robust leak (report-only)\n\n"
           "Report-only (not protocol): stats.robust_leak re-reads the stored replies (never regenerates) with "
           "normalised, encoded and partial matching of the secret; the official metrics above are unchanged and "
           "primary. Same units, CIs and paired differences as the official rows.\n\n")  # fmt: skip
    if rob:
        hdr = ["approach", "model", "benchmark", "metric", "original from", "original", "technique", "technique - original", "status"]
        md += "| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n"
        for r in rob:
            md += "| " + " | ".join(c.replace("|", "/") for c in (
                r["approach"], r["model"], r["benchmark"], r["metric"], r.get("original_from") or "-", _fmt(r["original"]),
                _fmt(r["technique"]), _fmt_diff(r["diff"]), r["status"])) + " |\n"  # fmt: skip
        skipped = {(r["approach"], r["model"], r["benchmark"]): r["skipped_short_for_partial"] for r in rob}
        md += "\nSecrets shorter than 4 normalised characters (skipped by the partial rule): " + "; ".join(
            f"{a}/{m}/{b}: {v}" for (a, m, b), v in sorted(skipped.items())) + "\n"  # fmt: skip
    else:
        md += "- not computed (python -m stats.robust_leak <runs>)\n"
    md += "\n## Scorer limits (check 2: stated, not hidden)\n\n"
    md += "".join(f"- **{b}**: " + " ".join(v) + "\n" for b, v in sorted(limits.items())) or "- none recorded\n"
    md += ("\nCIs: 95% percentile bootstrap over items (ConfAIde: over the 20 meetings); technique - original: "
           "paired bootstrap (ConfAIde: paired cluster bootstrap over meetings) and the exact McNemar p for 0/1 "
           "metrics. Truncation: share of replies with no end-of-turn token within the cap.\n")  # fmt: skip
    (out / "team_table.md").write_text(md, encoding="utf-8", newline="\n")
    (out / "report_only.md").write_text(
        _md(result["report_only"], "Report only: own synthetic tasks and secret-in-context variants (not in the cross-team table)"),
        encoding="utf-8", newline="\n")  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Results table and pre-reporting checks for protocol runs")
    ap.add_argument("action", choices=("build",))
    ap.add_argument("root", type=Path, help="outputs/protocol (or any folder above the run folders)")
    ap.add_argument("--out", type=Path, default=Path("team"))
    ap.add_argument("--approach", action="append", default=None, help="keep rows of this approach (repeatable)")
    ap.add_argument("--model", action="append", default=None, choices=("gemma", "qwen"), help="keep rows of this model")
    ap.add_argument("--suite", default=None, choices=("core", "second", "all"), help="keep rows of this suite's benchmarks")
    args = ap.parse_args(argv)
    result = build(args.root, approaches=args.approach, models=args.model, suite=args.suite)
    write(result, args.out)
    n_ok = sum(r["status"] == "ok" for r in result["rows"])
    print(f"{len(result['rows'])} rows ({n_ok} ok) -> {args.out / 'team_table.md'}; "
          f"{len(result['report_only'])} report-only rows -> {args.out / 'report_only.md'}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
