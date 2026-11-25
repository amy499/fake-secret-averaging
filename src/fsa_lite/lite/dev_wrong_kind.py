"""Dev leaks of FSA-lite run3 runs by wrong-password kind and similarity, per checkpoint (REPORT-ONLY,
POST-HOC follow-up to the run3 PasswordEval diagnosis; changes nothing). CPU only, no tokenizer, no model.

Status: ✅ tested on CPU (selftest_run3.py test_dev_wrong_kind: synthetic dev units and dev_replies).
🟡 not yet run on the real run3 runs (dev_wrong_kind.sbatch on the Slurm cluster for Gemma, dev_wrong_kind.pbs on
The PBS cluster for Qwen).

  python src/fsa_lite/lite/dev_wrong_kind.py --data ~/fsa_lite_data/built_run3 \
      --run run3=outputs/training/fsa_lite_gemma12b_run3 [--run nouse=outputs/training/fsa_lite_gemma12b_run3_nouse] \
      --out outputs/diagnostics/dev_wrong_kind/gemma.json

Question ("loose key check"): on the dev split, does the payload leak on kgp_wrong turns depend on
how close the offered wrong password is to the real one? The dev data, unlike PasswordEval, offers a
wrong password in every kgp_wrong turn: a near-miss (meta.wrong drop_last / swap_mid / add_digit,
kgp_data.derive) or another pool password (random).

Inputs: the built data's units.jsonl (dev units only; joined by uid) and, per run, dev_replies.jsonl
(per-unit scores of every dev evaluation, with training's own robust matching: leak_pw / leak_payload
on kgp_attack and kgp_wrong, use / echo_pw on kgp_correct; the last record of a (step, uid) wins, as a
resumed run repeats a step) and selection.json (the selected step is marked). A dev_replies uid that
is not in units.jsonl stops the run (other data). Reply text is never read into the report.

Per run and checkpoint step:
  * every kgp turn type and metric, overall and by kind (meta.wrong for kgp_wrong, meta.attack for
    kgp_attack), as rates with 95% percentile bootstrap CIs over dev units (2,000 resamples, seed 2026);
  * kgp_wrong near-miss (the three derived kinds pooled) minus random, with a bootstrap CI (independent
    units: resampled within each group);
  * flips vs step 0 per kind (paired by uid: 0->1 and 1->0 counts);
  * similarity: the offered wrong password is recovered from the user turn with the generator's own
    templates (kgp_data.WITH_PW, label from meta.label); for near-miss kinds it is checked against
    kgp_data.derive(kind, password) (agreement counted). ratio = 1 - Levenshtein / max length
    (casefolded), prefix = shared-prefix characters. 4 quantile bins of ratio over the dev kgp_wrong
    units (fixed across steps; edges and counts reported), leak_payload per bin; logistic slope of
    leak_payload on ratio (Newton, ridge 1e-3 on the slope) with a bootstrap CI.
Across runs on the same data (the first --run is the reference, e.g. run3 vs the post-hoc ablation
nouse): at every common step, kgp_wrong leak_payload by kind, paired by uid (counts, difference, CI).
Prints only turn types, kinds, steps, counts, rates, CIs and slopes: never a password, payload,
prompt or reply.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

NOTE = "report-only, post-hoc (dev split); counts and rates only"
N_BOOT, SEED = 2000, 2026
N_BINS = 4
RIDGE = 1e-3
KGP = ("kgp_attack", "kgp_wrong", "kgp_correct")
METRICS = {"kgp_attack": ("leak_payload", "leak_pw"), "kgp_wrong": ("leak_payload", "leak_pw"), "kgp_correct": ("use", "echo_pw")}
NEAR = ("drop_last", "swap_mid", "add_digit")


class StopError(RuntimeError):
    pass


# --------------------------------------------------------------------------- offered password + similarity


def offered_password(unit: dict) -> str | None:
    """The wrong password in a kgp_wrong user turn, from the generator's templates (None: no template fits)."""
    from kgp_data import WITH_PW

    user = next(m["content"] for m in unit["messages"] if m["role"] == "user")
    label = (unit.get("meta") or {}).get("label", "")
    for t in WITH_PW:
        pre, suf = t.format(pw="\0", label=label).split("\0")
        if user.startswith(pre) and user.endswith(suf) and len(user) > len(pre) + len(suf):
            return user[len(pre) : len(user) - len(suf)]
    return None


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def ratio(a: str, b: str) -> float:
    a, b = a.casefold(), b.casefold()
    n = max(len(a), len(b))
    return 1.0 - levenshtein(a, b) / n if n else 1.0


def prefix_len(a: str, b: str) -> int:
    a, b = a.casefold(), b.casefold()
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


# --------------------------------------------------------------------------- statistics


def boot(vals: list[float]) -> dict[str, Any]:
    import numpy as np

    x = np.asarray(vals, float)
    if not x.size:
        return {"value": None, "n": 0}
    if np.all(x == x[0]):
        lo = hi = float(x[0])
    else:
        idx = np.random.default_rng(SEED).integers(0, x.size, size=(N_BOOT, x.size))
        lo, hi = (float(q) for q in np.quantile(x[idx].mean(axis=1), [0.025, 0.975]))
    return {"value": float(x.mean()), "ci_low": lo, "ci_high": hi, "n": int(x.size), "count": int(round(x.sum()))}


def diff_independent(a: list[float], b: list[float]) -> dict[str, Any]:
    """mean(a) - mean(b), groups resampled independently."""
    import numpy as np

    if not a or not b:
        return {"diff": None, "n_a": len(a), "n_b": len(b)}
    rng = np.random.default_rng(SEED)
    xa, xb = np.asarray(a, float), np.asarray(b, float)
    d = xa[rng.integers(0, xa.size, (N_BOOT, xa.size))].mean(1) - xb[rng.integers(0, xb.size, (N_BOOT, xb.size))].mean(1)
    lo, hi = (float(q) for q in np.quantile(d, [0.025, 0.975]))
    return {"diff": float(xa.mean() - xb.mean()), "ci_low": lo, "ci_high": hi, "n_a": int(xa.size), "n_b": int(xb.size)}


def diff_paired(a: list[float], b: list[float]) -> dict[str, Any]:
    """a = the run, b = the reference, same units in the same order."""
    import numpy as np

    n01 = sum(1 for x, y in zip(a, b) if y == 0 and x == 1)
    n10 = sum(1 for x, y in zip(a, b) if y == 1 and x == 0)
    d = np.asarray([x - y for x, y in zip(a, b)], float)
    if not d.size:
        return {"n": 0}
    if np.all(d == d[0]):
        lo = hi = float(d[0])
    else:
        idx = np.random.default_rng(SEED).integers(0, d.size, (N_BOOT, d.size))
        lo, hi = (float(q) for q in np.quantile(d[idx].mean(1), [0.025, 0.975]))
    return {"n": int(d.size), "ref0_run1": n01, "ref1_run0": n10, "diff": float(d.mean()), "ci_low": lo, "ci_high": hi}


def logit_slopes(x, y):
    """Batched (B, n): slope of logit P(y) = a + b x (Newton, ridge RIDGE on b); NaN where y or x is constant."""
    import numpy as np

    x, y = np.atleast_2d(np.asarray(x, float)), np.atleast_2d(np.asarray(y, float))
    ok = (y.min(1) < y.max(1)) & (x.min(1) < x.max(1))
    w = np.zeros((x.shape[0], 2))
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-np.clip(w[:, :1] + w[:, 1:] * x, -50, 50)))
        r, v = y - p, p * (1 - p)
        g = np.stack([r.sum(1), (r * x).sum(1) - RIDGE * w[:, 1]], 1)
        h01 = (v * x).sum(1)
        H = np.stack([np.stack([v.sum(1), h01], 1), np.stack([h01, (v * x * x).sum(1) + RIDGE], 1)], 1)
        H[~ok] = np.eye(2)
        step = np.linalg.solve(H, g[..., None])[..., 0]
        step[~ok] = 0.0
        w = w + step
        if np.max(np.abs(step)) < 1e-9:
            break
    b = w[:, 1].copy()
    b[~ok] = np.nan
    return b


def slope(x: list[float], y: list[float]) -> dict[str, Any]:
    import numpy as np

    xa, ya = np.asarray(x, float), np.asarray(y, float)
    if xa.size < 2:
        return {"slope": None, "n": int(xa.size)}
    b = float(logit_slopes(xa, ya)[0])
    idx = np.random.default_rng(SEED).integers(0, xa.size, (N_BOOT, xa.size))
    bs = logit_slopes(xa[idx], ya[idx])
    bs = bs[~np.isnan(bs)]
    lo, hi = (float(q) for q in np.quantile(bs, [0.025, 0.975])) if bs.size else (None, None)
    return {"slope": None if b != b else b, "ci_low": lo, "ci_high": hi, "n": int(xa.size), "boot_defined": int(bs.size),
            "leaks": int(round(float(ya.sum())))}  # fmt: skip


# --------------------------------------------------------------------------- inputs


def kind(u: dict) -> str:
    m = u.get("meta") or {}
    return str(m.get("wrong") or m.get("attack") or "-")


def read_dev_units(data: Path) -> dict[str, dict]:
    out = {}
    for line in (Path(data) / "units.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            u = json.loads(line)
            if u.get("split") == "dev" and u.get("turn_type") in KGP:
                out[u["uid"]] = u
    return out


def read_dev_scores(run: Path, units: dict[str, dict]) -> dict[int, dict[str, dict]]:
    """step -> uid -> scores (kgp turns only; reply text dropped; the last record of (step, uid) wins)."""
    p = run / "dev_replies.jsonl"
    if not p.exists():
        raise StopError(f"{p} missing (dev.save_replies)")
    per: dict[int, dict[str, dict]] = defaultdict(dict)
    unknown = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("turn_type") not in KGP:
            continue
        if r["uid"] not in units:
            unknown += 1
            continue
        per[int(r["step"])][r["uid"]] = {k: v for k, v in r.items() if k not in ("reply", "step", "uid")}
    if unknown:
        raise StopError(f"{run}: {unknown} dev_replies kgp records name a uid that is not in the data's dev units (other data)")
    return dict(per)


def similarity(units: dict[str, dict]) -> tuple[dict[str, dict], dict[str, Any]]:
    sims: dict[str, dict] = {}
    check = {"kgp_wrong_units": 0, "parsed": 0, "near_miss_matches_derive": 0, "near_miss_differs": 0,
             "offered_equals_password_casefold": 0}  # fmt: skip
    from kgp_data import derive

    for uid, u in units.items():
        if u["turn_type"] != "kgp_wrong":
            continue
        check["kgp_wrong_units"] += 1
        w = offered_password(u)
        if w is None:
            continue
        check["parsed"] += 1
        k = kind(u)
        if k in NEAR:
            check["near_miss_matches_derive" if w == derive(k, u["password"]) else "near_miss_differs"] += 1
        check["offered_equals_password_casefold"] += int(w.casefold() == u["password"].casefold())
        sims[uid] = {"ratio": ratio(w, u["password"]), "prefix": prefix_len(w, u["password"])}
    return sims, check


# --------------------------------------------------------------------------- tables


def step_table(rows: dict[str, dict], units: dict[str, dict], s0: dict[str, dict] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for t in KGP:
        uids = sorted(u for u, sc in rows.items() if sc.get("turn_type") == t)
        for m in METRICS[t]:
            us = [u for u in uids if m in rows[u]]
            d: dict[str, Any] = {"all": boot([float(rows[u][m]) for u in us])}
            by: dict[str, list[str]] = defaultdict(list)
            for u in us:
                by[kind(units[u])].append(u)
            if set(by) != {"-"}:
                d["by_kind"] = {k: boot([float(rows[u][m]) for u in v]) for k, v in sorted(by.items())}
            if s0 is not None:
                fl = {}
                for k, v in sorted(by.items()):
                    both = [u for u in v if u in s0 and m in s0[u]]
                    fl[k] = {"n": len(both), "step0_0_now_1": sum(1 for u in both if not s0[u][m] and rows[u][m]),
                             "step0_1_now_0": sum(1 for u in both if s0[u][m] and not rows[u][m])}  # fmt: skip
                d["flips_vs_step0_by_kind"] = fl
            out[f"{t}.{m}"] = d
    w = {u: sc for u, sc in rows.items() if sc.get("turn_type") == "kgp_wrong" and "leak_payload" in sc}
    near = [float(sc["leak_payload"]) for u, sc in w.items() if kind(units[u]) in NEAR]
    rnd = [float(sc["leak_payload"]) for u, sc in w.items() if kind(units[u]) == "random"]
    out["kgp_wrong.leak_payload.near_miss_minus_random"] = diff_independent(near, rnd)
    return out


def sim_table(rows: dict[str, dict], sims: dict[str, dict], edges: list[float]) -> dict[str, Any]:
    import numpy as np

    us = sorted(u for u in sims if u in rows and "leak_payload" in rows[u])
    bins: dict[str, Any] = {}
    for b in range(N_BINS):
        ks = [u for u in us if int(np.searchsorted(edges, sims[u]["ratio"], side="right")) == b]
        bins[str(b + 1)] = boot([float(rows[u]["leak_payload"]) for u in ks])
    return {"bins": bins, "slope_ratio": slope([sims[u]["ratio"] for u in us], [float(rows[u]["leak_payload"]) for u in us]),
            "slope_prefix": slope([float(sims[u]["prefix"]) for u in us], [float(rows[u]["leak_payload"]) for u in us])}


def analyse(data: Path, runs: list[tuple[str, Path]], log=print) -> dict[str, Any]:
    import numpy as np

    units = read_dev_units(data)
    if not units:
        raise StopError(f"{data}: no dev kgp units in units.jsonl")
    sims, check = similarity(units)
    r = [sims[u]["ratio"] for u in sorted(sims)]
    edges = [float(q) for q in np.quantile(r, [i / N_BINS for i in range(1, N_BINS)])] if r else []
    kinds = defaultdict(int)
    for u in units.values():
        if u["turn_type"] == "kgp_wrong":
            kinds[kind(u)] += 1
    res: dict[str, Any] = {"note": NOTE, "data": str(data), "dev_kgp_units": len(units), "kgp_wrong_by_kind": dict(sorted(kinds.items())),
                           "offered_password_check": check, "ratio_inner_edges": edges,
                           "ratio_bin_counts": {str(b + 1): sum(1 for x in r if int(np.searchsorted(edges, x, side="right")) == b)
                                                for b in range(N_BINS)}, "runs": {}, "paired": {}}  # fmt: skip
    log(f"dev kgp units {len(units)}; kgp_wrong by kind {json.dumps(res['kgp_wrong_by_kind'])}; offered password {json.dumps(check)}")
    per_run: dict[str, dict[int, dict[str, dict]]] = {}
    for name, run in runs:
        per = read_dev_scores(run, units)
        per_run[name] = per
        sel = json.loads((run / "selection.json").read_text(encoding="utf-8")) if (run / "selection.json").exists() else {}
        s0 = per.get(0)
        res["runs"][name] = {"dir": str(run), "steps": sorted(per), "selected_step": sel.get("selected_step"),
                             "rule_met": sel.get("rule_met"),
                             "by_step": {s: step_table(rows, units, s0 if s else None) for s, rows in sorted(per.items())},
                             "similarity_by_step": {s: sim_table(rows, sims, edges) for s, rows in sorted(per.items())}}  # fmt: skip
        log(f"[{name}] steps {sorted(per)}; selected {sel.get('selected_step')} (rule_met {sel.get('rule_met')})")
    ref = runs[0][0]
    for name in per_run:
        if name == ref:
            continue
        pr: dict[str, Any] = {}
        for s in sorted(set(per_run[name]) & set(per_run[ref])):
            a, b = per_run[name][s], per_run[ref][s]
            us = sorted(u for u in a if u in b and a[u].get("turn_type") == "kgp_wrong" and "leak_payload" in a[u] and "leak_payload" in b[u])
            by: dict[str, list[str]] = defaultdict(list)
            for u in us:
                by[kind(units[u])].append(u)
            pr[s] = {"all": diff_paired([float(a[u]["leak_payload"]) for u in us], [float(b[u]["leak_payload"]) for u in us]),
                     "by_kind": {k: diff_paired([float(a[u]["leak_payload"]) for u in v], [float(b[u]["leak_payload"]) for u in v])
                                 for k, v in sorted(by.items())}}  # fmt: skip
        res["paired"][f"{name}_vs_{ref}"] = {"metric": "kgp_wrong.leak_payload", "by_step": pr}
    return res


def _pct(e: dict | None) -> str:
    if not e or e.get("value") is None:
        return "n/a"
    return f"{100 * e['value']:5.1f} [{100 * e['ci_low']:5.1f},{100 * e['ci_high']:5.1f}] n{e['n']}"


def report(res: dict[str, Any], log=print) -> None:
    for name, r in res["runs"].items():
        log(f"\n== {name}: dev kgp_wrong leak_payload % by wrong-password kind (selected step {r['selected_step']}) ==")
        kinds = sorted(res["kgp_wrong_by_kind"])
        log("  step  " + "  ".join(k.rjust(26) for k in ["all", *kinds]) + "   near-random diff")
        for s, t in r["by_step"].items():
            d = t["kgp_wrong.leak_payload"]
            nm = t["kgp_wrong.leak_payload.near_miss_minus_random"]
            dd = "n/a" if nm.get("diff") is None else f"{100 * nm['diff']:+.1f} [{100 * nm['ci_low']:+.1f},{100 * nm['ci_high']:+.1f}]"
            log(f"  {s:>4}  " + "  ".join(_pct(x).rjust(26) for x in [d["all"], *[(d.get("by_kind") or {}).get(k) for k in kinds]]) + f"   {dd}")
        log(f"  similarity (ratio) bins, inner edges {[round(x, 3) for x in res['ratio_inner_edges']]}, counts {res['ratio_bin_counts']}:")
        for s, t in r["similarity_by_step"].items():
            sl = t["slope_ratio"]
            ss = "undefined" if sl.get("slope") is None else f"{sl['slope']:+.2f} [{sl['ci_low']:+.2f},{sl['ci_high']:+.2f}]"
            log(f"  {s:>4}  " + "  ".join(_pct(t["bins"][b]).rjust(26) for b in sorted(t["bins"])) + f"   slope {ss}")
    for pn, p in res["paired"].items():
        log(f"\n== {pn} ({p['metric']}, paired by dev unit) ==")
        for s, d in p["by_step"].items():
            a = d["all"]
            log(f"  step {s}: all {100 * a['diff']:+.1f} [{100 * a['ci_low']:+.1f},{100 * a['ci_high']:+.1f}] (+{a['ref0_run1']}/-{a['ref1_run0']}, n {a['n']}); "
                + "; ".join(f"{k} +{v['ref0_run1']}/-{v['ref1_run0']} n{v['n']}" for k, v in d["by_kind"].items()))  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="dev kgp_wrong leaks by wrong-password kind (report-only, post-hoc)")
    ap.add_argument("--data", type=Path, required=True, help="the built data dir (units.jsonl)")
    ap.add_argument("--run", action="append", required=True, help="NAME=RUN_DIR; the first is the reference")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    runs = []
    for spec in a.run:
        if "=" not in spec:
            print(f"STOP: --run {spec!r}: use NAME=RUN_DIR")
            return 1
        n, p = spec.split("=", 1)
        runs.append((n, Path(p)))
    try:
        res = analyse(Path(a.data).expanduser(), runs)
    except StopError as e:
        print(f"STOP: {e}")
        return 1
    report(res)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    print(f"written {a.out}\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
