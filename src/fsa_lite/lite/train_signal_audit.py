"""Training-signal audit of an FSA-lite run3 run (H3; REPORT-ONLY, changes nothing): what did training
actually push on, per turn type? CPU only (tokenizer, no model).

Status: ✅ tested on CPU (selftest_run3.py test_train_signal_audit: synthetic units, a char tokenizer,
hand-made replies and a train / dev log written by the test). 🟡 not yet run on the real run3 runs
(train_signal_audit.sbatch on the Slurm cluster for Gemma, train_signal_audit.pbs on the PBS cluster for Qwen).

  python src/fsa_lite/lite/train_signal_audit.py --config configs/training/experiments/fsa_lite_gemma12b_run3.yaml \
      --run-dir outputs/training/fsa_lite_gemma12b_run3 [--data DIR] [--tokenizer ID_OR_PATH] [--out JSON]

Inputs: the built data (manifest + PASSing contamination report checked, as training does), the run's
replies.jsonl (the cached frozen-model replies training used), train_log.jsonl, dev_log.jsonl,
dev_replies.jsonl (if saved) and selection.json. The tokenizer is the config's model at its pinned
revision, offline (it decodes the cached reply ids and applies the same token-length filter).

1. buckets: every cached train reply of every drawn turn type, classified with training's own
   rules (LiteSource.holds with fsa.secret_match, the user's near-miss scrubbed on FSA turns):
     FSA turns (kgp_attack / kgp_wrong): pw_and_payload, pw_only, payload_only, none
     FSA turns of run2's types (tt_attack / cimem_unauthorised): secret, none
     USE turn (kgp_correct): release_without_pw (the USE draw pool with require_no_password),
                             release_with_pw, pw_only (echo, no release), none
   for the FULL train set (not the pilot's sample), split by attack kind (meta.attack), wrong-
   password kind (meta.wrong) and payload type; also the draw pools training used (fsa.units
   leaking_reply / use.units releasing_reply), by bucket.
2. draws: the 500 x grad_accum draws REPLAYED with LiteSource.draw (seeded per step and micro-batch,
   so this is exactly what training drew), counted by turn type and bucket; checked step by step
   against train_log.jsonl (the logged turn types and FSA leak flags must match; mismatches counted).
   The FSA loss (fsa_each) is attributed to each replayed FSA draw and summed by bucket.
3. loss shares (train_log.jsonl): per step, the contribution of each term to the optimised loss
   (FSA: sum of the FSA examples' terms / grad_accum; need: need.weight x need_term x n_need /
   grad_accum; USE: use.weight x use_term x n_use / grad_accum), averaged over 125-step windows.
4. dev (dev_log.jsonl; per-unit CIs and step-0 flips from dev_replies.jsonl when saved): per
   checkpoint, kgp_attack and kgp_wrong leak_pw / leak_payload, kgp_correct use / echo_pw, joint.
Prints only turn types, bucket names, kinds, counts, rates, CIs and losses: never a secret, password,
payload, prompt or reply.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import lite_source as LS  # noqa: E402

N_BOOT, SEED = 2000, 2026
WINDOWS = ((1, 125), (126, 250), (251, 375), (376, 500))
DEV_COLS = (("kgp_attack", "leak_pw"), ("kgp_attack", "leak_payload"), ("kgp_wrong", "leak_pw"),
            ("kgp_wrong", "leak_payload"), ("kgp_correct", "use"), ("kgp_correct", "echo_pw"))  # fmt: skip


# --------------------------------------------------------------------------- classification


def bucket(src: LS.LiteSource, tok: Any, cache: Any, u: dict, r: int) -> str:
    """The reply's bucket, with training's own matching rules (see the module docstring)."""
    text = tok.decode(cache.data[LS.reply_cache_key(u, r)], skip_special_tokens=True)
    if u["objective"] == "fsa":
        text = src.scrub(u, text)
        if not u.get("payload"):
            return "secret" if src.holds(text, u["secret"]) else "none"
        pw, pl = src.holds(text, u["password"]), src.holds(text, u["payload"])
        return "pw_and_payload" if pw and pl else "pw_only" if pw else "payload_only" if pl else "none"
    if u.get("payload"):
        pw, pl = src.holds(text, u["password"]), src.holds(text, u["payload"])
        return "release_with_pw" if pl and pw else "release_without_pw" if pl else "pw_only" if pw else "none"
    return "n/a"


def kind(u: dict) -> str:
    m = u.get("meta") or {}
    return str(m.get("attack") or m.get("wrong") or "-")


def buckets(src: LS.LiteSource, tok: Any, cache: Any) -> tuple[dict, dict[str, str]]:
    """Full train set: per turn type, bucket counts (overall, by kind, by payload type), and the
    draw pools by bucket. Returns (report, {reply key: bucket})."""
    of: dict[str, str] = {}
    rep: dict[str, Any] = {}
    for t in LS.TURN_TYPES:
        if t not in src.weights:
            continue
        total, by_kind, by_ptype = Counter(), defaultdict(Counter), defaultdict(Counter)
        missing = 0
        for u in src.by_type[t]:
            for r in range(src.n_replies(u)):
                key = LS.reply_cache_key(u, r)
                if key not in cache.data:
                    missing += 1
                    continue
                b = of.setdefault(key, bucket(src, tok, cache, u, r))
                total[b] += 1
                by_kind[kind(u)][b] += 1
                by_ptype[str((u.get("meta") or {}).get("payload_type") or "-")][b] += 1
        d: dict[str, Any] = {"units": len(src.by_type[t]), "replies": sum(total.values()), "missing_replies": missing,
                             "buckets": dict(total)}  # fmt: skip
        if len(by_kind) > 1 or "-" not in by_kind:
            d["by_kind"] = {k: dict(v) for k, v in sorted(by_kind.items())}
        if len(by_ptype) > 1 or "-" not in by_ptype:
            d["by_payload_type"] = {k: dict(v) for k, v in sorted(by_ptype.items())}
        pool = src.fsa_pairs.get(t) or src.use_pairs.get(t)
        if pool is not None:
            d["draw_pool"] = {"pairs": len(pool), "buckets": dict(Counter(of[LS.reply_cache_key(u, r)] for u, r in pool))}
        rep[t] = d
    return rep, of


# --------------------------------------------------------------------------- logs


def last_per_step(path: Path) -> dict[int, dict]:
    """jsonl records keyed by step; the last record of a step wins (a resumed run repeats steps)."""
    out: dict[int, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[int(r["step"])] = r
    return out


def replay(src: LS.LiteSource, tok: Any, cache: Any, of: dict[str, str], train: dict[int, dict], steps: int,
           G: int) -> dict[str, Any]:  # fmt: skip
    drawn: dict[str, Counter] = defaultdict(Counter)
    drawn_kind: dict[str, Counter] = defaultdict(Counter)
    fsa_loss: dict[str, list[float]] = defaultdict(list)
    match = Counter()
    mismatched: list[int] = []
    events = Counter()
    for step in range(steps):
        fams, leaks, kinds = [], [], []
        for micro in range(G):
            ex, _ = src.draw(cache, step, micro, tok)
            u = ex.unit
            key = LS.reply_cache_key(u, ex.reply_no)
            b = of.get(key) or bucket(src, tok, cache, u, ex.reply_no)
            drawn[u["turn_type"]][b] += 1
            drawn_kind[f"{u['turn_type']}:{kind(u)}"][b] += 1
            fams.append(u["turn_type"])
            if u["objective"] == "fsa":
                leaks.append(src.leaks(ex))
                kinds.append(f"{u['turn_type']}:{b}")
        for e in src.pop_fake_events():
            events[e["event"]] += 1
        rec = train.get(step + 1)
        if rec is None:
            match["step_not_in_log"] += 1
            continue
        ok = rec.get("families") == fams and rec.get("fsa_leaks") == leaks
        match["match" if ok else "mismatch"] += 1
        if not ok and len(mismatched) < 20:
            mismatched.append(step + 1)
        if ok:
            for k, v in zip(kinds, rec.get("fsa_each") or []):
                fsa_loss[k].append(float(v))
    return {"draws": {t: dict(c) for t, c in sorted(drawn.items())},
            "draws_by_kind": {t: dict(c) for t, c in sorted(drawn_kind.items())},
            "log_check": dict(match), "first_mismatched_steps": mismatched, "fake_events": dict(events),
            "fsa_loss_by_bucket": {k: {"n": len(v), "mean": sum(v) / len(v), "sum": sum(v)} for k, v in sorted(fsa_loss.items())}}  # fmt: skip


def loss_shares(train: dict[int, dict], objective: dict[str, str], cfg: dict) -> dict[str, Any]:
    G = int(cfg["train"]["grad_accum"])
    need_w = float(cfg["need"]["weight"])
    use_w = float((cfg.get("use") or {}).get("weight") or 0.0)
    out: dict[str, Any] = {"definition": "per step: fsa = sum(fsa_each)/G; need = need.weight*need_term*n_need/G; "
                           "use = use.weight*use_term*n_use/G; window = mean over its steps; share = window sum / total",
                           "windows": {}}  # fmt: skip
    for a, b in WINDOWS + ((1, max(train) if train else 0),):
        rows = [train[s] for s in range(a, b + 1) if s in train]
        if not rows:
            continue
        acc = Counter()
        for r in rows:
            fams = r.get("families") or []
            n_fsa = sum(1 for f in fams if objective.get(f) == "fsa")
            n_use = sum(1 for f in fams if objective.get(f) == "use")
            acc["fsa"] += sum(r.get("fsa_each") or []) / G
            acc["need"] += need_w * (r.get("need_term") or 0.0) * (len(fams) - n_fsa) / G
            acc["use"] += use_w * (r.get("use_term") or 0.0) * n_use / G if use_w else 0.0
            acc["fsa_term"] += r.get("fsa_term") or 0.0
            acc["need_term"] += r.get("need_term") or 0.0
            acc["use_term"] += r.get("use_term") or 0.0
            acc["n_fsa"] += n_fsa
            acc["n_use"] += n_use
        tot = acc["fsa"] + acc["need"] + acc["use"]
        n = len(rows)
        out["windows"][f"{a}-{b}"] = {
            "steps": n, "mean_contribution": {k: acc[k] / n for k in ("fsa", "need", "use")},
            "share": {k: (acc[k] / tot if tot else None) for k in ("fsa", "need", "use")},
            "mean_term": {k: acc[k] / n for k in ("fsa_term", "need_term", "use_term")},
            "fsa_examples": acc["n_fsa"], "use_examples": acc["n_use"]}  # fmt: skip
    return out


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


def dev_table(run: Path, units_by_uid: dict[str, dict]) -> dict[str, Any]:
    dev = last_per_step(run / "dev_log.jsonl")
    out: dict[str, Any] = {"steps": sorted(dev), "from_dev_log": {}}
    for s, r in sorted(dev.items()):
        row = {f"{t}.{m}": (r.get(t) or {}).get(m) for t, m in DEV_COLS}
        row.update({f"{t}.n": (r.get(t) or {}).get("n") for t in ("kgp_attack", "kgp_wrong", "kgp_correct")})
        row["kgp.joint"] = (r.get("kgp") or {}).get("joint")
        out["from_dev_log"][s] = row
    p = run / "dev_replies.jsonl"
    if not p.exists():
        out["per_unit"] = None
        return out
    per: dict[int, dict[str, dict]] = defaultdict(dict)  # step -> uid -> scores (reply text never kept)
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            per[int(r["step"])][r["uid"]] = {k: v for k, v in r.items() if k not in ("reply", "step", "uid")}
    cis: dict[int, Any] = {}
    flips: dict[int, Any] = {}
    s0 = per.get(0, {})
    for s, rows in sorted(per.items()):
        c: dict[str, Any] = {}
        f: dict[str, Any] = {}
        for t, m in DEV_COLS:
            uids = sorted(u for u, sc in rows.items() if sc.get("turn_type") == t and m in sc)
            c[f"{t}.{m}"] = boot([float(rows[u][m]) for u in uids])
            by_k: dict[str, list[float]] = defaultdict(list)
            for u in uids:
                by_k[kind(units_by_uid.get(u, {}))].append(float(rows[u][m]))
            if len(by_k) > 1:
                c[f"{t}.{m}.by_kind"] = {k: boot(v) for k, v in sorted(by_k.items())}
            if s and s0:
                both = [u for u in uids if u in s0 and m in s0[u]]
                f[f"{t}.{m}"] = {"n": len(both), "step0_0_now_1": sum(1 for u in both if not s0[u][m] and rows[u][m]),
                                 "step0_1_now_0": sum(1 for u in both if s0[u][m] and not rows[u][m])}  # fmt: skip
        cis[s] = c
        if f:
            flips[s] = f
    out["per_unit"] = {"ci": cis, "flips_vs_step0": flips, "unit": "dev unit (kgp turn)", "resamples": N_BOOT, "seed": SEED}
    return out


# --------------------------------------------------------------------------- driver


def audit(cfg: dict, run: Path, tok: Any, log=print) -> dict[str, Any]:
    from v2_train import ReplyCache  # noqa: PLC0415

    src = LS.LiteSource(cfg, pilot=False)  # manifest hashes + PASSing contamination report checked
    log(f"data units_sha256 {src.manifest.get('units_sha256', '')[:16]} pools_sha256 {src.manifest.get('pools_sha256', '')[:16]}")
    src.prepare(tok)
    cache = ReplyCache(run / "replies.jsonl")
    reply_stats = src.after_replies(cache, tok)  # sets the leaking / releasing draw pools as training did
    rep, of = buckets(src, tok, cache)
    train = last_per_step(run / "train_log.jsonl")
    steps = max(train) if train else int(cfg["train"]["steps"])
    G = int(cfg["train"]["grad_accum"])
    log(f"train_log: {len(train)} steps logged; replaying {steps} x {G} draws")
    rp = replay(src, tok, cache, of, train, steps, G)
    objective = {t: us[0]["objective"] for t, us in src.by_type.items() if us}
    sel = json.loads((run / "selection.json").read_text(encoding="utf-8")) if (run / "selection.json").exists() else {}
    return {"note": "report-only training-signal audit (H3); counts only", "config": cfg["name"],
            "data_units_sha256": src.manifest.get("units_sha256"), "token_dropped": src.token_dropped,
            "reply_stats_as_training": reply_stats, "buckets_full_train_set": rep, "replay": rp,
            "loss_shares": loss_shares(train, objective, cfg),
            "dev": dev_table(run, {u["uid"]: u for u in src.units}),
            "selection": {k: sel.get(k) for k in ("selected_step", "rule_met", "metric", "adapter_sha256", "fake_fallback", "fake_skip") if k in sel}}  # fmt: skip


def _pct(x: Any) -> str:
    return "  n/a" if x is None else f"{100 * x:5.1f}"


def report(res: dict[str, Any], log=print) -> None:
    log(f"\n== {res['config']}: buckets of cached base replies, FULL train set (counts) ==")
    for t, d in res["buckets_full_train_set"].items():
        log(f"  {t}: replies {d['replies']} (missing {d['missing_replies']}) {json.dumps(d['buckets'], sort_keys=True)}")
        for k, v in (d.get("by_kind") or {}).items():
            log(f"      kind {k}: {json.dumps(v, sort_keys=True)}")
        if "draw_pool" in d:
            log(f"      draw pool {d['draw_pool']['pairs']}: {json.dumps(d['draw_pool']['buckets'], sort_keys=True)}")
    rp = res["replay"]
    log(f"\n== draws replayed (log check {json.dumps(rp['log_check'])}; fake events {json.dumps(rp['fake_events'])}) ==")
    if rp["first_mismatched_steps"]:
        log(f"  first mismatched steps: {rp['first_mismatched_steps']}")
    for t, c in rp["draws"].items():
        log(f"  {t}: {json.dumps(c, sort_keys=True)}")
    log("  FSA loss by bucket (n, mean): " + "; ".join(f"{k} {v['n']} {v['mean']:.4f}" for k, v in rp["fsa_loss_by_bucket"].items()))
    log("\n== loss shares (mean contribution per step; share of the optimised loss) ==")
    for w, d in res["loss_shares"]["windows"].items():
        mc, sh = d["mean_contribution"], d["share"]
        log(f"  steps {w}: fsa {mc['fsa']:.4f} ({_pct(sh['fsa'])}%)  need {mc['need']:.4f} ({_pct(sh['need'])}%)  "
            f"use {mc['use']:.4f} ({_pct(sh['use'])}%)  | fsa ex {d['fsa_examples']} use ex {d['use_examples']}")  # fmt: skip
    log("\n== dev by checkpoint (rates %) ==")
    head = "  step " + " ".join(f"{t.replace('kgp_', '')}.{m}".rjust(18) for t, m in DEV_COLS) + "  joint"
    log(head)
    for s, row in res["dev"]["from_dev_log"].items():
        log(f"  {s:>4} " + " ".join(_pct(row[f'{t}.{m}']).rjust(18) for t, m in DEV_COLS) + f"  {_pct(row['kgp.joint'])}")
    pu = res["dev"].get("per_unit")
    if pu:
        for s, f in pu["flips_vs_step0"].items():
            log(f"  flips vs step 0 at step {s}: " + "; ".join(f"{k} +{v['step0_0_now_1']}/-{v['step0_1_now_0']} (n {v['n']})"
                                                         for k, v in f.items()))  # fmt: skip
    log(f"\nselection: {json.dumps(res['selection'])}")


def main(argv: list[str] | None = None) -> int:
    import yaml

    from preflight_fakes import load_tokenizer  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None, help="built data dir (default: the config's data.dir)")
    ap.add_argument("--tokenizer", default=None, help="tokenizer id or path instead of the config's model (local test)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    if a.data is not None:
        cfg["data"]["dir"] = str(a.data)
    for f in ("replies.jsonl", "train_log.jsonl"):
        if not (a.run_dir / f).exists():
            print(f"STOP: {a.run_dir / f} missing")
            return 1
    tok = load_tokenizer(cfg["model"]["hf_id"], cfg["model"].get("revision"), a.tokenizer)
    res = audit(cfg, a.run_dir, tok)
    report(res)
    out = a.out or Path("outputs/diagnostics/train_signal_audit") / f"{cfg['name']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    print(f"written {out}\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
