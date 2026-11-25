"""Cost estimate: forward passes and A40 GPU hours per part, from the config and prompts.

Status: ✅ implemented + tested (printed by every run; tests/test_internals.py,
tests/test_internals_bench.py).

Assumes parts run in order in one run folder (later parts reuse earlier replies). Timing
constants come from `cost:` in config.yaml (README, "Cost"); they are estimates, not measurements.
Which conditions each part uses comes from the prompts file (`parts:`, else the hand-written
defaults). Two optional constants serve longer prompts (the bench set) and leave the hand-written
estimate unchanged when absent:
  chars_per_token (+ template_tokens)  pass time scales with (prompt + reply tokens) /
                                       (prompt_tokens + reply tokens), never below 1
  probe_ref_rows                       probe_s is for this many probe training rows per layer
"""

from __future__ import annotations

import math
from typing import Any

from analysis.internals.parts import D, load_prompts, prompt_source


def estimate(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    c = cfg["cost"]
    pr = load_prompts(prompt_source(cfg), cfg.get("prompts_per_condition"))
    parts = pr["parts"]
    conds, probe, lens = list(parts["secret"]), list(parts["probe"]), parts["lens"]
    n = {k: len(v) for k, v in pr["conds"].items()}
    g = cfg["generation"]
    bs, max_new = int(g["batch_size"]), int(g["max_new_tokens"])
    p1, p2, p3 = cfg["part1"], cfg["part2"], cfg["part3"]
    n_tr, n_te = int(p1["probe"]["n_train"]), int(p1["probe"]["n_test"])
    n_lens, n_att = int(p1["logit_lens_secrets"]), int(p2["n_secrets"])
    fixed = {k: int(v) for k, v in (p1.get("fixed_replies") or {}).items()}
    k1 = {cond: fixed.get(cond, 1) for cond in conds}

    def reply_tokens(cond: str) -> int:
        return int(c["reply_tokens"].get(cond, max(c["reply_tokens"].values())))

    def gen_s(cond: str, n_prompts: int, n_secrets: int) -> float:
        steps = min(max_new, 2 * reply_tokens(cond))  # a batch runs until its longest reply
        return n_prompts * math.ceil(n_secrets / bs) * steps * float(c["step_s"])

    def scale(cond: str) -> float:
        """Pass-time factor for this condition's prompt length (1 without chars_per_token)."""
        if "chars_per_token" not in c:
            return 1.0
        src = [(pid, u, cond) for pid, u in pr["conds"][cond]] if cond != D else pr["D"]
        if not src:
            return 1.0
        sys_i = 0 if cond != D else 1
        toks = [(len(pr["systems"][(s, pid)][sys_i]) + len(u)) / float(c["chars_per_token"])
                + float(c.get("template_tokens", 0)) for pid, u, s in src]  # fmt: skip
        r = reply_tokens(cond)
        return max(1.0, (sum(toks) / len(toks) + r) / (float(c["prompt_tokens"]) + r))

    light, heavy, lens_s = float(c["pass_light_s"]), float(c["pass_heavy_s"]), float(c["lens_s"])
    out_parts = []
    # 1a: fixed replies (part1.fixed_replies per prompt, default 1); n_codes passes per reply (+ final-layer lens).
    g1a = sum(n[x] * k1[x] for x in conds)
    s1a = sum(gen_s(x, n[x], k1[x]) for x in conds)
    f1a = g1a * int(p1["n_codes"])
    t1a = sum(n[x] * k1[x] * int(p1["n_codes"]) * (light + 0.1) * scale(x) for x in conds)
    out_parts.append(("1a", g1a, f1a, s1a + t1a))
    # 1b: probe conditions x (train + test) replies, pooled-feature passes, secret-position control, probes on CPU.
    cap = p1["probe"].get("prompts_per_condition")
    npb = {x: min(n[x], int(cap)) if cap else n[x] for x in probe}  # part1.probe.prompts_per_condition
    g1b = sum(npb[x] for x in probe) * (n_tr + n_te)
    f1b = g1b + n_tr + n_te
    rows = 2 * sum(npb[x] for x in probe) * n_tr + 4 * n_tr  # probe training rows per layer
    probe_secs = float(c["probe_s"]) * rows / float(c.get("probe_ref_rows", rows))
    t1b = sum(npb[x] * (n_tr + n_te) * light * scale(x) for x in probe) + (n_tr + n_te) * light * scale(probe[0])
    out_parts.append(("1b", g1b, f1b, sum(gen_s(x, npb[x], n_tr + n_te) for x in probe) + t1b + probe_secs))
    # 1c: lens replies reused from 1b when n_lens <= n_test; D replies are new; lens over all layers.
    nb = n[lens]
    reused = lens in probe and n_lens <= n_te and npb[lens] == nb
    g1c = nb * n_lens + (0 if reused else nb * n_lens)
    f1c = 2 * nb * n_lens
    s1c = gen_s(D, nb, n_lens) + (0 if reused else gen_s(lens, nb, n_lens))
    out_parts.append(("1c", g1c, f1c, s1c + nb * n_lens * (light + lens_s) * (scale(lens) + scale(D))))
    # 2: new replies unless covered by 1b.
    covered = {x for x in conds if x in probe and n_att <= n_te and npb[x] == n[x]}
    g2 = sum(n[x] * n_att for x in conds if x not in covered)
    s2 = sum(gen_s(x, n[x], n_att) for x in conds if x not in covered)
    f2 = sum(n[x] for x in conds) * n_att
    out_parts.append(("2", g2, f2, s2 + sum(n[x] * n_att * heavy * scale(x) for x in conds)))
    # 3: reuses the 1a replies; n_codes passes per fixed reply + one D pass per D prompt.
    nd = len(pr["D"])
    f3 = g1a * int(p3["n_codes"]) + nd
    t3 = sum(n[x] * k1[x] * int(p3["n_codes"]) * heavy * scale(x) for x in conds) + nd * heavy * scale(D)
    out_parts.append(("3", 0, f3, t3))
    return [{"part": name, "generations": gens, "passes": passes, "gpu_h": secs / 3600}
            for name, gens, passes, secs in out_parts]  # fmt: skip


def by_part(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    agg: dict[str, dict[str, float]] = {}
    for r in rows:
        k = r["part"][0]
        a = agg.setdefault(k, {"generations": 0, "passes": 0, "gpu_h": 0.0})
        for f in ("generations", "passes", "gpu_h"):
            a[f] += r[f]
    return agg


def report(cfg: dict[str, Any]) -> str:
    rows = estimate(cfg)
    c = cfg["cost"]
    load_h, limit = float(c["load_s"]) / 3600, float(c["job_limit_h"])
    lines = [f"Cost estimate (one A40, {cfg['model']['name']}; constants from config `cost:`, estimated for "
             f"Gemma 3 12B bf16; x3 = safety margin)",
             f"{'part':<6}{'generations':>12}{'passes':>9}{'GPU h':>8}{'x3':>7}  job"]  # fmt: skip
    for r in rows:
        lines.append(f"{r['part']:<6}{r['generations']:>12}{r['passes']:>9}{r['gpu_h']:>8.2f}{3 * r['gpu_h']:>7.2f}")
    for k, a in by_part(rows).items():
        h = a["gpu_h"] + load_h
        jobs = max(1, math.ceil(3 * h / limit))
        fits = "fits one 6 h job" if jobs == 1 else f"split into {jobs} resumable jobs (--max-hours)"
        lines.append(f"{'P' + k:<6}{int(a['generations']):>12}{int(a['passes']):>9}{h:>8.2f}{3 * h:>7.2f}  {fits}")
    if "chars_per_token" in c:  # bench configs: one job for all parts?
        tot = sum(r["gpu_h"] for r in rows) + load_h
        verdict = "fits ONE 6 h job" if 3 * tot <= limit else ("fits one 6 h job without the x3 margin"
                                                              if tot <= limit else "does NOT fit one 6 h job")  # fmt: skip
        lines.append(f"{'all':<6}{'':>12}{'':>9}{tot:>8.2f}{3 * tot:>7.2f}  parts 1+2+3 in one job: {verdict}")
    return "\n".join(lines)
