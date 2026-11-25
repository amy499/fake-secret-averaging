"""Preflight of the run3 fake draws: for every slot unit (kgp_attack / kgp_wrong, train and dev), N
seeded joint fake draws exactly as lite_source.fakes_for_slots makes them, and the valid rates.

Status: ✅ run locally on the built run3 data (Qwen2.5 tokenizer as a stand-in) and in selftest_run3.py;
🟡 not yet run with the Gemma tokenizer (preflight_fakes.sbatch on the Slurm cluster).

  python src/fsa_lite/lite/preflight_fakes.py --config configs/training/experiments/fsa_lite_gemma12b_run3.yaml
      [--n 200] [--data DIR] [--run-dir RUN] [--tokenizer ID_OR_PATH] [--out JSON]

The validity check (fake_mapping_ok: after substitution no real value is left and every fake is
present) does not use the tokenizer; the tokenizer decides which units the model trains on
(LiteSource.prepare drops prompts longer than data.max_seq_tokens - max_new_tokens), so it is loaded
offline (HF_HUB_OFFLINE=1: local cache only) at the config's pinned revision. --run-dir (a run with
replies.jsonl) also marks the units in the leaking-reply draw pool, the only ones training draws.

Per unit:
  tuple_rate  share of the N x K drawn tuples that pass fake_mapping_ok
  draw_rate   share of the N joint draws whose K tuples ALL pass (what fakes_for_slots needs)
  p_old_crash (1 - draw_rate) ** FAKE_TRIES: the chance that a draw of this unit raised before the fix
  fallback    for units with draw_rate < 0.2: whether lite_source.fallback_tuples finds K tuples
              (yes: fake_fallback; no: fake_skip)
Prints only uids, turn types, slot pools / derive kinds, value lengths and token counts: never a
secret, password or payload.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import lite_source as LS  # noqa: E402

LOW = 0.2


def load_tokenizer(hf_id: str, revision: str | None, override: str | None):
    from transformers import AutoTokenizer

    kw: dict = {}
    if override:
        return AutoTokenizer.from_pretrained(override, local_files_only=True)
    try:  # as v2_train.load_pinned (token and cache dir from the FSA settings)
        from fsa_runtime.settings import load_settings

        s = load_settings()
        kw = {"token": s.hf_token, "cache_dir": s.hf_home}
    except Exception:  # noqa: BLE001 - outside the FSA package: the default HF cache
        kw = {}
    if revision:
        kw["revision"] = revision
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        kw["local_files_only"] = True
    return AutoTokenizer.from_pretrained(hf_id, **kw)


def unit_rates(src: LS.LiteSource, u: dict, n: int, seed: int, idx: int) -> dict:
    """N joint draws as fakes_for_slots makes them (one try each), seeded per unit."""
    import numpy as np

    from kgp_data import derive

    rng = np.random.default_rng([seed, 9191, idx])
    k, slots = src.k, u["slots"]
    ok_tuples = ok_draws = 0
    for _ in range(n):
        cols: list = []
        for i, sl in enumerate(slots):
            cols.append(LS.VD.sample_fakes(rng, list(enumerate(src._slot_candidates(u, i))), sl["value"], k)
                        if sl.get("pool") else None)  # fmt: skip
        for i, sl in enumerate(slots):
            if cols[i] is None:
                cols[i] = [derive(sl["derive"], f) for f in cols[0]]
        good = [src.fake_mapping_ok(u, list(t)) for t in zip(*cols)]
        ok_tuples += sum(good)
        ok_draws += all(good)
    draw_rate = ok_draws / n
    return {"tuple_rate": round(ok_tuples / (n * k), 4), "draw_rate": round(draw_rate, 4),
            "p_old_crash": float(f"{(1 - draw_rate) ** LS.FAKE_TRIES:.3g}")}  # fmt: skip


def describe(src: LS.LiteSource, u: dict, tok) -> dict:
    """Shape of the unit without its values."""
    slots = []
    for i, sl in enumerate(u["slots"]):
        v = sl["value"]
        d = {"slot": i, "pool": sl.get("pool"), "derive": sl.get("derive"), "chars": len(v),
             "last_char_is_word": bool(v) and (v[-1].isalnum() or v[-1] == "_")}  # fmt: skip
        if tok is not None:
            d["tokens"] = len(tok.encode(v, add_special_tokens=False))
        if sl.get("pool"):
            d["candidates"] = len(src._slot_candidates(u, i))
        slots.append(d)
    m = u.get("meta") or {}
    return {"uid": u["uid"], "split": u["split"], "turn_type": u["turn_type"], "payload_type": m.get("payload_type"),
            "pw_source": m.get("pw_source"), "slots": slots}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    import numpy as np
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--data", type=Path, default=None, help="built data dir (default: the config's data.dir)")
    ap.add_argument("--run-dir", type=Path, default=None, help="a run with replies.jsonl: mark the draw pool")
    ap.add_argument("--tokenizer", default=None, help="tokenizer id or path instead of the config's model (local test)")
    ap.add_argument("--no-tokenizer", action="store_true", help="skip the token-length filter (rates only)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    if a.data is not None:
        cfg["data"]["dir"] = str(a.data)
    src = LS.LiteSource(cfg, pilot=False)  # checks the manifest hashes and the PASSing contamination report
    print(f"data {cfg['data']['dir']} units_sha256 {src.manifest.get('units_sha256', '')[:16]} "
          f"pools_sha256 {src.manifest.get('pools_sha256', '')[:16]} | K {src.k} FAKE_TRIES {LS.FAKE_TRIES} N {a.n}", flush=True)  # fmt: skip
    tok = None
    in_train: set[str] | None = None
    if not a.no_tokenizer:
        tok = load_tokenizer(cfg["model"]["hf_id"], cfg["model"].get("revision"), a.tokenizer)
        print(f"tokenizer {a.tokenizer or cfg['model']['hf_id']} (revision {cfg['model'].get('revision')})", flush=True)
        src.prepare(tok)
        in_train = {u["uid"] for u in src.train}
        print("token-length filter dropped:", json.dumps(src.token_dropped), flush=True)
    draw_pool: dict[str, int] | None = None
    if a.run_dir is not None and tok is not None:
        from v2_train import ReplyCache  # noqa: PLC0415

        cache = ReplyCache(a.run_dir / "replies.jsonl")
        src.after_replies(cache, tok)
        draw_pool = {}
        for t, pairs in src.fsa_pairs.items():
            for u, _ in pairs:
                draw_pool[u["uid"]] = draw_pool.get(u["uid"], 0) + 1
    units = [u for u in src.units if u.get("slots") and u["objective"] == "fsa"]
    rows = []
    for idx, u in enumerate(units):
        row = {**describe(src, u, tok), **unit_rates(src, u, a.n, int(cfg["seed"]), idx)}
        if in_train is not None:
            row["trained_on"] = u["uid"] in in_train if u["split"] == "train" else None
        if draw_pool is not None:
            row["draw_pool_replies"] = draw_pool.get(u["uid"], 0)
        if row["draw_rate"] < LOW:
            fb = src.fallback_tuples(np.random.default_rng([int(cfg["seed"]), 9292, idx]), u)
            row["fallback"] = "fake_fallback" if fb is not None else "fake_skip"
        rows.append(row)
        if (idx + 1) % 200 == 0:
            print(f"  {idx + 1}/{len(units)} units", flush=True)

    def short(r: dict) -> str:
        sl = " ".join(f"[{s['pool'] or 'derive:' + str(s['derive'])} {s['chars']}ch"
                      + (f" {s['tokens']}tok" if "tokens" in s else "") + ("" if s["last_char_is_word"] else " ends-nonword") + "]"
                      for s in r["slots"])  # fmt: skip
        extra = "".join(f" {k}={r[k]}" for k in ("trained_on", "draw_pool_replies", "fallback") if k in r)
        return (f"  {r['uid']} ({r['split']}, {r['payload_type']}) tuple_rate {r['tuple_rate']} draw_rate {r['draw_rate']} "
                f"p_old_crash {r['p_old_crash']}{extra} {sl}")  # fmt: skip

    zero = [r for r in rows if r["tuple_rate"] == 0]
    low = [r for r in rows if 0 < r["tuple_rate"] < LOW]
    low_draw = [r for r in rows if r["draw_rate"] < LOW]
    tot: dict = {}
    for r in rows:
        key = f"{r['split']}:{r['turn_type']}"
        t = tot.setdefault(key, {"units": 0, "tuple_rate_mean": 0.0, "draw_rate_mean": 0.0, "tuple_rate_0": 0,
                                 "tuple_rate_lt_0.2": 0, "draw_rate_lt_0.2": 0, "p_old_crash_gt_0.01": 0})  # fmt: skip
        t["units"] += 1
        t["tuple_rate_mean"] += r["tuple_rate"]
        t["draw_rate_mean"] += r["draw_rate"]
        t["tuple_rate_0"] += r["tuple_rate"] == 0
        t["tuple_rate_lt_0.2"] += r["tuple_rate"] < LOW
        t["draw_rate_lt_0.2"] += r["draw_rate"] < LOW
        t["p_old_crash_gt_0.01"] += r["p_old_crash"] > 0.01
    for t in tot.values():
        t["tuple_rate_mean"] = round(t["tuple_rate_mean"] / t["units"], 4)
        t["draw_rate_mean"] = round(t["draw_rate_mean"] / t["units"], 4)
    print(f"\nunits with tuple_rate 0 ({len(zero)}):")
    for r in zero:
        print(short(r))
    print(f"units with 0 < tuple_rate < {LOW} ({len(low)}):")
    for r in low:
        print(short(r))
    print(f"units with draw_rate < {LOW} (all K tuples valid in under {LOW:.0%} of joint draws; {len(low_draw)}):")
    for r in low_draw:
        print(short(r))
    fb = [r.get("fallback") for r in low_draw]
    print("fallback outcome for those:", json.dumps({k: fb.count(k) for k in ("fake_fallback", "fake_skip")}))
    print("TOTALS:", json.dumps(tot), flush=True)
    if a.out is not None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"n": a.n, "k": src.k, "fake_tries": LS.FAKE_TRIES, "totals": tot, "units": rows},
                                    indent=1) + "\n", encoding="utf-8", newline="\n")  # fmt: skip
        print(f"written {a.out}")
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
