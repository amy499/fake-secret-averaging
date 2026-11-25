"""Diagnose an FSA-lite run from its own files (dev and train data only; no evaluation benchmark).

Status: 🟡 the CPU part (counts, draws, train-log join, adapter weights) is tested on CPU with
synthetic data (selftest_v2.py); the --generate part needs the GPU and the run's adapter.

  python src/fsa_lite/lite/lite_diagnose.py --run-dir outputs/training/<run> [--generate --dev-n 112] [--device cuda]
On the Slurm cluster inside a job only: sbatch --export=ALL,RUN_DIR=outputs/training/<run>,GENERATE=1 scripts/slurm/lite_diagnose.sbatch

CPU part (the run's config.yaml, replies.jsonl, train_log.jsonl, dev_log.jsonl, adapter/):
  * per FSA turn type: train units, cached base replies, and how many hold the real secret
    (exact, case-insensitive), plus units with at least one such reply
  * the training draws, recomputed exactly (seeded per step and micro-batch): how many FSA examples
    of each type were drawn, how many of them had a leaking reply, how many distinct units
  * train_log joined to the draws: the FSA term of steps whose FSA examples had a leaking reply vs
    steps whose did not (does the signal come only from leaking replies?)
  * dilution: in leaking replies, the share of reply tokens inside the secret (the plain per-token
    mean weighs the code by that share)
  * the adapter: lora_B norms (zero = never trained), the LoRA scale and the per-layer spread
--generate (GPU): the dev evaluation exactly as dev_eval runs it, with the adapter disabled (= the
step-0 model: lora_B starts at zero) and enabled (the final adapter), on the run's own dev units
and on --dev-n dev tt_attack units: leak rates, how many replies are identical, which units leak
under each, and the largest logit change the adapter makes on dev prompts.
Writes <run>/diagnose.json. Dev replies are printed only as counts and unit ids.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FSA_DIR = HERE.parents[2]  # the repository root
sys.path[:0] = [str(HERE), str(FSA_DIR), str(FSA_DIR / "src" / "fsa_lite")]

import lite_source as LS  # noqa: E402
import v2_train as VT  # noqa: E402

FSA_TYPES = ("tt_attack", "cimem_unauthorised")


def load_tokenizer(cfg: dict):
    import os

    from transformers import AutoTokenizer

    from fsa_runtime.settings import load_settings

    s = load_settings()
    kw = {"cache_dir": s.hf_home, "token": s.hf_token}
    if cfg["model"].get("revision"):
        kw["revision"] = cfg["model"]["revision"]
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        kw["local_files_only"] = True
    return AutoTokenizer.from_pretrained(cfg["model"]["hf_id"], **kw)


def draw_counts(src: LS.LiteSource, cache: Any, steps: int, G: int) -> tuple[dict, dict[int, list[bool]]]:
    """The run's draws, recomputed: per type drawn / leaking / distinct; per step the leak flags of
    its FSA examples in micro-batch order (as train_log's fsa values)."""
    out: dict[str, dict] = {}
    per_step: dict[int, list[bool]] = {}
    seen: dict[str, set] = {}
    for step in range(steps):
        flags = []
        for micro in range(G):
            ex, _ = src.draw(cache, step, micro, None)
            t = ex.unit["turn_type"]
            c = out.setdefault(t, {"drawn": 0, "drawn_leaking": 0})
            c["drawn"] += 1
            seen.setdefault(t, set()).add(ex.unit["uid"])
            if ex.unit["objective"] == "fsa":
                hit = bool(src.leak.get(LS.reply_cache_key(ex.unit, ex.reply_no)))
                c["drawn_leaking"] += hit
                flags.append(hit)
        per_step[step + 1] = flags
    for t, c in out.items():
        c["distinct_units"] = len(seen[t])
    return out, per_step


def train_log_join(log_path: Path, per_step: dict[int, list[bool]]) -> dict:
    rows = [json.loads(x) for x in log_path.read_text(encoding="utf-8").splitlines() if x.strip()] if log_path.exists() else []
    with_leak, without = [], []
    for r in rows:
        if r.get("fsa_term") is None or r["step"] not in per_step:
            continue
        flags = per_step[r["step"]]
        each = r.get("fsa_each")
        if each and len(each) == len(flags):  # run2 logs: per example
            for v, f in zip(each, flags):
                (with_leak if f else without).append(v)
        elif flags:  # run1 logs: the step mean only; classify the step
            (with_leak if any(flags) else without).append(r["fsa_term"])

    def summ(v: list[float]) -> dict | None:
        if not v:
            return None
        s = sorted(v)
        return {"n": len(v), "mean": statistics.fmean(v), "median": statistics.median(v), "p10": s[len(s) // 10],
                "p90": s[(9 * len(s)) // 10], "max": s[-1]}  # fmt: skip

    return {"steps_logged": len(rows), "fsa_term_with_a_leaking_reply": summ(with_leak),
            "fsa_term_without": summ(without),
            "unit": "per example" if rows and rows[-1].get("fsa_each") is not None else "per step (any leaking FSA example in the step)"}  # fmt: skip


def dilution(src: LS.LiteSource, cache: Any, tok: Any) -> dict:
    """Share of reply tokens inside the secret, over leaking cached replies of FSA units."""
    saved, src.secret_w = src.secret_w, 2.0
    shares: list[float] = []
    try:
        for t in FSA_TYPES:
            for u in src.by_type.get(t, []):
                for r in range(src.n_replies(u)):
                    key = LS.reply_cache_key(u, r)
                    if not src.leak.get(key):
                        continue
                    ex = LS.LiteExample(u, u["secret"], cache.data[key], reply_no=r)
                    w = src.fsa_token_weights(tok, ex)
                    if w:
                        shares.append(sum(1 for x in w if x != 1.0) / len(w))
    finally:
        src.secret_w = saved
    return {"n_leaking_replies": len(shares), "mean_share_of_tokens_in_secret": statistics.fmean(shares) if shares else None,
            "median_share": statistics.median(shares) if shares else None}  # fmt: skip


def adapter_weights(adapter: Path) -> dict | None:
    f = adapter / "adapter_model.safetensors"
    if not f.exists():
        return None
    from safetensors.torch import load_file

    t = load_file(str(f))
    b = [v.float().norm().item() for k, v in t.items() if "lora_B" in k]
    a = [v.float().norm().item() for k, v in t.items() if "lora_A" in k]
    cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    return {"n_tensors": len(t), "lora_B_norm_mean": statistics.fmean(b), "lora_B_norm_max": max(b),
            "lora_B_zero": sum(1 for x in b if x == 0.0), "lora_A_norm_mean": statistics.fmean(a),
            "r": cfg.get("r"), "alpha": cfg.get("lora_alpha")}  # fmt: skip


def dev_log(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []


def generate_part(src: LS.LiteSource, cfg: dict, run: Path, tok: Any, device: str, dev_n: int) -> dict:
    import torch
    from peft import PeftModel

    from fsa.prompting import eos_ids

    _, model = VT.load_pinned(cfg["model"]["hf_id"], device, cfg["model"]["dtype"], cfg["model"].get("revision"))
    eos = eos_ids(tok, model)
    model = PeftModel.from_pretrained(model, str(run / "adapter"))
    model.eval()
    out: dict[str, Any] = {}
    for label, n in (("run_dev_units", None), (f"tt_attack_n{dev_n}", dev_n)):
        s = copy.copy(src)
        s.cfg = copy.deepcopy(src.cfg)
        if n is not None:
            nt = s.cfg["dev"]["n_per_type"]
            s.cfg["dev"]["n_per_type"] = {**({t: nt for t in LS.DEV_TYPES} if not isinstance(nt, dict) else nt), "tt_attack": n}
        with model.disable_adapter():
            base = s.dev_eval(model, tok, eos, device, 0)
        final = s.dev_eval(model, tok, eos, device, -1)
        rb, rf = base.pop("_replies"), final.pop("_replies")
        same = [a["reply"] == b["reply"] for a, b in zip(rb, rf)]
        att = [(a, b) for a, b in zip(rb, rf) if a["turn_type"] == "tt_attack"]
        out[label] = {
            "adapter_off": {t: base[t] for t in LS.DEV_TYPES if t in base},
            "adapter_on": {t: final[t] for t in LS.DEV_TYPES if t in final},
            "identical_replies": f"{sum(same)}/{len(same)}",
            "identical_tt_attack_replies": f"{sum(a['reply'] == b['reply'] for a, b in att)}/{len(att)}",
            "tt_attack_leaking_uids_off": [a["uid"] for a, _ in att if a["leak"]],
            "tt_attack_leaking_uids_on": [b["uid"] for _, b in att if b["leak"]],
        }  # fmt: skip
    # how much the adapter moves the next-token logits on dev attack prompts
    units = [u for u in src.dev if u["turn_type"] == "tt_attack"][:8]
    deltas = []
    import v2_data as VD

    with torch.no_grad():
        for u in units:
            ids = torch.tensor([VD.prompt_ids(tok, u["messages"])], device=device)
            on = model(input_ids=ids).logits[0, -1].float()
            with model.disable_adapter():
                off = model(input_ids=ids).logits[0, -1].float()
            deltas.append((on - off).abs().max().item())
    out["max_abs_logit_change_last_prompt_token"] = {"n": len(deltas), "mean": statistics.fmean(deltas), "max": max(deltas)}
    return out


def diagnose(run: Path, tok: Any, *, generate: bool = False, device: str = "cuda", dev_n: int = 112,
             cfg: dict | None = None, src: LS.LiteSource | None = None) -> dict:
    import yaml

    cfg = cfg or yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    src = src or LS.LiteSource(cfg, pilot=False)
    src.prepare(tok)
    cache = VT.ReplyCache(run / "replies.jsonl")
    res: dict[str, Any] = {"run": str(run), "config_name": cfg.get("name")}
    res["cached_replies"] = src.after_replies(cache, tok)
    steps = len([1 for x in (run / "train_log.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]) \
        if (run / "train_log.jsonl").exists() else int(cfg["train"]["steps"])  # fmt: skip
    drawn, per_step = draw_counts(src, cache, steps, int(cfg["train"]["grad_accum"]))
    res["draws"] = {"steps": steps, "grad_accum": cfg["train"]["grad_accum"], "per_type": drawn}
    res["train_log"] = train_log_join(run / "train_log.jsonl", per_step)
    res["dilution"] = dilution(src, cache, tok)
    res["adapter"] = adapter_weights(run / "adapter")
    tc = cfg["train"]
    res["optimiser"] = {"lr": tc["lr"], "warmup_steps": tc["warmup_steps"], "schedule": "linear warmup, then constant",
                        "effective_batch": tc["grad_accum"], "max_grad_norm": tc["max_grad_norm"],
                        "weight_decay": tc.get("weight_decay", 0.0),
                        "fsa_reduction": "per-token mean over all reply tokens" if float(cfg["fsa"].get("secret_token_weight") or 1) == 1
                        else f"weighted per-token mean, secret tokens x{cfg['fsa']['secret_token_weight']}"}  # fmt: skip
    res["dev_log"] = dev_log(run / "dev_log.jsonl")
    if generate:
        res["generate"] = generate_part(src, cfg, run, tok, device, dev_n)
    (run / "diagnose.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    return res


def main(argv: list[str] | None = None) -> int:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--dev-n", type=int, default=112)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    run = a.run_dir if a.run_dir.is_absolute() else FSA_DIR / a.run_dir
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    res = diagnose(run, load_tokenizer(cfg), generate=a.generate, device=a.device, dev_n=a.dev_n, cfg=cfg)
    for k in ("cached_replies", "draws", "train_log", "dilution", "adapter", "optimiser"):
        print(f"{k}: {json.dumps(res[k])}")
    for d in res["dev_log"]:
        print("dev", json.dumps({kk: v for kk, v in d.items() if kk != "seconds"}))
    if "generate" in res:
        print("generate:", json.dumps(res["generate"], indent=1))
    print(f"wrote {run / 'diagnose.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
