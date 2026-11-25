"""FSA-lite training: frozen-model replies, timing pilot, resumable training, dev evaluation and the
pre-registered checkpoint selection, for the FSA-lite data source (lite_source.LiteSource).

Status: 🟡 the data, masking and loss path is tested on CPU with a tiny random model
(src/fsa_lite/lite/selftest_v2.py, selftest_run3.py); run2 / run3 adapters were trained with it on GPUs.

  python src/fsa_lite/lite/v2_train.py --config configs/training/experiments/fsa_lite_gemma12b_run3.yaml --mode pilot
  python src/fsa_lite/lite/v2_train.py --config ... --mode train [--run-dir outputs/training/<run>] [--max-minutes 330]
Run from the repository root with the training environment (uv run --group ml), on a GPU node; the
job scripts are scripts/slurm/lite_*.sbatch and scripts/pbs/lite_*.pbs.

Data (config data.source: lite): Tensor Trust, CIMemories, benign and (run3) key-guarded-payload
units built by build_lite.py / build_lite_run3.py; mix.weights per turn type, K fakes drawn fresh at
every draw, dev metrics every dev.every steps (dev_log.jsonl).
Models (config model_key): gemma or qwen; model.hf_id / revision / dtype from the config. A null
revision records the HF cache's refs/main instead of checking it.

Phases:
  1. Model: bf16; the HF cache refs/main must equal the configured revision.
  2. Replies: the frozen model's sampled replies under each real context (temperature 1, top_k 0,
     top_p 1, repetition_penalty 1: fsa.prompting.SAMPLING; up to and including EOS), cached in
     <run>/replies.jsonl and reused on resume.
  3. LoRA (PEFT, the config's r / alpha / target modules / layers, grad checkpointing).
  4. Steps: grad_accum micro-batches of one example drawn by the source; K fakes per FSA example.
     Checkpoint every checkpoint_every steps and before --max-minutes runs out (exit code 3);
     --run-dir with an existing checkpoint resumes from it (the per-step RNG makes it exact).
  5. Saves <run>/adapter (PEFT) with training_info.json and <run>/v2_meta.json; with a config
     `selection` block (run3), <run>/selection.json and adapter/ = the checkpoint the rule chose.
Pilot: the config's pilot section; prints sec/step and the projected full-run time, writes
<run>/pilot_timing.json, saves no adapter.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import shutil
import stat
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FSA_DIR = HERE.parents[2]  # the repository root
sys.path[:0] = [str(HERE), str(FSA_DIR), str(FSA_DIR / "src" / "fsa_lite")]


EXIT_UNFINISHED = 3


# ----------------------------------------------------------------------------- inputs



def git_head(repo: Path) -> str:
    """HEAD commit without git (as scripts/lib/git_head.sh)."""
    g = repo / ".git"
    h = (g / "HEAD").read_text().strip()
    if not h.startswith("ref:"):
        return h
    ref = h.split(" ", 1)[1]
    if (g / ref).exists():
        return (g / ref).read_text().strip()
    for line in (g / "packed-refs").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == ref:
            return parts[0]
    raise RuntimeError(f"cannot resolve {ref} in {g}")






def cached_revision(hf_id: str) -> str | None:
    """The HF cache's refs/main commit for hf_id (settings hf_home, then ~/.cache/huggingface/hub)."""
    from fsa_runtime.settings import load_settings

    cache = load_settings().hf_home
    cands = [Path(os.path.expanduser(cache))] if cache else []
    cands.append(Path.home() / ".cache/huggingface/hub")
    name = "models--" + hf_id.replace("/", "--")
    for c in cands:
        ref = c / name / "refs/main"
        if ref.exists():
            return ref.read_text().strip()
    return None


def check_revision(hf_id: str, revision: str | None) -> str:
    """Raise unless the cache holds the configured revision; with revision None, return the cached
    one (recorded in training_info.json and the run meta)."""
    got = cached_revision(hf_id)
    if got is None:
        raise RuntimeError(f"no cached models--{hf_id.replace('/', '--')}/refs/main to check the revision against")
    if revision is not None and got != revision:
        raise RuntimeError(f"{hf_id} cache revision {got} != configured {revision}")
    return got


# ----------------------------------------------------------------------------- replies

class ReplyCache:
    def __init__(self, path: Path):
        self.path, self.data = path, {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.data[r["key"]] = r["reply"]

    def add(self, key: str, reply: list[int]) -> None:
        self.data[key] = reply
        new = not self.path.exists()
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"key": key, "reply": reply}) + "\n")
        if new and os.name != "nt":
            os.chmod(self.path, 0o600)


def generate_batch(model, tok, prompts: list[list[int]], gen: dict, eos: set, seed: int, device) -> list[list[int]]:
    import torch

    from fsa.prompting import SAMPLING, until_eos

    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    T = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), T), pad, dtype=torch.long)
    attn = torch.zeros_like(ids)
    for b, p in enumerate(prompts):  # left padding
        ids[b, T - len(p):] = torch.tensor(p)
        attn[b, T - len(p):] = 1
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(input_ids=ids.to(device), attention_mask=attn.to(device), do_sample=True,
                             temperature=gen["temperature"], max_new_tokens=gen["max_new_tokens"],
                             pad_token_id=pad, **SAMPLING)  # fmt: skip
    return [until_eos(row, eos) for row in out[:, T:].tolist()]






# ----------------------------------------------------------------------------- training





@contextlib.contextmanager
def frozen_model(model):
    import torch

    was = model.training
    model.eval()
    try:
        with model.disable_adapter(), torch.no_grad():
            yield model
    finally:
        if was:
            model.train()


def example_step(model, tok, src, cfg: dict, ex, prev: list[str], device) -> dict:
    """Forward + loss for one example (backward is the caller's). Returns tensors and counts.
    src: the data source (contexts and masks for its examples)."""
    import v2_objective as VO

    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    real, fakes = src.contexts(tok, ex, prev)
    fmask, nmask = src.masks(tok, ex)
    fsa_t = need_t = None
    with frozen_model(model) as fr:
        if any(fmask):
            fsa_t = VO.mixture_reply_target(fr, fakes, ex.reply, pad, device, cfg["train"]["fake_batch"])
        if any(nmask):
            need_t = VO.frozen_reply_target(fr, real, ex.reply, pad, device)
    student = VO.reply_logits(model, [real], ex.reply, pad, device)[0]
    weights = src.fsa_token_weights(tok, ex) if (any(fmask) and hasattr(src, "fsa_token_weights")) else None
    total, f, n = VO.example_loss(student, fsa_t, need_t, fmask, nmask, cfg["need"]["weight"], fsa_weights=weights)
    # run3 USE term (use.weight > 0, use units only; None otherwise, so run1 / run2 are unchanged)
    uw = src.use_token_weights(tok, ex) if hasattr(src, "use_token_weights") else None
    u_term = None
    if uw is not None:
        u_term = VO.use_nll(student, ex.reply, uw)
        total = total + float(cfg["use"]["weight"]) * u_term
    return {"total": total, "fsa": f, "need": n, "use": u_term, "n_fsa": sum(fmask), "n_need": sum(nmask),
            "n_weighted": 0 if weights is None else sum(1 for w in weights if w != 1.0)}  # fmt: skip


def attach_lora(model, cfg: dict):
    import torch
    from peft import LoraConfig, get_peft_model

    from fsa.training.lora import adapted_layers, n_decoder_layers, resolve_layers, target_regex

    lc = cfg["lora"]
    layers = resolve_layers(lc["layers"], n_decoder_layers(model), cfg["seed"])
    tm = target_regex(layers, lc["target_modules"])
    if cfg["train"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    torch.manual_seed(cfg["seed"] * 1000 + 7)
    model = get_peft_model(model, LoraConfig(r=lc["rank"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=tm, bias="none", task_type="CAUSAL_LM"))  # fmt: skip
    if adapted_layers(model) != set(layers):
        raise ValueError("LoRA landed on unexpected layers")
    return model, layers, tm


def save_adapter(model, path: Path, info: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(tmp)
    (tmp / "training_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8", newline="\n")
    if path.exists():
        shutil.rmtree(path)
    os.replace(tmp, path)


def latest_checkpoint(run: Path) -> Path | None:
    c = sorted((run / "checkpoints").glob("step_*/state.pt"))
    return c[-1].parent if c else None




def make_source(cfg: dict, a, *, pilot: bool):
    kind = (cfg.get("data") or {}).get("source", "lite")
    if kind == "lite":
        import lite_source

        return lite_source.LiteSource(cfg, pilot=pilot)
    raise SystemExit(f"data.source {kind!r}: only the FSA-lite source (lite) is in this repository")


MODEL_KEYS = {"gemma": "google/gemma-3-12b-it", "qwen": "Qwen/Qwen2.5-7B-Instruct"}


def load_pinned(hf_id: str, device: str, dtype: str, revision: str | None):
    """scripts.train_fsa.load_model, plus the revision passed to from_pretrained (so the pinned commit
    is what loads, not only what refs/main names) and local files only when HF_HUB_OFFLINE is set."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from fsa_runtime.generators.hf import resolve_dtype
    from fsa_runtime.settings import load_settings

    settings = load_settings()
    kw = {"token": settings.hf_token, "cache_dir": settings.hf_home}
    if revision:
        kw["revision"] = revision
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        kw["local_files_only"] = True
    tok = AutoTokenizer.from_pretrained(hf_id, **kw)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=getattr(torch, resolve_dtype(dtype, device)),
                                                 device_map=device, **kw)  # fmt: skip
    return tok, model


def loaded_facts(model) -> dict:
    """What actually loaded: the commit (config._commit_hash) and the parameter dtype."""
    return {"revision_loaded": getattr(model.config, "_commit_hash", None),
            "dtype_loaded": str(next(model.parameters()).dtype).replace("torch.", "")}  # fmt: skip


def repo_commit() -> str | None:
    """HEAD of the FSA repo, read from .git (the PBS cluster compute nodes have no git binary)."""
    try:
        return git_head(FSA_DIR)
    except (OSError, RuntimeError, IndexError):
        return None


def _round(x: Any) -> Any:
    """Numbers rounded to 3 places inside nested dicts (run3's kgp block has by_pw_source and None)."""
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    return round(x, 3) if isinstance(x, float) else x


def run_dev_eval(src, model, tok, eos, device, step: int, run: Path) -> dict | None:
    if getattr(src, "dev_eval", None) is None or not src.cfg.get("dev"):
        return None
    res = src.dev_eval(model, tok, eos, device, step)
    replies = res.pop("_replies", None)
    with (run / "dev_log.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(res) + "\n")
    if replies is not None and (src.cfg.get("dev") or {}).get("save_replies"):
        with (run / "dev_replies.jsonl").open("a", encoding="utf-8", newline="\n") as f:
            for r in replies:
                f.write(json.dumps({"step": step, **r}, ensure_ascii=False) + "\n")
    short = {t: _round(m) for t, m in res.items() if isinstance(m, dict)}
    print(f"dev step {step}: {json.dumps(short)} ({res['seconds']:.0f}s)", flush=True)
    return res


def main(argv: list[str] | None = None) -> int:
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--mode", choices=("pilot", "train"), required=True)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--max-minutes", type=float, default=330)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    started = time.monotonic()
    os.umask(0o077)
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    mkey = cfg.get("model_key", "gemma")
    if mkey not in MODEL_KEYS:
        raise SystemExit(f"model_key {mkey!r}: use one of {sorted(MODEL_KEYS)}")
    if cfg["model"]["hf_id"] != MODEL_KEYS[mkey]:
        raise SystemExit(f"model_key {mkey} expects {MODEL_KEYS[mkey]}, config has {cfg['model']['hf_id']}")
    pilot = a.mode == "pilot"
    run = a.run_dir or (FSA_DIR / "outputs/training" / f"{datetime.now():%Y%m%d-%H%M%S}_{cfg['name']}{'_pilot' if pilot else ''}")
    run.mkdir(parents=True, exist_ok=True)
    if (run / "config.yaml").exists():  # resume: refuse another config's run folder
        old = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8")) or {}
        if old.get("name") != cfg["name"]:
            raise SystemExit(f"{run} holds a run of config {old.get('name')!r}, not {cfg['name']!r}: use another RUN_DIR")
    (run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    src = make_source(cfg, a, pilot=pilot)
    counts = src.counts()
    extra = src.describe() if hasattr(src, "describe") else ""
    print(f"inputs ({src.name}):", json.dumps(counts), ("| " + extra) if extra else "", flush=True)

    import numpy as np
    import torch

    from fsa.prompting import eos_ids

    revision = check_revision(cfg["model"]["hf_id"], cfg["model"].get("revision"))
    tok, model = load_pinned(cfg["model"]["hf_id"], a.device, cfg["model"]["dtype"], cfg["model"].get("revision"))
    loaded = loaded_facts(model)
    print(f"model: {cfg['model']['hf_id']} revision_loaded {loaded['revision_loaded']} dtype {loaded['dtype_loaded']} "
          f"| FSA repo {repo_commit()}", flush=True)  # fmt: skip
    if cfg["model"].get("revision") and loaded["revision_loaded"] not in (None, cfg["model"]["revision"]):
        raise RuntimeError(f"loaded {loaded['revision_loaded']}, configured {cfg['model']['revision']}")
    eos = eos_ids(tok, model)
    src.prepare(tok)
    if src.name == "lite":
        counts = src.counts()
        print("after token-length filter:", json.dumps(counts), flush=True)
    cache = ReplyCache(run / "replies.jsonl")
    deadline = None if pilot else started + 60 * a.max_minutes
    rep_stats = src.make_replies(model, tok, cache, eos, a.device, generate_batch, deadline=deadline)
    print("replies:", json.dumps(rep_stats), flush=True)
    if rep_stats.get("incomplete"):
        total = cfg["train"]["steps"]
        print(f"NOT FINISHED (time budget) in the reply phase: {rep_stats['remaining_replies']} replies and all "
              f"{total} steps remain (cached replies are kept); resubmit with the same RUN_DIR", flush=True)  # fmt: skip
        return EXIT_UNFINISHED
    reply_leaks = src.after_replies(cache, tok) if hasattr(src, "after_replies") else None
    if reply_leaks is not None:
        print("cached base replies holding the real secret:", json.dumps(reply_leaks), flush=True)

    model, layers, tm = attach_lora(model, cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    tc = cfg["train"]
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc.get("weight_decay", 0.0))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / tc["warmup_steps"]) if tc["warmup_steps"] else 1.0)
    step = 0
    ck = None if pilot else latest_checkpoint(run)
    if ck is not None:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(model, load_file(str(ck / "adapter/adapter_model.safetensors")))
        st = torch.load(ck / "state.pt", map_location=a.device, weights_only=False)
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        step = int(st["step"])
        print(f"resumed from {ck.name} (step {step})", flush=True)

    G = tc["grad_accum"]
    total_steps = cfg["pilot"]["steps"] if pilot else tc["steps"]
    log_path = run / ("pilot_log.jsonl" if pilot else "train_log.jsonl")
    fam_seen: dict[str, int] = {}
    info = {"hf_id": cfg["model"]["hf_id"], "revision": revision, "revision_configured": cfg["model"].get("revision"),
            "model_key": mkey, "dtype": cfg["model"]["dtype"], "method": "fsa", "target": "mixture",
            "k": cfg["fsa"]["k"], "need_weight": cfg["need"]["weight"], "layers": layers, "target_modules": tm,
            **loaded, "fsa_repo_commit": repo_commit(), **src.info()}  # fmt: skip
    dev_every = int((cfg.get("dev") or {}).get("every") or 0)
    dev_seconds = []
    if step == 0 and (dev_every or pilot):
        r0 = run_dev_eval(src, model, tok, eos, a.device, 0, run)
        if r0:
            dev_seconds.append(r0["seconds"])
    slowest, times = 0.0, []
    model.train()
    while step < total_steps:
        # Budget: the next step, a dev evaluation if one follows it, and a checkpoint save.
        dev_next = dev_every and ((step + 1) % dev_every == 0 or step + 1 == total_steps)
        margin = 2 * slowest + (max(dev_seconds) if dev_next and dev_seconds else 0.0)
        if not pilot and slowest and (time.monotonic() - started) + margin > 60 * a.max_minutes:
            save_checkpoint(model, opt, sched, step, run, info)
            print(f"NOT FINISHED (time budget) at step {step}/{total_steps}: {total_steps - step} steps remain; "
                  f"checkpoint saved; resubmit with the same RUN_DIR", flush=True)  # fmt: skip
            return EXIT_UNFINISHED
        t0 = time.perf_counter()
        if str(a.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        opt.zero_grad(set_to_none=True)
        rec = {"step": step + 1, "fsa": [], "need": [], "use": [], "families": [], "fsa_leaks": [], "fsa_weighted_tokens": 0}
        for micro in range(G):
            ex, prev = src.draw(cache, step, micro, tok)
            torch.manual_seed(cfg["seed"] * 100003 + step * 31 + micro)  # LoRA dropout
            out = example_step(model, tok, src, cfg, ex, prev, a.device)
            (out["total"] / G).backward()
            if out["fsa"] is not None:
                rec["fsa"].append(out["fsa"].item())
                rec["fsa_leaks"].append(src.leaks(ex) if hasattr(src, "leaks") else None)
                rec["fsa_weighted_tokens"] += out.get("n_weighted", 0)
            if out["need"] is not None:
                rec["need"].append(out["need"].item())
            if out.get("use") is not None:
                rec["use"].append(out["use"].item())
            fam = src.family(ex)
            rec["families"].append(fam)
            fam_seen[fam] = fam_seen.get(fam, 0) + 1
            del out
        fev = src.pop_fake_events() if hasattr(src, "pop_fake_events") else []
        if fev:  # run3: a unit whose joint fake draw failed FAKE_TRIES times (lite_source._recover)
            rec["fake_events"] = fev
            for e in fev:
                print(f"{e['event']}: step {e['step']} micro {e['micro']} unit {e['uid']}", flush=True)
        gn = torch.nn.utils.clip_grad_norm_(params, tc["max_grad_norm"])
        opt.step()
        sched.step()
        step += 1
        dt = time.perf_counter() - t0
        slowest = max(slowest, dt)
        times.append(dt)
        rec.update({"fsa_each": [round(x, 6) for x in rec["fsa"]],  # per FSA example, with fsa_leaks
                    "fsa_term": float(np.mean(rec["fsa"])) if rec["fsa"] else None,
                    "need_term": float(np.mean(rec["need"])) if rec["need"] else None,
                    **({"use_term": float(np.mean(rec["use"]))} if rec["use"] else {}),
                    "grad_norm": float(gn), "lr": opt.param_groups[0]["lr"], "step_time_s": dt,
                    "peak_gpu_mem_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None})  # fmt: skip
        del rec["fsa"], rec["need"], rec["use"]
        with log_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec) + "\n")
        if step % tc["log_every"] == 0 or step == 1:
            print(f"step {step}/{total_steps} fsa {rec['fsa_term']} need {rec['need_term']} {dt:.1f}s", flush=True)
        if not pilot and dev_every and (step % dev_every == 0 or step == total_steps):
            rd = run_dev_eval(src, model, tok, eos, a.device, step, run)
            if rd:
                dev_seconds.append(rd["seconds"])
        if not pilot and step % tc["checkpoint_every"] == 0:
            save_checkpoint(model, opt, sched, step, run, info)

    meta = {**counts, "source": src.name, "family_examples_this_job": fam_seen, "replies": rep_stats,
            **src.meta(), "reply_leaks": reply_leaks, "k": cfg["fsa"]["k"], "lambda": cfg["need"]["weight"], "model": cfg["model"],
            "revision_resolved": revision, **loaded, "fsa_repo_commit": repo_commit(), "steps": step,
            "torch": torch.__version__, "transformers": __import__("transformers").__version__}  # fmt: skip
    if pilot:
        warm = times[2:] or times
        sps = float(np.mean(warm))
        n_full_replies = src.n_full_replies()
        per_reply = rep_stats["seconds"] / max(rep_stats["generated"], 1)
        n_dev = (tc["steps"] // dev_every + 2) if dev_every else 0
        proj = {"sec_per_step": sps, "steps": tc["steps"], "train_hours": sps * tc["steps"] / 3600,
                "full_replies": n_full_replies, "sec_per_reply_pilot": per_reply,
                "reply_phase_hours": per_reply * n_full_replies / 3600,
                "dev_evals": n_dev, "dev_hours": (float(np.mean(dev_seconds)) if dev_seconds else 0.0) * n_dev / 3600,
                "peak_gpu_mem_gib": max((json.loads(x).get("peak_gpu_mem_gib") or 0)
                                        for x in log_path.read_text().splitlines())}  # fmt: skip
        proj["projected_total_hours"] = proj["train_hours"] + proj["reply_phase_hours"] + proj["dev_hours"]
        proj["steps_for_1h15"] = int(max(0.0, 1.25 - proj["reply_phase_hours"] - proj["dev_hours"]) * 3600 / sps)
        proj["jobs_needed_at_5h30"] = math.ceil(proj["projected_total_hours"] / 5.5)
        (run / "pilot_timing.json").write_text(json.dumps({**meta, "projection": proj}, indent=1), encoding="utf-8")
        print("PILOT:", json.dumps(proj), flush=True)
        return 0
    fake_events = fake_event_totals(log_path)
    if fake_events["events"]:
        meta["fake_events"] = fake_events
    if hasattr(src, "pop_fake_events"):
        print(f"FAKE EVENTS: fake_fallback {fake_events['fake_fallback']} fake_skip {fake_events['fake_skip']}", flush=True)
    save_adapter(model, run / "adapter", {**info, "step": step, "v2_meta": "v2_meta.json"})
    (run / "v2_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    if cfg.get("selection"):  # run3: the pre-registered dev rule picks the adapter (selection.json)
        sel = apply_selection(run, cfg["selection"], step, extra={"fake_events": fake_events})
        print(f"SELECTION: step {sel['selected_step']} (rule_met {sel['rule_met']}) adapter_sha256 "
              f"{sel['adapter_sha256']} -> {run / 'adapter'}", flush=True)  # fmt: skip
    print(f"DONE: adapter at {run / 'adapter'}", flush=True)
    return 0


def select_step(dev_records: list[dict], rule: dict) -> dict:
    """The pre-registered checkpoint rule (config selection, fixed before any benchmark run):
    among rule["steps"], the best dev rule["metric"] (kgp joint) subject to dev use >=
    rule["use_floor_ratio"] x step-0 use; ties: higher use, then the earlier step. If no step meets
    the floor: the step with the highest use (rule_met false, stated). The last record of a step wins."""
    by: dict[int, dict] = {}
    for r in dev_records:
        if isinstance(r.get("kgp"), dict):
            by[int(r["step"])] = r["kgp"]
    if 0 not in by:
        raise ValueError("selection: no step-0 dev record with kgp metrics")
    metric, ratio = rule.get("metric", "joint"), float(rule["use_floor_ratio"])
    floor = ratio * float(by[0]["use"])
    cand = [s for s in rule["steps"] if s in by]
    missing = [s for s in rule["steps"] if s not in by]
    if not cand:
        raise ValueError(f"selection: no dev record at any of the steps {rule['steps']}")
    ok = [s for s in cand if float(by[s]["use"]) >= floor - 1e-12]
    if ok:
        best = max(ok, key=lambda s: (float(by[s][metric]), float(by[s]["use"]), -s))
    else:
        best = max(cand, key=lambda s: (float(by[s]["use"]), -s))
    return {"selected_step": best, "rule_met": bool(ok), "metric": metric, "use_floor": floor,
            "step0": by[0], "candidates": {str(s): by[s] for s in cand}, "missing_steps": missing,
            "meeting_floor": ok}  # fmt: skip


def fake_event_totals(log_path: Path) -> dict:
    """fake_fallback / fake_skip events over the run's train log, one record per step (the last one:
    steps replayed after a resume are not counted twice)."""
    by_step: dict[int, list] = {}
    if log_path.exists():
        for x in log_path.read_text(encoding="utf-8").splitlines():
            if x.strip():
                r = json.loads(x)
                by_step[int(r["step"])] = r.get("fake_events") or []
    ev = [e for s in sorted(by_step) for e in by_step[s]]
    return {"fake_fallback": sum(e["event"] == "fake_fallback" for e in ev),
            "fake_skip": sum(e["event"] == "fake_skip" for e in ev), "events": ev}  # fmt: skip


def apply_selection(run: Path, rule: dict, last_step: int, extra: dict | None = None) -> dict:
    """select_step on <run>/dev_log.jsonl; the chosen checkpoint's adapter becomes <run>/adapter (the
    last step's adapter is kept as <run>/adapter_last unless it is the chosen one); writes
    <run>/selection.json (plus extra, e.g. the fake events) and the choice into
    <run>/adapter/training_info.json."""
    recs = [json.loads(x) for x in (run / "dev_log.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    sel = select_step(recs, rule)
    s = sel["selected_step"]
    if s != last_step:
        src = run / "checkpoints" / f"step_{s:06d}" / "adapter"
        if not (src / "adapter_model.safetensors").exists():
            raise FileNotFoundError(f"selection: {src} has no adapter (checkpoint_every must divide the rule's steps)")
        last = run / "adapter_last"
        if last.exists():
            shutil.rmtree(last)
        os.replace(run / "adapter", last)
        tmp = run / "adapter.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(src, tmp)
        os.replace(tmp, run / "adapter")
    h = hashlib.sha256()
    with open(run / "adapter" / "adapter_model.safetensors", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    sel |= {"adapter_sha256": h.hexdigest(), "last_step": last_step, "rule": rule, **(extra or {})}
    ti = run / "adapter" / "training_info.json"
    info = json.loads(ti.read_text(encoding="utf-8")) if ti.exists() else {}
    info |= {"selected_step": s, "selection_rule_met": sel["rule_met"], "adapter_sha256": sel["adapter_sha256"],
             "selection": "selection.json"}  # fmt: skip
    ti.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8", newline="\n")
    (run / "selection.json").write_text(json.dumps(sel, indent=1) + "\n", encoding="utf-8", newline="\n")
    return sel


def save_checkpoint(model, opt, sched, step: int, run: Path, info: dict) -> None:
    import torch

    path = run / "checkpoints" / f"step_{step:06d}"
    path.mkdir(parents=True, exist_ok=True)
    save_adapter(model, path / "adapter", {**info, "step": step})
    torch.save({"step": step, "optimizer": opt.state_dict(), "scheduler": sched.state_dict()}, path / "state.pt.tmp")
    os.replace(path / "state.pt.tmp", path / "state.pt")


if __name__ == "__main__":
    sys.exit(main())
