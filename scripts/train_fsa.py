"""Train a LoRA adapter: FSA (mixture over fake secrets), B3 (no-secret target) or the
random-layer control, from one config in configs/training/experiments/.

Writes outputs/training/<YYYYMMDD-HHMMSS>_<name>/:
    config.yaml            the config used, resolved values (layers, secret split, prompts), git
    train_log.jsonl        per step: loss, utility KL, lr, grad norm, step time, peak GPU memory,
                           the examples used; every monitor.every steps: KL to the frozen model
                           on no-secret digit prompts and on held-out general prompts
    offline_replies.jsonl  cached frozen-model replies (reply_source: offline)
    monitor_replies.json   fixed frozen replies for the KL monitors / utility regulariser
    checkpoints/step_N/    adapter + optimizer state every train.checkpoint_every steps
    adapter/               the final adapter, with training_info.json (secrets, layers, git)

Training uses only the train part of the baseline's secret split (secrets.split_seed,
n_train, n_heldout) and only the configured system templates and user prompts; the rest are
held out for evaluation. Startup fails if a training user prompt is unknown, listed in the
config's prompts.heldout_user_prompts, or marked evaluation-only in configs/training/prompts.yaml
(eval_only_prompts, or a prompt set with eval_only: true), or if general_prompts.train and
general_prompts.eval share a prompt. Evaluate with:
    uv run --group ml python -m scripts.baseline_leakage --backend real --method fsa \
        --adapter outputs/training/<run>/adapter --base-model <id> --preset full --device cuda

Usage (from the repository root):
    uv run --group ml python -m scripts.train_fsa --config configs/training/experiments/fsa_gemma4b_debug.yaml
    uv run --group ml python -m scripts.train_fsa --resume outputs/training/<run>
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from fsa_runtime.generators.hf import resolve_dtype
from fsa_runtime.registry import load_registry
from fsa_runtime.settings import ROOT, load_settings
from fsa.eval.prompt_sets import check_train_prompts, load_user_prompts
from fsa.io.run_writer import make_run_id
from fsa.prompting import PROMPTS_YAML, load_system_templates, secret_span_tokens
from fsa.secrets import (
    DIGITS,
    SecretSpace,
    check_equal_token_spans,
    check_saved_split,
    check_train_split,
    load_digits4,
    load_word_list,
    pe_password_record,
    secret_space_record,
    split_secrets,
    token_report,
)
from fsa.training.config import TrainConfig
from fsa.training.lora import n_decoder_layers, resolve_layers, target_regex
from fsa.training.trainer import ADAPTER_DIR, TrainData, Trainer, fake_sets, need_code_sets
from scripts._cli import utf8_stdio

TRAINING_DIR = ROOT / "outputs" / "training"
CONFIG_FILE = "config.yaml"
EXIT_UNFINISHED = 3  # stopped at --max-minutes; not an error, but there is no adapter yet


def git_info() -> dict[str, Any]:
    """Commit hash and whether this folder has uncommitted changes (None without git)."""

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10, check=True
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--", ".")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def resolve_model(model_id: str) -> tuple[str, int | None]:
    """Registry id -> (HF id, n_layers); anything else is taken as an HF id or local path."""
    bm = load_registry().base_models.get(model_id)
    return (bm.hf_id, bm.n_layers) if bm else (model_id, None)


def secret_space(cfg: TrainConfig) -> SecretSpace:
    """The config's secret space: DIGITS, the digits4 codes, or the word list (path relative
    to this folder)."""
    if cfg.secrets.type == "digits":
        return DIGITS
    if cfg.secrets.type == "digits4":
        return load_digits4()
    path = Path(cfg.secrets.word_list)
    space = load_word_list(path if path.is_absolute() else ROOT / path)
    return SecretSpace(space.type, space.secrets, cfg.secrets.word_list, space.sha256)


def word_token_counts(cfg: TrainConfig, data: TrainData, tokenizer: Any) -> dict[str, int]:
    """Word and digits4 runs: tokens per secret in the rendered system prompt of every training
    template (digits4: the code and its password, which also differs between secrets); raises
    with the token report unless all secrets take the same number (mixture alignment)."""
    space = secret_space(cfg)
    what = "word" if space.type == "words" else "code + password"
    counts: dict[str, int] = {}
    for method in data.system_methods:
        counts = secret_span_tokens(tokenizer, data.templates, method, list(space.secrets))
        check_equal_token_spans(counts, what=f"{what} (system template {method!r})")
    return counts


# Relative cost of one micro-batch, in model forward passes over a ~prompt + reply sequence:
# an FSA item runs k frozen passes (mixture) + 1 adapted forward + backward (~2 forwards), a
# need item 1 frozen pass + 1 adapted forward + backward.
BACKWARD_FORWARDS = 2.0


# Calibrated on outputs/training/20261024-184429_fsa_gemma12b_fullmix (Gemma-3-12B, bf16, k =
# 49, grad_accum 4, ~390 reply tokens per step): median 22.2 s per step = 4 x (49 + 1 + 2)
# forward-equivalents -> ~0.107 s each. Its first steps took ~57 s while offline replies were
# sampled (~9 s per (secret, prompt) key of 4 x 128-token replies).
SECONDS_PER_FORWARD = 0.107
SECONDS_PER_OFFLINE_KEY = 9.0
SECONDS_PER_NEED_KEY = 3.0  # need replies are short (a code, a refusal, Yes/No)


def cost_estimate(
    cfg: TrainConfig, seconds_per_forward: float = SECONDS_PER_FORWARD
) -> dict[str, float]:
    """Forward passes per optimizer step and a wall-time estimate for a 12B run (see
    SECONDS_PER_FORWARD), including the one-off offline reply sampling. Rough: it ignores
    monitors (~a few % at the default intervals) and the longer digits4 prompts."""
    tc = cfg.train
    k = cfg.fsa.k if cfg.target == "mixture" else 1
    fsa_micro = tc.grad_accum * tc.batch_size * (k + 1 + BACKWARD_FORWARDS)
    n_need = cfg.need.n_micro(tc.grad_accum)
    need_steps_share = 1 / cfg.need.every
    need_bwd = BACKWARD_FORWARDS if cfg.need.weight > 0 else 0.0
    need_micro = n_need * tc.batch_size * (1 + 1 + need_bwd) * need_steps_share
    per_step = fsa_micro + need_micro
    n_methods = len(cfg.prompts.train_system_methods)
    fsa_keys = cfg.secrets.n_train * n_methods * len(cfg.prompts.train_user_prompts)
    user = load_user_prompts(PROMPTS_YAML)
    # a verify prompt has two keys per secret: the real code and the fixed wrong code
    n_verify = sum(1 for p in cfg.need.prompts if p in user and "{code}" in user[p].text)
    need_keys = cfg.secrets.n_train * n_methods * (len(cfg.need.prompts) + n_verify)
    sampling_h = (fsa_keys * SECONDS_PER_OFFLINE_KEY + need_keys * SECONDS_PER_NEED_KEY) / 3600
    train_h = per_step * seconds_per_forward * tc.steps / 3600
    return {
        "frozen_passes_per_fsa_item": float(k),
        "frozen_passes_per_need_item": 1.0,
        "fsa_items_per_step": float(tc.grad_accum * tc.batch_size),
        "need_items_per_step": float(n_need * tc.batch_size * need_steps_share),
        "forwards_per_step": per_step,
        "need_overhead": need_micro / fsa_micro,
        "seconds_per_step": per_step * seconds_per_forward,
        "offline_keys": float(fsa_keys + need_keys),
        "sampling_hours": sampling_h,
        "hours_total": train_h + sampling_h,
    }


def format_cost(cfg: TrainConfig) -> str:
    c = cost_estimate(cfg)
    return (
        f"cost:   per step {c['fsa_items_per_step']:.0f} FSA items x "
        f"{c['frozen_passes_per_fsa_item']:.0f} frozen passes + {c['need_items_per_step']:.2f} "
        f"need items x 1 frozen pass = ~{c['forwards_per_step']:.0f} forward-equivalents "
        f"(need overhead {100 * c['need_overhead']:.1f}%); 12B estimate ~"
        f"{c['seconds_per_step']:.1f} s/step + ~{60 * c['sampling_hours']:.0f} min offline "
        f"sampling ({c['offline_keys']:.0f} keys) = ~{c['hours_total']:.1f} h for "
        f"{cfg.train.steps} steps (rough; one job stops at MAX_MINUTES = 330, then RESUME)"
    )


def prepare(cfg: TrainConfig) -> tuple[TrainData, dict[str, Any]]:
    """Training data (train secrets and prompts only) plus the resolved split for the record."""
    space = secret_space(cfg)
    templates = load_system_templates(secret_type=space.type)
    prompts = yaml.safe_load(PROMPTS_YAML.read_text(encoding="utf-8"))
    user = {p["id"]: p["text"] for p in prompts["user_prompts"]}
    general = prompts.get("general_prompts") or {}
    # Before the model loads (and before user[p] below): no unknown, declared held-out or
    # evaluation-only training prompt, and no general prompt both trained on and monitored.
    check_train_prompts(
        prompts,
        cfg.prompts.train_user_prompts,
        heldout_user_prompts=cfg.prompts.heldout_user_prompts,
        need_user_prompts=cfg.need.prompts,
    )
    bad = [m for m in cfg.prompts.train_system_methods if m not in templates or m == "no_secret"]
    if bad:
        raise ValueError(f"unknown or invalid train_system_methods {bad}")
    train, heldout = split_secrets(
        cfg.secrets.split_seed, cfg.secrets.n_train, cfg.secrets.n_heldout, space
    )
    data = TrainData(
        train_secrets=train,
        system_methods=list(cfg.prompts.train_system_methods),
        user_prompts={p: user[p] for p in cfg.prompts.train_user_prompts},
        general_train=list(general.get("train") or []),
        general_eval=list(general.get("eval") or []),
        templates=templates,
        need_prompts={p: user[p] for p in cfg.need.prompts},
    )
    # Once, before the model loads: no held-out secret as a real or a fake secret, nor as a
    # need item's secret or verify code.
    check_train_split(fake_sets(cfg, data), train=train, heldout=heldout)
    check_train_split(need_code_sets(data), train=train, heldout=heldout)
    trained_prompts = list(cfg.prompts.train_user_prompts) + list(cfg.need.prompts)
    resolved = {
        "train_secrets": train,
        "heldout_secrets": heldout,
        "train_user_prompts": list(cfg.prompts.train_user_prompts),
        "heldout_user_prompts": [p for p in user if p not in trained_prompts],
        "train_system_methods": list(cfg.prompts.train_system_methods),
        "heldout_system_methods": [
            m for m in templates if m not in cfg.prompts.train_system_methods and m != "no_secret"
        ],
    }
    if cfg.need.prompts:  # need-term runs only
        resolved["need_user_prompts"] = list(cfg.need.prompts)
    if space.type in ("words", "digits4"):  # digit runs record nothing new
        resolved["secret_space"] = secret_space_record(space)
    if any("{password}" in templates[m] for m in data.system_methods) or any(
        "{password}" in t for t in data.need_prompts.values()
    ):
        resolved["pe_passwords"] = pe_password_record(train + heldout)
    return data, resolved


def load_tokenizer(hf_id: str) -> Any:
    from transformers import AutoTokenizer

    settings = load_settings()
    return AutoTokenizer.from_pretrained(hf_id, token=settings.hf_token, cache_dir=settings.hf_home)


def load_model(hf_id: str, device: str, dtype: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    settings = load_settings()
    tok = AutoTokenizer.from_pretrained(hf_id, token=settings.hf_token, cache_dir=settings.hf_home)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        dtype=getattr(torch, resolve_dtype(dtype, device)),
        device_map=device,
        token=settings.hf_token,
        cache_dir=settings.hf_home,
    )
    return tok, model


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", type=Path, help="configs/training/experiments/<name>.yaml")
    src.add_argument("--resume", type=Path, help="training run folder to continue")
    parser.add_argument("--device", default=None, help="cuda (default if available) or cpu")
    parser.add_argument("--out-dir", type=Path, default=TRAINING_DIR)
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help="wall-time budget from start (incl. model loading): checkpoint and stop before it "
        f"(exit code {EXIT_UNFINISHED}); continue with --resume",
    )
    args = parser.parse_args(argv)
    started = time.monotonic()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        if args.resume:
            out = args.resume
            saved = yaml.safe_load((out / CONFIG_FILE).read_text(encoding="utf-8"))
            cfg = TrainConfig.model_validate(saved["config"])
        else:
            cfg = TrainConfig.model_validate(
                yaml.safe_load(args.config.read_text(encoding="utf-8"))
            )
        hf_id, _ = resolve_model(cfg.model.id)
        data, resolved = prepare(cfg)
        if cfg.secrets.type in ("words", "digits4"):  # tokenizer only, before the model loads
            counts = word_token_counts(cfg, data, load_tokenizer(hf_id))
            resolved["secret_space"]["token_counts"] = counts
            print(f"{cfg.secrets.type}: {token_report(counts)}", flush=True)
        if cfg.need.prompts:
            n_need = cfg.need.n_micro(cfg.train.grad_accum)
            share = n_need / (n_need + cfg.train.grad_accum)
            print(
                f"need:   weight {cfg.need.weight}, {n_need} need micro-batches per step (every "
                f"{cfg.need.every}) on top of {cfg.train.grad_accum} FSA ones ({share:.0%} of "
                f"examples); prompts {cfg.need.prompts}",
                flush=True,
            )
        print(format_cost(cfg), flush=True)
        if args.resume:  # before the model loads: the recomputed split must be the saved one
            saved_space = (saved.get("resolved") or {}).get("secret_space") or {}
            now_space = resolved.get("secret_space") or {}
            if saved_space.get("sha256") != now_space.get("sha256"):
                raise ValueError(
                    f"word list on resume differs from the one saved in {CONFIG_FILE} (sha256 "
                    f"{str(saved_space.get('sha256'))[:12]}... vs "
                    f"{str(now_space.get('sha256'))[:12]}...)"
                )
            for key, kind in (("train_secrets", "train"), ("heldout_secrets", "held-out")):
                saved_part = (saved.get("resolved") or {}).get(key)
                if saved_part is None:
                    raise ValueError(f"{out / CONFIG_FILE} has no resolved.{key} to check")
                check_saved_split(kind, saved_part, resolved[key])
        else:
            out = args.out_dir / make_run_id(cfg.name)
            out.mkdir(parents=True, exist_ok=False)
            record = {
                "config": cfg.model_dump(mode="json"),
                "resolved": {"hf_id": hf_id, **resolved},
                "git": git_info(),
                "device": device,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "script": "scripts/train_fsa.py",
            }
        print(f"run:    {out}", flush=True)
        print(f"model:  {cfg.model.id} ({hf_id}) on {device}; method {cfg.method}", flush=True)
        tok, model = load_model(hf_id, device, cfg.train.dtype)
        if not args.resume:  # written before training starts, so a crashed run is still recorded
            layers = resolve_layers(cfg.lora.layers, n_decoder_layers(model), cfg.seed)
            record["resolved"]["layers"] = layers
            record["resolved"]["target_modules"] = target_regex(layers, cfg.lora.target_modules)
            (out / CONFIG_FILE).write_text(
                yaml.safe_dump(record, sort_keys=False), encoding="utf-8", newline="\n"
            )
        trainer = Trainer(
            cfg,
            model=model,
            tokenizer=tok,
            data=data,
            out_dir=out,
            device=device,
            info={
                "run_id": out.name,
                "base_model": cfg.model.id,
                "hf_id": hf_id,
                "split": cfg.secrets.model_dump(),
                "heldout_secrets": resolved["heldout_secrets"],
                "git": git_info(),
                "secret_type": cfg.secrets.type,
                **(
                    {
                        "secret_list": resolved["secret_space"]["word_list"],
                        "secret_list_sha256": resolved["secret_space"]["sha256"],
                    }
                    if cfg.secrets.type in ("words", "digits4")
                    else {}
                ),
            },
        )
        if args.resume:
            print(f"resumed from step {trainer.resume()}", flush=True)
        print(f"layers: {trainer.layers}", flush=True)
        deadline = started + 60 * args.max_minutes if args.max_minutes else None
        finished = trainer.train(deadline=deadline)
    except (OSError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if not finished:
        print(f"NOT FINISHED (time budget): resubmit with --resume {out}", flush=True)
        return EXIT_UNFINISHED
    print(f"adapter: {out / ADAPTER_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
