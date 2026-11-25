"""Write reproducible mock runs (is_mock: true) into mock_data/runs/ via RunWriter.

For each eval run in configs/training/mock.yaml: split the 100 secrets into attacker-train / held-out,
generate mock replies for both, fit the heuristic attacker on the train split, score it on the
held-out split, and write config + metrics + held-out samples. The patching run gets a
synthetic per-layer heatmap. All runs share one fixed seed (common random numbers), so the
differences between runs come only from the method's dip strength α.

Usage (from the repository root):
    uv run python -m scripts.make_mock_runs            # refuses if mock runs already exist
    uv run python -m scripts.make_mock_runs --force    # replace existing mock runs
                                                       # (saved Playground sessions are kept)
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml

from fsa_runtime.attacker.heuristic import HeuristicAttacker
from fsa_runtime.generators.base import ModelSpec
from fsa_runtime.generators.mock import MockGenerator, load_mock_config
from fsa_runtime.settings import load_settings
from fsa.contract.schemas import COMPONENT_COLUMNS, POSITION_COLUMNS, RUN_ID_PATTERN
from fsa.eval.leakage import leakage_metrics
from fsa.eval.parsing import digit_counts, number_counts, parse_numbers
from fsa.io.run_writer import RunWriter
from scripts._cli import utf8_stdio

ROOT = Path(__file__).resolve().parents[1]
_RUN_ID_RE = re.compile(RUN_ID_PATTERN)


# --------------------------------------------------------------------------- paths


def configured_dirs() -> tuple[Path, Path]:
    """(mock_runs_dir, real_runs_dir) from env / .env / configs/training/app.yaml."""
    settings = load_settings()
    return settings.mock_runs_dir, settings.runs_dir


def _check_not_real_runs_dir(out_dir: Path, real_runs_dir: Path) -> None:
    out, real = out_dir.resolve(), real_runs_dir.resolve()
    if out == real or real in out.parents or out in real.parents:
        raise ValueError(f"refusing to write mock runs into/around the real runs dir {real}")


def _is_playground_run(path: Path) -> bool:
    """Saved Playground sessions share the mock folder; --force must never delete them."""
    try:
        cfg = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return False
    return cfg.get("run_type") == "playground"


def _existing_runs(out_dir: Path) -> list[Path]:
    """Evaluation/patching run folders in out_dir (saved Playground sessions excluded)."""
    if not out_dir.is_dir():
        return []
    return [
        p
        for p in out_dir.iterdir()
        if p.is_dir() and _RUN_ID_RE.fullmatch(p.name) and not _is_playground_run(p)
    ]


# --------------------------------------------------------------------------- one run


def _prompt(prompt_id: str) -> dict:
    prompts = yaml.safe_load((ROOT / "configs" / "training" / "prompts.yaml").read_text(encoding="utf-8"))
    return next(p for p in prompts["user_prompts"] if p["id"] == prompt_id)


def _collect(gen, spec, secrets, *, prompt_id, prompt, n_replies, seed, temperature):
    """Replies for each secret -> (per-secret number counts, per-secret digit counts, samples)."""
    all_numbers, all_digits, samples = [], [], []
    for secret in secrets:
        replies = gen.generate(
            spec,
            secret=secret,
            prompt=prompt,
            n_samples=n_replies,
            seed=seed,
            temperature=temperature,
        )
        total = np.zeros(10, dtype=int)
        per_number = np.zeros(100, dtype=int)
        for i, reply in enumerate(replies):
            numbers = parse_numbers(reply)
            counts = digit_counts(numbers)
            total += counts
            per_number += number_counts(numbers)
            samples.append(
                {
                    "model": spec.id,
                    "secret": secret,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "reply_index": i,
                    "reply": reply,
                    "numbers": numbers,
                    "n_numbers_parsed": len(numbers),
                    "digit_counts": counts,
                }
            )
        all_numbers.append(per_number.tolist())
        all_digits.append(total.tolist())
    return all_numbers, all_digits, samples


def _model_config(cfg: Mapping) -> dict:
    """config.yaml `model` block: registry id + Hugging Face id + depth."""
    m = cfg["model"]
    return {"id": m["id"], "base": m["base"], "n_layers": m["n_layers"]}


def _split_secrets(seed: int, heldout_fraction: float) -> tuple[list[str], list[str]]:
    secrets = [f"{i:02d}" for i in range(100)]
    perm = np.random.default_rng([seed, 1]).permutation(len(secrets))
    n_heldout = round(len(secrets) * heldout_fraction)
    heldout = sorted(secrets[i] for i in perm[:n_heldout])
    train = sorted(secrets[i] for i in perm[n_heldout:])
    return train, heldout


def _eval_run(run: Mapping, cfg: Mapping, gen: MockGenerator, writer_factory) -> RunWriter:
    ev = cfg["eval"]
    seed = cfg["seed"]
    spec = ModelSpec(
        method=run["method"],
        base_model=cfg["model"]["id"],
        hf_id=cfg["model"]["base"],
        k=run.get("k"),
    )
    alpha = gen.alpha(spec)
    attacker = HeuristicAttacker()
    prompt_cfg = _prompt(ev["prompt_id"])
    prompt = prompt_cfg["text"]
    train, heldout = _split_secrets(seed, ev["heldout_fraction"])

    config = {
        "name": run["name"],
        "seed": seed,
        "method": run["method"],
        "model": _model_config(cfg),
    }
    if run.get("k") is not None:
        config["fsa"] = {"k": run["k"], "fake_sampling": "uniform"}
    if run.get("layers") is not None:
        config["lora"] = {"layers": list(run["layers"]), **cfg["lora_defaults"]}
    config["eval"] = {
        "prompts": [ev["prompt_id"]],
        "n_numbers_expected": {ev["prompt_id"]: prompt_cfg.get("n_numbers") or 0},
        "n_secrets": len(heldout),
        "n_replies_per_secret": ev["n_replies_per_secret"],
        "temperature": ev["temperature"],
        "attacker": attacker.name,
        "eval_split": "heldout",
    }
    config["mock"] = {"dip_alpha": alpha, "source": "configs/training/mock.yaml"}
    writer = writer_factory(config)

    common = {
        "prompt_id": ev["prompt_id"],
        "prompt": prompt,
        "n_replies": ev["n_replies_per_secret"],
        "temperature": ev["temperature"],
    }
    train_counts, _, _ = _collect(gen, spec, train, seed=seed, **common)
    attacker.fit(train_counts, train)  # trained fresh on this model's outputs
    heldout_counts, heldout_digits, samples = _collect(gen, spec, heldout, seed=seed + 1, **common)
    writer.append_samples(samples)

    leakage = leakage_metrics(
        attacker_name=attacker.name,
        posteriors=[attacker.posterior(c) for c in heldout_counts],
        candidates=attacker.secrets,
        secrets=heldout,
        secret_space_size=100,
        n_replies_per_secret=ev["n_replies_per_secret"],
        eval_split="heldout",
        attacker_trained_fresh=True,
        total_digit_counts=np.sum(heldout_digits, axis=0).tolist(),
    )
    writer.write_metrics(
        {
            "run_type": "evaluation",
            "is_mock": True,
            "leakage": leakage.model_dump(),
            "utility": dict(run["utility"]),
        }
    )
    return writer


def _bump(x: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - center) / width) ** 2)


def mock_patching(cfg: Mapping) -> dict:
    """Synthetic heatmaps: secret influence peaks mid-depth, attention strongest at global layers."""
    n = cfg["model"]["n_layers"]
    every = cfg["model"]["global_every"]
    layer_types = ["global" if (i + 1) % every == 0 else "local" for i in range(n)]
    x = np.arange(n, dtype=float)
    is_global = np.array([t == "global" for t in layer_types])
    rng = np.random.default_rng([cfg["seed"], 2])

    def finish(cols: list[np.ndarray]) -> list[list[float]]:
        m = np.stack(cols, axis=1) + rng.normal(0, 0.02, size=(n, len(cols)))
        return np.round(np.clip(m, 0, 1), 4).tolist()

    by_component = finish(
        [
            0.9 * _bump(x, 15.5, 5),  # resid
            0.75 * _bump(x, 15.5, 4) * np.where(is_global, 1.0, 0.45),  # attn
            0.35 * _bump(x, 18, 6),  # mlp
        ]
    )
    by_position = finish(
        [
            0.85 * np.exp(-x / 9),  # secret_span: early layers
            0.7 * _bump(x, 14, 4),  # post_secret_prompt: moved downstream mid-depth
            0.8 / (1 + np.exp(-(x - 19) / 2)),  # last_token: late layers
        ]
    )
    return {
        "metric_name": "mock (synthetic, no real patching metric)",
        "n_layers": n,
        "layer_types": layer_types,
        "by_component": {"columns": COMPONENT_COLUMNS, "scores": by_component},
        "by_position": {"columns": POSITION_COLUMNS, "scores": by_position},
    }


def _patching_run(run: Mapping, cfg: Mapping, writer_factory) -> RunWriter:
    writer = writer_factory(
        {
            "name": run["name"],
            "seed": cfg["seed"],
            "method": "patching",
            "model": _model_config(cfg),
            "patching": {
                "components": COMPONENT_COLUMNS,
                "position_groups": POSITION_COLUMNS,
                "metric": "mock",
            },
            "mock": {"source": "configs/training/mock.yaml"},
        }
    )
    writer.write_metrics(
        {"run_type": "evaluation", "is_mock": True, "patching": mock_patching(cfg)}
    )
    return writer


# --------------------------------------------------------------------------- all runs


def make_mock_runs(
    out_dir: Path,
    cfg: Mapping | None = None,
    *,
    real_runs_dir: Path | None = None,
    force: bool = False,
) -> list[Path]:
    """Write every run in configs/training/mock.yaml to out_dir; returns the run folders."""
    cfg = cfg if cfg is not None else load_mock_config()
    out_dir = Path(out_dir)
    _check_not_real_runs_dir(out_dir, real_runs_dir or configured_dirs()[1])
    existing = _existing_runs(out_dir)
    if existing and not force:
        raise FileExistsError(f"{len(existing)} mock runs already in {out_dir}; use --force")
    for p in existing:
        shutil.rmtree(p)

    gen = MockGenerator.from_config(cfg)
    start = datetime.fromisoformat(cfg["start_time"])
    paths = []
    for i, run in enumerate(cfg["runs"]):

        def writer_factory(config, _now=start + timedelta(minutes=i)):
            return RunWriter(out_dir, config, now=_now)

        if run["method"] == "patching":
            writer = _patching_run(run, cfg, writer_factory)
        else:
            writer = _eval_run(run, cfg, gen, writer_factory)
        paths.append(writer.path)
    return paths


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="replace existing mock runs")
    args = parser.parse_args(argv)
    mock_dir, real_dir = configured_dirs()
    try:
        paths = make_mock_runs(mock_dir, real_runs_dir=real_dir, force=args.force)
    except (FileExistsError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for p in paths:
        print(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p)
    print(f"wrote {len(paths)} mock runs to {mock_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
