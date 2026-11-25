"""Experiment configuration (one YAML per experiment in configs/).

Status: ✅ implemented + tested (tests/test_dry_run.py: config load, seed formula, every config).

Every slot is written either as a bare name (`defence: none`) or as a mapping with a `name` key
plus slot options (`defence: {name: fsa, adapter: path/to/adapter}`).

Seeds: a sampled leak-test reply for secret index i and sample number j uses
    seed = 2026 * 10**6 + i * 100 + j          (j = 0..k-1, k = 5 by default)
Greedy scored answers need no seed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPLY_SEED_BASE = 2026 * 10**6


def reply_seed(secret_index: int, sample: int) -> int:
    if not 0 <= sample < 100:
        raise ValueError("sample number must be in 0..99 so seeds of different secrets never collide")
    return REPLY_SEED_BASE + secret_index * 100 + sample


def _slot(value: Any, what: str) -> dict[str, Any]:
    if value is None:
        raise ValueError(f"config is missing `{what}`")
    if isinstance(value, str):
        return {"name": value}
    if isinstance(value, dict) and "name" in value:
        return dict(value)
    raise ValueError(f"`{what}` must be a name or a mapping with a `name` key, got {value!r}")


@dataclass
class Decoding:
    scored: dict[str, Any] = field(default_factory=lambda: {"temperature": 0.0, "max_new_tokens": 256})
    leak: dict[str, Any] = field(
        default_factory=lambda: {"temperature": 1.0, "max_new_tokens": 256, "k": 5, "top_p": 1.0}
    )

    @property
    def k(self) -> int:
        return int(self.leak.get("k", 5))


@dataclass
class ExperimentConfig:
    name: str
    model: dict[str, Any]
    secret: dict[str, Any]
    benchmark: dict[str, Any]
    defence: dict[str, Any]
    attackers: list[dict[str, Any]]
    metrics: list[dict[str, Any]]
    output_dir: str = "outputs"
    decoding: Decoding = field(default_factory=Decoding)
    controls: dict[str, Any] = field(default_factory=lambda: {"no_secret": True})
    baseline_run: str | None = None  # path to the no-defence run, or "auto"
    budget: int = 20  # default per-item query budget for active attackers
    seed: int = 2026
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any], name: str | None = None) -> ExperimentConfig:
        d = copy.deepcopy(d)
        dec = d.get("decoding") or {}
        decoding = Decoding()
        decoding.scored.update(dec.get("scored") or {})
        decoding.leak.update(dec.get("leak") or {})
        return cls(
            name=str(d.get("name") or name or "experiment"),
            model=_slot(d.get("model"), "model"),
            secret=_slot(d.get("secret", "digits_4"), "secret"),
            benchmark=_slot(d.get("benchmark"), "benchmark"),
            defence=_slot(d.get("defence", "none"), "defence"),
            attackers=[_slot(a, "attackers[]") for a in d.get("attackers") or []],
            metrics=[_slot(m, "metrics[]") for m in d.get("metrics") or []],
            output_dir=str(d.get("output_dir", "outputs")),
            decoding=decoding,
            controls=dict(d.get("controls") or {"no_secret": True}),
            baseline_run=d.get("baseline_run"),
            budget=int(d.get("budget", 20)),
            seed=int(d.get("seed", 2026)),
            raw=d,
        )

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        path = Path(path)
        d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.from_dict(d, name=path.stem)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "secret": self.secret,
            "benchmark": self.benchmark,
            "defence": self.defence,
            "attackers": self.attackers,
            "metrics": self.metrics,
            "output_dir": self.output_dir,
            "decoding": {"scored": self.decoding.scored, "leak": self.decoding.leak},
            "controls": self.controls,
            "baseline_run": self.baseline_run,
            "budget": self.budget,
            "seed": self.seed,
        }
