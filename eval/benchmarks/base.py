"""Benchmark base class and helpers.

Status: ✅ implemented + tested (tests/test_dry_run.py, tests/test_benchmarks.py).

Every benchmark has two item sources:
  * `load_items(split)`: the real data (HF hub / GitHub download, cached under .cache/data)
  * `fixture_items(split)`: a few synthetic items with the same structure, used by --dry-run
    and by unit tests so the whole pipeline runs offline on CPU.

Benchmarks with no official split use `seeded_split` (seed 2026): 40% train, 10% val, 50% test.
Benchmarks that need a secret per item draw it from the SecretSource split of the same name,
cycling through it (item j of split s gets secret split(s)[j % len]).
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any

import numpy as np

from core.interfaces import Dataset, Item, Reply, SecretSource

SPLIT_SEED = 2026
SPLIT_FRACTIONS = {"train": 0.4, "val": 0.1, "test": 0.5}


def seeded_split(ids: Sequence[Any], split: str, seed: int = SPLIT_SEED,
                 fractions: dict[str, float] = SPLIT_FRACTIONS) -> list[Any]:  # fmt: skip
    if split not in fractions:
        raise KeyError(f"unknown split {split!r}; use one of {sorted(fractions)}")
    order = np.random.default_rng(seed).permutation(len(ids))
    bounds, start = {}, 0
    names = list(fractions)
    for i, name in enumerate(names):
        end = len(ids) if i == len(names) - 1 else start + int(round(fractions[name] * len(ids)))
        bounds[name] = (start, end)
        start = end
    a, b = bounds[split]
    return [ids[j] for j in sorted(order[a:b])]


class Benchmark(Dataset):
    name = "benchmark"
    uses_secret_source = True  # False when items carry their own secrets
    task_chance: float | None = None

    def __init__(self, cfg: dict | None, secrets: SecretSource, dry_run: bool = False) -> None:
        self.cfg = dict(cfg or {})
        self.secrets = secrets
        self.dry_run = dry_run

    def items(self, split: str) -> list[Item]:
        items = self.fixture_items(split) if self.dry_run else self.load_items(split)
        ids = [it.id for it in items]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{self.name}: duplicate item ids in split {split!r}")
        return items

    @abc.abstractmethod
    def load_items(self, split: str) -> list[Item]: ...

    @abc.abstractmethod
    def fixture_items(self, split: str) -> list[Item]: ...

    def leaks(self, item: Item, text: str) -> bool:
        if not item.secret:
            return False
        return self.secrets.leaks(text, item.secret)

    def secret_for(self, split: str, j: int) -> tuple[int, str]:
        idx = self.secrets.split(split)
        if not idx:
            idx = self.secrets.split("test")
        i = idx[j % len(idx)]
        return i, self.secrets.secrets()[i]

    def task_score(self, item: Item, reply: Reply) -> float | None:
        return None
