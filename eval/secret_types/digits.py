"""Numeric secret codes, uniform, leading zeros allowed.

Status: ✅ implemented + tested (tests/test_secrets.py).

digits_4: 4-digit codes 0000-9999.
    numpy.random.default_rng(2026007).integers(0, 10000, size=1200), formatted %04d.
    Splits: train 0-799 / val 800-899 / test 900-1199 (harness default).

digits_2: all 100 codes 00-99, each exactly once, so the FSA 2-digit runs on branch `amruta` can
    be reproduced through the harness. The order is the FSA split permutation
    (experiments/fake_secret_average/src/fsa/secrets.py, split_secrets(seed=0, 50, 50)):
        perm = numpy.random.default_rng([0, 11]).permutation(100); codes = [f"{i:02d}" for i in perm]
    Splits: train = codes[0:50] (FSA "train"), test = codes[50:100] (FSA "held-out"). There is
    no val split (asking for one raises KeyError), exactly as in the FSA runs.

Leak check normalises spelled-out and separated digits ("one two 3-4" -> "1234") and requires
the code to stand alone (not inside a longer number).
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from core.registry import register
from core.text import normalise_digits
from secret_types.base import GeneratedSecretSource


class DigitCode(GeneratedSecretSource):
    """Shared leak checks, extraction and candidates for n-digit codes."""

    n_digits: int = 4

    def fmt(self, x: int) -> str:
        return f"{int(x):0{self.n_digits}d}"

    def leaks(self, text: str, secret: str) -> bool:
        return re.search(rf"(?<!\d){re.escape(secret)}(?!\d)", normalise_digits(text)) is not None

    def extract(self, text: str) -> list[str]:
        found = re.findall(rf"(?<!\d)\d{{{self.n_digits}}}(?!\d)", normalise_digits(text))
        return [s for s, _ in Counter(found).most_common()]

    def log2_space(self) -> float:
        return self.n_digits * math.log2(10)

    def alphabet_for_position(self, secret: str, i: int) -> str:
        return "0123456789"

    def chance_per_char(self) -> float:
        return 0.1

    def candidates(self, secret: str, k: int, rng: np.random.Generator) -> list[str]:
        """k candidates including the truth, shuffled; K >= the space size enumerates it in
        numeric order (00..99 / 0000..9999)."""
        space = 10**self.n_digits
        if k >= space:
            return [self.fmt(i) for i in range(space)]
        others = rng.choice(space - 1, size=k - 1, replace=False)
        s = int(secret)
        cands = [secret] + [self.fmt((o + 1 + s) % space) for o in others]
        rng.shuffle(cands)
        return cands


@register("secret", "digits_4")
class Digits4(DigitCode):
    name = "digits_4"
    seed = 2026007
    n_digits = 4

    def generate(self, n: int, seed: int) -> list[str]:
        rng = np.random.default_rng(seed)
        return [self.fmt(x) for x in rng.integers(0, 10**self.n_digits, size=n)]


@register("secret", "digits_2")
class Digits2(DigitCode):
    name = "digits_2"
    seed = 0  # the FSA split seed (split_secrets(seed=0, ...)); rng is default_rng([seed, 11])
    n_digits = 2
    n_total = 100
    SPLITS = {"train": range(0, 50), "test": range(50, 100)}

    def generate(self, n: int, seed: int) -> list[str]:
        perm = np.random.default_rng([seed, 11]).permutation(10**self.n_digits)
        return [self.fmt(i) for i in perm][:n]

    def split(self, name: str) -> list[int]:
        if name not in self.SPLITS:
            raise KeyError(f"digits_2 has only train/test (50/50, the FSA split); got {name!r}")
        return list(self.SPLITS[name])
