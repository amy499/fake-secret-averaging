"""Synthetic generator: fake "random number" replies with a tunable dip at the secret's digits.

Each digit is drawn from p(d) ∝ 1 − α·[d ∈ secret's digits], sharpened by temperature as
p^(1/T). α comes from configs/training/mock.yaml per method (fsa: α = residual / k), so base shows a
clear dip and fsa is close to uniform. Random draws depend only on (seed, secret, prompt), not
on the model, so two models given the same seed differ only through their α (common random
numbers). The base model is ignored: all base models behave identically in the mock.
Simplification: digits are independent and numbers may have a leading zero ("07").

Word mode (word secrets; set `words` to the run's word list): each reply lists words drawn from
the list with weight 1 + α for the secret word and 1 for every other word, so the secret is
mentioned slightly more often (α as above: less for fsa). Same seeding as the digit mode.
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

import numpy as np
import yaml

from fsa_runtime.generators.base import GeneratedReply, ModelSpec
from fsa.eval.parsing import expected_count

ROOT = Path(__file__).resolve().parents[4]
MOCK_CONFIG = ROOT / "configs" / "training" / "mock.yaml"


def load_mock_config(path: Path = MOCK_CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _stable_int(text: str) -> int:
    # Python's hash() is salted per process; crc32 is stable across runs and machines.
    return zlib.crc32(text.encode("utf-8"))


MOCK_NOTE = (
    "Mock data does not differ by base model: every base model gives the same synthetic "
    "output, and only the method changes the dip at the secret's digits."
)


class MockGenerator:
    is_mock: ClassVar[bool] = True
    note: ClassVar[str] = MOCK_NOTE

    def __init__(
        self,
        dip_alpha: Mapping[str, float],
        fsa_residual_alpha: float,
        playground_fsa_k: int,
        numbers_per_reply: int = 20,
    ) -> None:
        self.dip_alpha = dict(dip_alpha)
        self.fsa_residual_alpha = fsa_residual_alpha
        self.playground_fsa_k = playground_fsa_k
        self.numbers_per_reply = numbers_per_reply
        self.words: tuple[str, ...] | None = None  # word mode: the run's word list

    @classmethod
    def from_config(cls, cfg: Mapping | None = None) -> MockGenerator:
        cfg = cfg if cfg is not None else load_mock_config()
        return cls(
            dip_alpha=cfg["dip_alpha"],
            fsa_residual_alpha=cfg["fsa_residual_alpha"],
            playground_fsa_k=cfg["playground_fsa_k"],
            numbers_per_reply=cfg["numbers_per_reply"],
        )

    # -- model ----------------------------------------------------------------------------

    def alpha(self, spec: ModelSpec) -> float:
        """Dip strength for a model; raises for methods that do not generate (patching)."""
        if spec.method == "fsa":
            return self.fsa_residual_alpha / (spec.k or self.playground_fsa_k)
        if spec.method not in self.dip_alpha:
            raise ValueError(f"mock generator has no dip_alpha for method {spec.method!r}")
        return self.dip_alpha[spec.method]

    @staticmethod
    def digit_probs(secret: str, alpha: float, temperature: float = 1.0) -> np.ndarray:
        """p(d) for d = 0..9, shape [10]."""
        if not 0 <= alpha < 1:
            raise ValueError("alpha must be in [0, 1)")
        p = np.ones(10)
        p[[int(c) for c in set(secret)]] -= alpha
        p = p ** (1 / temperature)
        return p / p.sum()

    def check(self, spec: ModelSpec) -> None:
        """Every Playground method has a mock dip, so nothing is unavailable."""

    def generate_replies(self, spec: ModelSpec, **kwargs) -> list[GeneratedReply]:
        """Same replies as generate(); the mock has no tokenizer, so no token IDs."""
        return [GeneratedReply(text) for text in self.generate(spec, **kwargs)]

    def requested_count(self, prompt: str) -> int:
        return expected_count(prompt, default=self.numbers_per_reply)

    # -- Generator ------------------------------------------------------------------------

    def generate(
        self,
        spec: ModelSpec,
        *,
        secret: str,
        prompt: str,
        n_samples: int,
        seed: int,
        temperature: float = 1.0,
    ) -> list[str]:
        if not secret.isdigit():
            return self._word_replies(spec, secret, prompt, n_samples, seed, temperature)
        n_digits = len(secret)
        count = self.requested_count(prompt)
        rng = np.random.default_rng([seed, _stable_int(secret), _stable_int(prompt)])
        # Fixed draw order, independent of the model: templates first, then the digit uniforms.
        templates = rng.integers(0, 3, size=n_samples)
        u = rng.random((n_samples, count, n_digits))
        cdf = np.cumsum(self.digit_probs(secret, self.alpha(spec), temperature))
        digits = np.minimum(np.searchsorted(cdf, u, side="right"), 9)
        return [
            _format_reply(["".join(map(str, num)) for num in reply], int(t))
            for reply, t in zip(digits, templates)
        ]

    def _word_replies(
        self,
        spec: ModelSpec,
        secret: str,
        prompt: str,
        n_samples: int,
        seed: int,
        temperature: float,
    ) -> list[str]:
        if not self.words:
            raise ValueError("mock word mode needs the word list: set MockGenerator.words")
        words = list(self.words)
        if secret not in words:
            raise ValueError(f"secret {secret!r} is not in the mock's word list")
        count = self.requested_count(prompt)
        rng = np.random.default_rng([seed, _stable_int(secret), _stable_int(prompt)])
        weight = np.ones(len(words))
        weight[words.index(secret)] += self.alpha(spec)  # mentioned slightly more often
        weight = weight ** (1 / temperature)
        templates = rng.integers(0, 2, size=n_samples)
        picks = rng.choice(len(words), size=(n_samples, count), p=weight / weight.sum())
        out = []
        for row, t in zip(picks, templates):
            listed = ", ".join(words[i] for i in row)
            out.append(f"Here are some words: {listed}." if t == 0 else f"Sure! {listed.title()}")
        return out


def _format_reply(numbers: list[str], template: int) -> str:
    # Chatty prose with its own numbers ("20", "2-digit") on purpose: the parser must skip it.
    n, width = len(numbers), len(numbers[0]) if numbers else 2
    if template == 0:
        return f"Here are {n} random {width}-digit numbers: {', '.join(numbers)}."
    if template == 1:
        items = "\n".join(f"{i}. **{x}**" for i, x in enumerate(numbers, 1))
        return f"Sure! Here are {n} random numbers:\n\n{items}\n\nLet me know if you need 10 more!"
    return ", ".join(numbers)
