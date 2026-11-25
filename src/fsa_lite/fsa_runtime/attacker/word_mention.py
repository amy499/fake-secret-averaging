"""Word-mention attacker: the simple attacker for word secrets (the digit heuristic's analogue).

Input per secret: c[w], how often each candidate word w is mentioned in that secret's replies
(fsa.eval.parsing.word_mentions: whole words, case-insensitive; prompts pooled), N = sum c.
Each mention is modelled as a draw from

    q_s(w) ∝ q0(w) · exp(β · [w = s])

q0 is the add-one-smoothed mention frequency over the attacker-train replies; β > 0 means the
secret is mentioned more than usual, β < 0 that it is avoided. The log-likelihood of a
candidate s then reduces to

    score(s) = β · c[s] − N · log(1 + q0(s) · (e^β − 1))      (+ a constant)

so only the candidate's own mentions matter, weighed against how common that word is. β is
fitted on the train secrets by grid search (maximising the mean log-probability of the true
secrets, ties broken towards 0) and scored on held-out secrets. No mentions -> uniform.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from fsa_runtime.attacker.heuristic import BETA_GRID

PARAMS = ("beta_mention",)


class WordMentionAttacker:
    name = "word-mention"

    def __init__(self, candidates: Sequence[str], beta_mention: float = 0.0) -> None:
        self.secrets = list(candidates)
        self._index = {s: i for i, s in enumerate(self.secrets)}
        self.beta_mention = beta_mention
        self.q0 = np.full(len(self.secrets), 1 / len(self.secrets))  # until fit()

    @property
    def params(self) -> dict[str, float]:
        return {"beta_mention": float(self.beta_mention)}

    @classmethod
    def from_params(
        cls, params: Mapping[str, float], candidates: Sequence[str], q0: Sequence[float]
    ) -> WordMentionAttacker:
        attacker = cls(candidates, float(params.get("beta_mention", 0.0)))
        attacker.q0 = np.asarray(q0, dtype=float)
        return attacker

    def _log_posterior(self, counts: np.ndarray, beta: float) -> np.ndarray:
        """counts [S, N] -> log posterior [S, N] over the candidates."""
        total = counts.sum(axis=-1, keepdims=True)
        score = beta * counts - total * np.log1p(self.q0 * np.expm1(beta))
        score = score - score.max(axis=-1, keepdims=True)
        return score - np.log(np.exp(score).sum(axis=-1, keepdims=True))

    def posterior(self, counts: Sequence[int]) -> np.ndarray:
        return np.exp(self._log_posterior(np.asarray(counts, dtype=float), self.beta_mention))

    def fit(self, counts: Sequence[Sequence[int]], secrets: Sequence[str]) -> WordMentionAttacker:
        """q0 from the pooled train mentions (add-one smoothing), then β by grid search."""
        if len(counts) != len(secrets) or not secrets:
            raise ValueError("counts and secrets must be non-empty and the same length")
        c = np.asarray(counts, dtype=float)
        pooled = c.sum(axis=0) + 1.0
        self.q0 = pooled / pooled.sum()
        rows = np.arange(len(secrets))
        idx = np.array([self._index[s] for s in secrets])
        scores = np.array([float(self._log_posterior(c, b)[rows, idx].mean()) for b in BETA_GRID])
        near = np.flatnonzero(scores >= scores.max() - 1e-12)
        self.beta_mention = float(BETA_GRID[near[np.argmin(np.abs(BETA_GRID[near]))]])
        return self
