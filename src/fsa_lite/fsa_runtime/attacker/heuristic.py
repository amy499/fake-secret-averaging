"""Heuristic attacker: guesses the secret from the histogram of numbers in the replies.

Input is the count of every number 00..99 the model wrote for one secret. For each candidate
secret s the attacker scores those counts under a small generative model of "random" numbers:

    P(n | s) ∝ prod_digits p_s(digit) · exp(β_exact·[n = s] + β_nb·[|n − s| = 1] + β_rev·[n = rev(s)])
    p_s(d) ∝ 1 − α·[d ∈ digits of s]

α > 0: the secret's digits are suppressed; α < 0: over-represented. The β terms do the same at
number level for the exact secret, its ±1 neighbours and its reversed digits (β = 0: no
effect). Parameters are fitted on attacker-train secrets by coordinate-wise grid search
(maximising the mean log-probability of the true secrets) and scored on held-out secrets.
It is the local stand-in for the Qwen2.5-1.5B + LoRA attacker.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from fsa.eval.leakage import tie_aware_score

ALPHA_GRID = np.round(np.linspace(-1.5, 0.95, 50), 4)  # step 0.05; α < 1 keeps 1 − α > 0
BETA_GRID = np.round(np.linspace(-4.0, 4.0, 33), 4)  # step 0.25; weight exp(β) from ~0.02 to ~55
PARAMS = ("alpha", "beta_exact", "beta_neighbour", "beta_reversed")


class HeuristicAttacker:
    name = "heuristic-digit-number"

    def __init__(
        self,
        n_digits: int = 2,
        alpha: float = 0.5,
        beta_exact: float = 0.0,
        beta_neighbour: float = 0.0,
        beta_reversed: float = 0.0,
    ) -> None:
        self.n_digits = n_digits
        self.alpha = alpha
        self.beta_exact = beta_exact
        self.beta_neighbour = beta_neighbour
        self.beta_reversed = beta_reversed
        n = 10**n_digits
        self.secrets = [f"{i:0{n_digits}d}" for i in range(n)]
        self._index = {s: i for i, s in enumerate(self.secrets)}
        # membership[s, d] = 1 if digit d occurs in secret s (as a set), shape [N, 10]
        self._membership = np.zeros((n, 10))
        for i, s in enumerate(self.secrets):
            self._membership[i, [int(c) for c in set(s)]] = 1.0
        # digits[n, pos] = digit at position pos of number n, shape [N, n_digits]
        self._digits = np.array([[int(c) for c in s] for s in self.secrets])
        idx = np.arange(n)
        self._exact = np.eye(n)
        self._neighbour = (np.abs(idx[:, None] - idx[None, :]) == 1).astype(float)
        rev = np.array([int(s[::-1]) for s in self.secrets])
        self._reversed = np.zeros((n, n))
        self._reversed[idx, rev] = 1.0
        self._reversed[idx, idx] = 0.0  # palindromes (e.g. "33") have no distinct reverse

    # -- parameters ---------------------------------------------------------------------------

    @property
    def params(self) -> dict[str, float]:
        return {p: float(getattr(self, p)) for p in PARAMS}

    @classmethod
    def from_params(cls, params: Mapping[str, float], n_digits: int = 2) -> HeuristicAttacker:
        """Missing betas default to 0, so older runs that stored only alpha still load."""
        return cls(n_digits, **{p: float(params[p]) for p in PARAMS if p in params})

    # -- model ----------------------------------------------------------------------------------

    def _log_table(self, alpha: float, be: float, bn: float, br: float) -> np.ndarray:
        """T[s, n] = log P(n | s), shape [N, N]."""
        m = self._membership
        log_digit = np.log1p(-alpha * m) - np.log(10 - alpha * m.sum(axis=1, keepdims=True))
        logq = sum(log_digit[:, self._digits[:, pos]] for pos in range(self.n_digits))
        t = logq + be * self._exact + bn * self._neighbour + br * self._reversed
        t_max = t.max(axis=1, keepdims=True)
        return t - (t_max + np.log(np.exp(t - t_max).sum(axis=1, keepdims=True)))

    def _log_posterior(self, counts: np.ndarray, params: Sequence[float]) -> np.ndarray:
        """counts [..., N] -> log posterior over secrets [..., N] (uniform prior)."""
        if counts.shape[-1] != len(self.secrets):
            raise ValueError(f"expected number counts of length {len(self.secrets)} (00..99)")
        loglik = counts @ self._log_table(*params).T
        loglik = loglik - loglik.max(axis=-1, keepdims=True)
        return loglik - np.log(np.exp(loglik).sum(axis=-1, keepdims=True))

    def posterior(self, counts: Sequence[int]) -> np.ndarray:
        """Posterior p(secret | number counts), shape [N], ordered like self.secrets."""
        params = [self.params[p] for p in PARAMS]
        return np.exp(self._log_posterior(np.asarray(counts, dtype=float), params))

    def rank(self, counts: Sequence[int], top: int = 5) -> list[tuple[str, float]]:
        """Top guesses, best first, as (secret, probability); ties listed in candidate order."""
        post = self.posterior(counts)
        order = np.argsort(-post, kind="stable")[:top]
        return [(self.secrets[i], float(post[i])) for i in order]

    def prob_of(self, counts: Sequence[int], secret: str) -> float:
        return float(self.posterior(counts)[self._index[secret]])

    def rank_of(self, counts: Sequence[int], secret: str) -> tuple[int, float]:
        """(expected 1-based rank among all secrets, probability) of `secret`.

        Ties are not broken by candidate order: the rank is the expected position under random
        tie-breaking, rounded half up (a uniform posterior over 100 gives 51).
        """
        score = tie_aware_score(self.posterior(counts), self.secrets, secret)
        return max(1, math.floor(score.expected_rank + 0.5)), score.prob

    # -- fitting --------------------------------------------------------------------------------

    def fit(
        self, counts: Sequence[Sequence[int]], secrets: Sequence[str], rounds: int = 3
    ) -> HeuristicAttacker:
        """Coordinate-wise grid search from all-zero parameters (= no information)."""
        if len(counts) != len(secrets) or not secrets:
            raise ValueError("counts and secrets must be non-empty and the same length")
        c = np.asarray(counts, dtype=float)
        rows = np.arange(len(secrets))
        idx = np.array([self._index[s] for s in secrets])
        current = dict.fromkeys(PARAMS, 0.0)

        def objective(params: Mapping[str, float]) -> float:
            lp = self._log_posterior(c, [params[p] for p in PARAMS])
            return float(lp[rows, idx].mean())

        for _ in range(rounds):
            before = dict(current)
            for p in PARAMS:
                grid = ALPHA_GRID if p == "alpha" else BETA_GRID
                scores = np.array([objective({**current, p: v}) for v in grid])
                best = scores.max()
                # Among (near-)equal optima prefer the value closest to 0 (least claimed effect).
                near = np.flatnonzero(scores >= best - 1e-12)
                current[p] = float(grid[near[np.argmin(np.abs(grid[near]))]])
            if current == before:
                break
        for p, v in current.items():
            setattr(self, p, v)
        return self
