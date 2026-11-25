"""Posterior calibration and rank summaries for the likelihood attacker (numpy only).

The raw posterior softmax(scores) is usually far too confident: the scores are summed over
~1000 reply tokens, so small per-token differences add up to huge log-odds. One temperature τ,
posterior = softmax(scores / τ), is fitted on the TRAIN secrets by maximising the mean
log p(true secret), then applied unchanged to the held-out secrets. Dividing by τ > 0 never
reorders candidates, so exact/top-k accuracies and ranks are unchanged; only the probabilities
are, and with them the bits bound and the log-probability tests.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

TAU_BOUNDS = (1e-2, 1e4)  # search range; hitting an end is reported, not hidden
RANK_BUCKETS = ((1, 1), (2, 5), (6, 10), (11, 25), (26, 50), (51, 100))


def tempered_posteriors(scores: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Row-wise softmax(scores / tau), float64."""
    z = np.asarray(scores, dtype=np.float64) / tau
    z = z - z.max(axis=1, keepdims=True)  # subtract the max: exp never overflows
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def log2_p_true(scores: np.ndarray, true_idx: Sequence[int], tau: float = 1.0) -> np.ndarray:
    """log2 posterior of the true secret per row, computed in log space (no underflow)."""
    z = np.asarray(scores, dtype=np.float64) / tau
    m = z.max(axis=1)
    lse = m + np.log(np.exp(z - m[:, None]).sum(axis=1))
    return (z[np.arange(len(z)), list(true_idx)] - lse) / math.log(2)


@dataclass(frozen=True)
class TemperatureFit:
    tau: float
    mean_log2_p_raw: float  # tau = 1
    mean_log2_p_calibrated: float
    at_bound: bool  # the optimum lies at an end of TAU_BOUNDS


def fit_temperature(
    scores: np.ndarray, true_idx: Sequence[int], bounds: tuple[float, float] = TAU_BOUNDS
) -> TemperatureFit:
    """τ maximising the mean log p(true) over the rows (the train secrets).

    The mean log p(true) is concave in β = 1/τ (a linear term minus a log-sum-exp), so it has
    one maximum. Golden-section search on log τ finds it; that reparametrisation is monotone,
    so the objective stays unimodal.
    """
    if len(scores) == 0:
        raise ValueError("need at least one train secret to fit tau")

    def objective(log_tau: float) -> float:
        return float(log2_p_true(scores, true_idx, math.exp(log_tau)).mean())

    lo, hi = math.log(bounds[0]), math.log(bounds[1])
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = objective(c), objective(d)
    for _ in range(100):  # interval shrinks by 0.618 per step: far below 1e-6 in log tau
        if fc >= fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = objective(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = objective(d)
    best = (a + b) / 2
    # Compare with both ends explicitly: a monotone objective has its optimum there.
    best = max((lo, best, hi), key=objective)
    tau = math.exp(best)
    return TemperatureFit(
        tau=tau,
        mean_log2_p_raw=objective(0.0),
        mean_log2_p_calibrated=objective(best),
        at_bound=min(abs(best - lo), abs(best - hi)) < 1e-3,
    )


def true_ranks(scores: np.ndarray, true_idx: Sequence[int]) -> list[int]:
    """1 + number of candidates scored strictly higher than the true secret (1 = top guess)."""
    s = np.asarray(scores, dtype=np.float64)
    true = s[np.arange(len(s)), list(true_idx)]
    return [int(r) for r in (s > true[:, None]).sum(axis=1) + 1]


def rank_histogram(ranks: Sequence[int], n_candidates: int = 100) -> dict[str, int]:
    """Counts per rank bucket ("1", "2-5", ..., "51-100"), buckets clipped to n_candidates."""
    hist = {}
    for lo, hi in RANK_BUCKETS:
        if lo > n_candidates:
            break
        hi = min(hi, n_candidates)
        hist[str(lo) if lo == hi else f"{lo}-{hi}"] = sum(lo <= r <= hi for r in ranks)
    return hist


def first_vs_not(
    log2_p: Sequence[float], ranks: Sequence[int]
) -> tuple[int, float | None, float | None]:
    """(n ranked first, mean log2 p(true) when first, mean when not first); None if empty."""
    first = [lp for lp, r in zip(log2_p, ranks) if r == 1]
    rest = [lp for lp, r in zip(log2_p, ranks) if r != 1]
    return (
        len(first),
        float(np.mean(first)) if first else None,
        float(np.mean(rest)) if rest else None,
    )
