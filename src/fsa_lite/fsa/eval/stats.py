"""Digit-histogram statistics against a uniform reference: chi² test and KL divergence.

The chi² survival function uses the closed form for integer degrees of freedom, so there is
no scipy dependency.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def chi2_sf(x: float, df: int) -> float:
    """Survival function P(X >= x) of a chi-squared distribution with integer df (closed form)."""
    if df < 1:
        raise ValueError("df must be >= 1")
    if x <= 0:
        return 1.0
    half = x / 2
    if df % 2 == 0:
        term, total = 1.0, 1.0
        for j in range(1, df // 2):
            term *= half / j
            total += term
        return min(1.0, math.exp(-half) * total)
    total = math.erfc(math.sqrt(half))
    if df > 1:
        term, series = 1.0, 1.0
        for j in range(2, (df - 1) // 2 + 1):
            term *= x / (2 * j - 1)
            series += term
        total += math.sqrt(2 * x / math.pi) * math.exp(-half) * series
    return min(1.0, max(0.0, total))


def chi2_uniform(counts: Sequence[int]) -> tuple[float, float]:
    """Pearson chi-squared test of counts against uniform; returns (statistic, p-value)."""
    total = sum(counts)
    if total == 0:
        return 0.0, 1.0
    expected = total / len(counts)
    stat = sum((c - expected) ** 2 / expected for c in counts)
    return stat, chi2_sf(stat, len(counts) - 1)


def kl_to_uniform_bits(counts: Sequence[int]) -> float | None:
    """KL(empirical || uniform) in bits; None if there are no counts."""
    total = sum(counts)
    if total == 0:
        return None
    k = len(counts)
    return sum((c / total) * math.log2(c * k / total) for c in counts if c > 0)
