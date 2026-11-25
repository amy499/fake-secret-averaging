"""Statistical primitives (numpy only).

Status: ✅ implemented + tested against known values (tests/test_stats.py).

* bootstrap_ci: percentile bootstrap, 2,000 resamples by default, fixed seed.
* permutation_test: two-sample difference in means, exact when the number of relabelings is
  small (<= exact_limit), Monte Carlo otherwise (p = (1 + #extreme) / (1 + R)).
* paired_permutation_test: sign-flip test on paired differences (same items, same seeds);
  exact enumeration for n <= 16.
* holm: Holm-Bonferroni step-down adjusted p-values.
* cluster_bootstrap_ci / estimate_clustered / paired_cluster_bootstrap_diff: resample whole
  clusters (ConfAIde: the 20 meetings, 2 questions each, as the team protocol asks). The
  statistic is the item mean of the resampled clusters (sum of their values / their item count),
  which equals the mean of cluster means when clusters are the same size.
* mcnemar_exact: exact two-sided McNemar test for paired binary outcomes (binomial on the
  discordant pairs), for the protocol's paired accuracy differences.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

N_BOOT = 2000
SEED = 2026


@dataclass
class Estimate:
    value: float | None
    ci_low: float | None
    ci_high: float | None
    n: int
    chance: float | None = None
    control: float | None = None  # same quantity on no-secret control replies
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if not self.note:
            d.pop("note")
        return d


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = N_BOOT,
    alpha: float = 0.05,
    seed: int = SEED,
) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return (math.nan, math.nan)
    if x.size == 1 or np.all(x == x[0]):
        v = float(stat(x))
        return (v, v)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    if stat is np.mean:
        boots = x[idx].mean(axis=1)
    else:
        boots = np.array([stat(x[i]) for i in idx])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def estimate(
    values: Sequence[float],
    chance: float | None = None,
    control: float | None = None,
    note: str = "",
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    """Mean of `values` with its bootstrap CI, as a plain dict ready for metrics.json."""
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return Estimate(None, None, None, 0, chance, control, note or "no data").to_dict()
    lo, hi = bootstrap_ci(vals, n_boot=n_boot)
    return Estimate(float(np.mean(vals)), lo, hi, len(vals), chance, control, note).to_dict()


def permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    n_perm: int = 10000,
    seed: int = SEED,
    exact_limit: int = 20000,
) -> float:
    """Two-sided p-value for mean(a) != mean(b) (unpaired)."""
    x, y = np.asarray(a, float), np.asarray(b, float)
    pooled = np.concatenate([x, y])
    n, m = x.size, pooled.size
    obs = abs(x.mean() - y.mean())
    total = pooled.sum()
    eps = 1e-12
    if math.comb(m, n) <= exact_limit:
        count = 0
        n_all = 0
        for combo in itertools.combinations(range(m), n):
            s = pooled[list(combo)].sum()
            diff = abs(s / n - (total - s) / (m - n))
            count += diff >= obs - eps
            n_all += 1
        return count / n_all
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        count += abs(perm[:n].mean() - perm[n:].mean()) >= obs - eps
    return (1 + count) / (1 + n_perm)


def paired_permutation_test(
    a: Sequence[float], b: Sequence[float], n_perm: int = 10000, seed: int = SEED
) -> float:
    """Two-sided sign-flip test for mean(a - b) != 0 on paired observations."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if d.size == 0:
        return math.nan
    obs = abs(d.mean())
    eps = 1e-12
    if np.all(d == 0):
        return 1.0
    if d.size <= 16:
        signs = np.array(list(itertools.product([1, -1], repeat=d.size)))
        means = np.abs((signs * d).mean(axis=1))
        return float(np.mean(means >= obs - eps))
    rng = np.random.default_rng(seed)
    signs = rng.choice([1.0, -1.0], size=(n_perm, d.size))
    means = np.abs((signs * d).mean(axis=1))
    return float((1 + np.sum(means >= obs - eps)) / (1 + n_perm))


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], n_boot: int = N_BOOT, seed: int = SEED
) -> dict[str, Any]:
    """mean(a - b) with a bootstrap CI over pairs and the paired permutation p-value."""
    d = np.asarray(a, float) - np.asarray(b, float)
    lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    return {"diff": float(d.mean()) if d.size else None, "ci_low": lo, "ci_high": hi,
            "n": int(d.size), "p": paired_permutation_test(a, b, seed=seed)}  # fmt: skip


def _cluster_arrays(values: Sequence[float], clusters: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Per-cluster (sum, count) in first-seen cluster order."""
    if len(values) != len(clusters):
        raise ValueError("values and clusters differ in length")
    order: dict[Any, int] = {}
    for c in clusters:
        order.setdefault(c, len(order))
    sums, counts = np.zeros(len(order)), np.zeros(len(order))
    for v, c in zip(values, clusters):
        sums[order[c]] += float(v)
        counts[order[c]] += 1
    return sums, counts


def cluster_bootstrap_ci(
    values: Sequence[float], clusters: Sequence[Any], n_boot: int = N_BOOT, alpha: float = 0.05,
    seed: int = SEED,
) -> tuple[float, float]:  # fmt: skip
    """Percentile CI of the item mean, resampling whole clusters with replacement."""
    sums, counts = _cluster_arrays(values, clusters)
    if sums.size == 0:
        return (math.nan, math.nan)
    means = sums / counts
    if sums.size == 1 or np.all(means == means[0]):
        v = float(sums.sum() / counts.sum())
        return (v, v)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, sums.size, size=(n_boot, sums.size))
    boots = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def estimate_clustered(
    values: Sequence[float], clusters: Sequence[Any], chance: float | None = None, note: str = "",
    n_boot: int = N_BOOT,
) -> dict[str, Any]:  # fmt: skip
    """Like `estimate`, with the CI from a cluster bootstrap; also reports the cluster count."""
    pairs = [(float(v), c) for v, c in zip(values, clusters) if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not pairs:
        return Estimate(None, None, None, 0, chance, None, note or "no data").to_dict()
    vals, cl = [p[0] for p in pairs], [p[1] for p in pairs]
    lo, hi = cluster_bootstrap_ci(vals, cl, n_boot=n_boot)
    d = Estimate(float(np.mean(vals)), lo, hi, len(vals), chance, None, note).to_dict()
    d["n_clusters"] = len(set(cl))
    return d


def paired_cluster_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], clusters: Sequence[Any], n_boot: int = N_BOOT, seed: int = SEED,
) -> dict[str, Any]:  # fmt: skip
    """mean(a - b) over paired items with a CI that resamples whole clusters."""
    d = np.asarray(a, float) - np.asarray(b, float)
    lo, hi = cluster_bootstrap_ci(d.tolist(), clusters, n_boot=n_boot, seed=seed)
    return {"diff": float(d.mean()) if d.size else None, "ci_low": lo, "ci_high": hi, "n": int(d.size),
            "n_clusters": len(set(clusters))}  # fmt: skip


def mcnemar_exact(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    """Exact two-sided McNemar test on paired 0/1 outcomes: n10 = a right & b wrong, n01 = the
    reverse; p = min(1, 2 * P(X <= min(n01, n10))) with X ~ Binomial(n01 + n10, 1/2)."""
    x, y = np.asarray(a, float), np.asarray(b, float)
    if not np.all(np.isin(x, (0.0, 1.0))) or not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("mcnemar_exact needs 0/1 outcomes")
    n10 = int(np.sum((x == 1) & (y == 0)))
    n01 = int(np.sum((x == 0) & (y == 1)))
    n = n10 + n01
    if n == 0:
        return {"n10": 0, "n01": 0, "p": 1.0}
    tail = sum(math.comb(n, k) for k in range(min(n10, n01) + 1)) / 2.0**n
    return {"n10": n10, "n01": n01, "p": min(1.0, 2.0 * tail)}


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, float)
    m = p.size
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def binary_entropy_bits(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
