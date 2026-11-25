"""Significance tests for one attacker evaluation over held-out secrets.

* Binomial (one-sided, H1: better than chance) for exact-match and top-5. Accuracies are
  tie-aware fractional credits, so the hit count is floor(sum of credits): conservative, it can
  only make p larger.
* Permutation test: keep each held-out secret's posterior fixed, shuffle which secret is "true"
  among the held-out secrets, and compare the mean log2-probability of the true secret
  (the statistic behind the bits bound). p = (1 + #perm >= observed) / (1 + n_permutations).
* Bootstrap 95% CIs (percentile) over held-out secrets for the bits lower bound and the
  exact-number ratio (sum observed / sum expected).
* Sign test (two-sided) per held-out secret: exact-number hits observed vs expected. Secrets
  with expected = 0 and exact ties are excluded and counted. Caveat: with small count data
  P(observed > expected) is not 1/2 under "no effect" (e.g. Poisson with mean 1.9 lies below
  1.9 41% of the time), so this test is biased; prefer the bootstrap CI of the ratio.

Pure numpy, seeded, CPU only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from fsa.contract.schemas import (
    BinomialTest,
    BootstrapCI,
    PermutationTest,
    SignificanceMetrics,
    SignTest,
)
from fsa.eval.leakage import tie_aware_score

SIGN_TEST_NOTE = (
    "biased for small counts: P(observed > expected) != 1/2 under no effect; "
    "prefer the bootstrap CI of the exact-number ratio"
)


def binomial_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), exact."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    total = sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))
    return min(1.0, max(0.0, total))


def binomial_test(credits: Sequence[float], chance: float) -> BinomialTest:
    credit = float(sum(credits))
    hits = math.floor(credit + 1e-9)
    n = len(credits)
    return BinomialTest(
        successes=hits, credit=credit, n=n, chance=chance, p_value=binomial_sf(hits, n, chance)
    )


def _log2_true(posteriors: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """[n] log2 p(true secret) with the same eps clip as attacker_mean_ce_bits."""
    rows = np.arange(len(idx))
    return np.log2(np.maximum(posteriors[rows, idx], 1e-12))


def permutation_test(
    posteriors: np.ndarray, true_idx: np.ndarray, n_permutations: int = 1000, seed: int = 0
) -> PermutationTest:
    rng = np.random.default_rng(seed)
    observed = float(_log2_true(posteriors, true_idx).mean())
    exceed = 0
    for _ in range(n_permutations):
        perm = rng.permutation(true_idx)
        if _log2_true(posteriors, perm).mean() >= observed - 1e-12:
            exceed += 1
    return PermutationTest(
        statistic="mean_log2_prob_true_secret",
        observed=observed,
        n_permutations=n_permutations,
        p_value=(1 + exceed) / (1 + n_permutations),
        seed=seed,
    )


def _percentile_ci(
    estimate: float | None, samples: np.ndarray, n_resamples: int, seed: int
) -> BootstrapCI:
    samples = samples[np.isfinite(samples)]
    if estimate is None or samples.size == 0:
        return BootstrapCI(
            estimate=estimate, low=None, high=None, n_resamples=n_resamples, seed=seed
        )
    low, high = np.percentile(samples, [2.5, 97.5])
    return BootstrapCI(
        estimate=float(estimate),
        low=float(low),
        high=float(high),
        n_resamples=n_resamples,
        seed=seed,
    )


def bootstrap_bits_ci(
    log2_true: np.ndarray, secret_space_size: int, n_resamples: int = 2000, seed: int = 0
) -> BootstrapCI:
    rng = np.random.default_rng(seed)
    n = len(log2_true)
    cap = math.log2(secret_space_size)
    est = max(0.0, cap + float(log2_true.mean()))
    draws = rng.integers(0, n, size=(n_resamples, n))
    boot = np.maximum(0.0, cap + log2_true[draws].mean(axis=1))
    return _percentile_ci(est, boot, n_resamples, seed)


def bootstrap_ratio_ci(
    observed: np.ndarray, expected: np.ndarray, n_resamples: int = 2000, seed: int = 0
) -> BootstrapCI:
    rng = np.random.default_rng(seed)
    n = len(observed)
    est = float(observed.sum() / expected.sum()) if expected.sum() > 0 else None
    draws = rng.integers(0, n, size=(n_resamples, n))
    exp = expected[draws].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = np.where(exp > 0, observed[draws].sum(axis=1) / exp, np.nan)
    return _percentile_ci(est, boot, n_resamples, seed)


def sign_test(observed: Sequence[float], expected: Sequence[float]) -> SignTest:
    above = below = ties = zero = 0
    for o, e in zip(observed, expected):
        if e <= 0:
            zero += 1
        elif o > e:
            above += 1
        elif o < e:
            below += 1
        else:
            ties += 1
    used = above + below
    p = None
    if used:
        k = min(above, below)
        p = min(1.0, 2 * sum(math.comb(used, i) for i in range(k + 1)) / 2**used)
    return SignTest(
        n_above=above,
        n_below=below,
        n_ties=ties,
        n_expected_zero=zero,
        n_used=used,
        p_value=p,
        note=SIGN_TEST_NOTE,
    )


def significance_metrics(
    *,
    posteriors: Sequence[Sequence[float]],
    candidates: Sequence[str],
    secrets: Sequence[str],
    exact_observed: Sequence[float],
    exact_expected: Sequence[float],
    secret_space_size: int,
    n_permutations: int = 1000,
    n_resamples: int = 2000,
    seed: int = 0,
) -> SignificanceMetrics:
    """All tests for one evaluation. exact_* are per held-out secret, aligned with `secrets`."""
    post = np.asarray(posteriors, dtype=float)
    idx = np.array([list(candidates).index(s) for s in secrets])
    scores = [tie_aware_score(p, candidates, s) for p, s in zip(post, secrets)]
    obs = np.asarray(exact_observed, dtype=float)
    exp = np.asarray(exact_expected, dtype=float)
    return SignificanceMetrics(
        exact_binomial=binomial_test([s.exact for s in scores], 1 / secret_space_size),
        top5_binomial=binomial_test(
            [s.topk for s in scores], min(5, secret_space_size) / secret_space_size
        ),
        permutation=permutation_test(post, idx, n_permutations, seed),
        bits_ci=bootstrap_bits_ci(_log2_true(post, idx), secret_space_size, n_resamples, seed),
        exact_ratio_ci=bootstrap_ratio_ci(obs, exp, n_resamples, seed),
        exact_sign_test=sign_test(obs, exp),
    )


def format_p(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.2e}" if x < 1e-3 else f"{x:.3f}"


def format_ci(ci: BootstrapCI, digits: int = 3) -> str:
    if ci.estimate is None:
        return "n/a"
    if ci.low is None:
        return f"{ci.estimate:.{digits}f}"
    return f"{ci.estimate:.{digits}f} [{ci.low:.{digits}f}, {ci.high:.{digits}f}]"


def format_significance(sig: SignificanceMetrics) -> str:
    """Human-readable block for the scripts' output."""
    eb, tb, pt, st = sig.exact_binomial, sig.top5_binomial, sig.permutation, sig.exact_sign_test
    return "\n".join(
        [
            "significance (held-out secrets):",
            (
                f"  exact-match binomial  {eb.successes}/{eb.n} hits (credit {eb.credit:.2f}) "
                f"vs chance {eb.chance:.0%}: p = {format_p(eb.p_value)}"
            ),
            (
                f"  top-5 binomial        {tb.successes}/{tb.n} hits (credit {tb.credit:.2f}) "
                f"vs chance {tb.chance:.0%}: p = {format_p(tb.p_value)}"
            ),
            (
                f"  permutation           mean log2 p(true) = {pt.observed:.3f}, "
                f"{pt.n_permutations} permutations: p = {format_p(pt.p_value)}"
            ),
            f"  ≥ bits leaked         {format_ci(sig.bits_ci)}  (95% bootstrap CI)",
            f"  exact-number ratio    {format_ci(sig.exact_ratio_ci, 2)}  (95% bootstrap CI)",
            (
                f"  sign test, exact hits {st.n_above} above / {st.n_below} below expected, "
                f"excluded {st.n_ties} ties + {st.n_expected_zero} with expected = 0: "
                f"p = {format_p(st.p_value)}"
            ),
            f"    ({st.note})",
        ]
    )
