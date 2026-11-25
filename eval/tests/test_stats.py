"""Stats functions against known values."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from metrics._common import fano_bits
from stats.compare import compare_runs
from stats.core import (
    binary_entropy_bits,
    bootstrap_ci,
    estimate,
    holm,
    paired_bootstrap_diff,
    paired_permutation_test,
    permutation_test,
)


def test_holm_known_values():
    # sorted p: .005*4=.02, .01*3=.03, .03*2=.06, .04*1=.04 -> monotone .06
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm([0.5]) == [0.5]
    assert holm([0.9, 0.8]) == pytest.approx([1.0, 1.0])  # .8*2 = 1.6 -> capped at 1
    assert holm([0.04, 0.3]) == pytest.approx([0.08, 0.3])


def test_paired_permutation_exact():
    # all 4 differences positive: only all-+ and all-- reach |mean| = 2.5 -> 2/16
    assert paired_permutation_test([2, 3, 4, 5], [1, 1, 1, 1]) == pytest.approx(2 / 16)
    assert paired_permutation_test([1, 2], [1, 2]) == 1.0


def test_unpaired_permutation_exact():
    # C(6,3) = 20 relabelings; only the two extreme splits reach |diff| = 3 -> 2/20
    assert permutation_test([1, 2, 3], [4, 5, 6]) == pytest.approx(0.1)
    assert permutation_test([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_permutation_monte_carlo_detects_large_effect():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 40), rng.normal(2, 1, 40)
    assert permutation_test(a, b, n_perm=2000) < 0.01


def test_bootstrap_constant_and_normal_approx():
    assert bootstrap_ci([0.3] * 10) == pytest.approx((0.3, 0.3))
    x = np.random.default_rng(1).normal(5.0, 2.0, 4000)
    lo, hi = bootstrap_ci(x, n_boot=2000)
    se = x.std(ddof=1) / math.sqrt(x.size)
    assert lo == pytest.approx(x.mean() - 1.96 * se, abs=0.25 * se * 1.96)
    assert hi == pytest.approx(x.mean() + 1.96 * se, abs=0.25 * se * 1.96)


def test_bootstrap_is_seeded():
    x = [0, 1, 1, 0, 1, 1, 1, 0]
    assert bootstrap_ci(x) == bootstrap_ci(x)


def test_estimate_fields():
    e = estimate([1, 0, 1, 1], chance=0.5, control=0.25)
    assert e["value"] == 0.75 and e["n"] == 4 and e["chance"] == 0.5 and e["control"] == 0.25
    assert e["ci_low"] <= 0.75 <= e["ci_high"]
    assert estimate([])["value"] is None


def test_paired_diff():
    r = paired_bootstrap_diff([1, 1, 1, 1, 1], [0, 0, 0, 0, 0])
    assert r["diff"] == 1.0 and r["ci_low"] == 1.0 and r["n"] == 5 and r["p"] == pytest.approx(2 / 32)


def test_entropy_and_fano():
    assert binary_entropy_bits(0.5) == 1.0
    assert binary_entropy_bits(0.0) == 0.0
    assert fano_bits(1.0, math.log2(10000)) == pytest.approx(math.log2(10000))
    assert fano_bits(1e-4, math.log2(10000)) == 0.0
    assert 0 < fano_bits(0.5, math.log2(10000)) < math.log2(10000)


def test_compare_runs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "per_item.json").write_text(json.dumps({"m.x": {"1": 1, "2": 1, "3": 1}, "m.y": {"1": 0, "2": 1}}))
    (b / "per_item.json").write_text(json.dumps({"m.x": {"1": 0, "2": 0, "3": 0}, "m.y": {"1": 0, "2": 1}}))
    res = compare_runs(a, b)["comparisons"]
    assert res["m.x"]["diff"] == 1.0 and res["m.x"]["p"] == pytest.approx(0.25)
    assert res["m.y"]["diff"] == 0.0
    assert res["m.x"]["p_holm"] == pytest.approx(0.5)  # 2 tests: .25*2


# --------------------------------------------------------------------------- protocol additions


def test_mcnemar_exact_known_values():
    from stats.core import mcnemar_exact

    # n10 = 5 (a right, b wrong), n01 = 1: p = 2 * (C(6,0) + C(6,1)) / 2**6 = 14/64
    a = [1, 1, 1, 1, 1, 0, 1, 0]
    b = [0, 0, 0, 0, 0, 1, 1, 0]
    r = mcnemar_exact(a, b)
    assert (r["n10"], r["n01"]) == (5, 1) and r["p"] == pytest.approx(14 / 64)
    assert mcnemar_exact([1, 0], [1, 0]) == {"n10": 0, "n01": 0, "p": 1.0}
    assert mcnemar_exact([1, 0], [0, 1])["p"] == 1.0  # one discordant each way: 2 * 3/4 capped at 1
    with pytest.raises(ValueError):
        mcnemar_exact([0.5], [1])


def test_cluster_bootstrap_matches_item_mean_and_is_wider_when_clusters_correlate():
    from stats.core import bootstrap_ci, cluster_bootstrap_ci, estimate_clustered

    # 20 meetings x 2 questions; both questions of a meeting agree (perfect within-cluster correlation)
    meet = [m for m in range(20) for _ in range(2)]
    vals = [float(m % 4 == 0) for m in meet]
    est = estimate_clustered(vals, meet)
    assert est["value"] == pytest.approx(np.mean(vals)) and est["n"] == 40 and est["n_clusters"] == 20
    lo_c, hi_c = cluster_bootstrap_ci(vals, meet)
    lo_i, hi_i = bootstrap_ci(vals)
    assert hi_c - lo_c > hi_i - lo_i  # 20 effective units, not 40
    assert cluster_bootstrap_ci([1.0, 1.0, 1.0], ["a", "a", "b"]) == (1.0, 1.0)
    with pytest.raises(ValueError):
        cluster_bootstrap_ci([1.0], ["a", "b"])


def test_paired_cluster_bootstrap_diff():
    from stats.core import paired_cluster_bootstrap_diff

    meet = [m for m in range(10) for _ in range(2)]
    a = [1.0] * 20
    b = [float(m % 2) for m in meet]
    r = paired_cluster_bootstrap_diff(a, b, meet)
    assert r["diff"] == pytest.approx(0.5) and r["n"] == 20 and r["n_clusters"] == 10
    assert r["ci_low"] <= 0.5 <= r["ci_high"]
