"""Statistics: bootstrap CIs, permutation tests, Holm correction, paired run comparisons.

Status: ✅ implemented + tested against known values (tests/test_stats.py).

Every reported number is an `Estimate`: value, 95% CI, n, chance level and (where relevant) the
no-secret control value.
"""

from stats.core import (
    Estimate,
    bootstrap_ci,
    cluster_bootstrap_ci,
    estimate,
    estimate_clustered,
    holm,
    mcnemar_exact,
    paired_bootstrap_diff,
    paired_cluster_bootstrap_diff,
    paired_permutation_test,
    permutation_test,
)

__all__ = [
    "Estimate",
    "bootstrap_ci",
    "cluster_bootstrap_ci",
    "estimate",
    "estimate_clustered",
    "holm",
    "mcnemar_exact",
    "paired_bootstrap_diff",
    "paired_cluster_bootstrap_diff",
    "paired_permutation_test",
    "permutation_test",
]
