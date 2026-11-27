"""Paired leakage statistics: the original model vs an adapter on the same secrets (report only).

Input: two likelihood-attack results (fsa.contract.schemas.LikelihoodAttack) on the same held-out
secrets, one per variant. For every secret i and each variant, from the saved candidate scores:
    log2 p_i   log2 posterior of the true secret (raw: softmax(scores); calibrated:
               softmax(scores / tau) with the file's own tau), clipped at 1e-12 like
               fsa.eval.leakage.attacker_mean_ce_bits
    exact_i    tie-aware top-1 credit (fsa.eval.leakage.tie_aware_score)
    top5_i     tie-aware top-5 credit
Statistics over the n secrets: bits = max(0, log2 N + mean log2 p) (the lower bound every result
file reports), exact = mean exact_i, top-5 = mean top5_i. 95% CIs: percentile bootstrap over
secrets (default 2000 draws, seed 0). The paired difference (adapter - original) uses the SAME
resampled secrets for both variants in every draw, so it is a paired bootstrap; its point
estimate is the difference of the two statistics.

Only aggregates leave this module: no secret, code, score or reply text, and no per-secret row
(a per-secret row would be keyed by its code).

Calibration: a variant whose tau sits at the search bound (0.01 or 10000) has a degenerate
calibrated posterior; its calibrated bits are reported with flag `degenerate` (the display rule
of docs/fsa/likelihood_attack.md: read the raw row instead). exact / top-5 do not depend on tau
except through ties; they come from the calibrated posterior when it is usable, else the raw one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from fsa.attack.calibration import tempered_posteriors
from fsa.eval.leakage import tie_aware_score

EPS = 1e-12
METRICS = ("bits_calibrated", "bits_raw", "exact", "top5")


class PairingError(ValueError):
    """The two results cannot be paired (different secrets, prompts, decoding, ...)."""


@dataclass(frozen=True)
class PerSecret:
    """Per-secret values in the ORIGINAL variant's secret order (never written out)."""

    log2p_raw: np.ndarray
    log2p_cal: np.ndarray | None
    exact: np.ndarray
    top5: np.ndarray
    tau: float | None
    tau_at_bound: bool


@dataclass(frozen=True)
class Estimate:
    value: float
    low: float
    high: float


@dataclass
class PairedResult:
    n_secrets: int
    secret_space: int
    original: dict[str, Estimate | None]
    adapter: dict[str, Estimate | None]
    difference: dict[str, Estimate | None]
    tau: dict[str, float | None]
    degenerate: dict[str, bool]
    n_resamples: int
    seed: int
    notes: list[str] = field(default_factory=list)


def per_secret(attack, order: Sequence[str]) -> PerSecret:
    """Per-secret values of one LikelihoodAttack, rows reordered to `order`."""
    secrets = list(attack.secrets)
    if sorted(secrets) != sorted(order) or len(set(secrets)) != len(secrets):
        raise PairingError("the two results hold different held-out secrets")
    pos = {s: i for i, s in enumerate(secrets)}
    rows = [pos[s] for s in order]
    scores = np.asarray(attack.scores, dtype=np.float64)[rows]
    n_cand = scores.shape[1]
    candidates = [f"{i:0{len(order[0])}d}" for i in range(n_cand)] if order[0].isdigit() else None
    if candidates is None or any(s not in candidates for s in order):
        raise PairingError("paired statistics need digit secrets (candidates 00..99)")
    true_idx = [candidates.index(s) for s in order]
    cal = attack.calibration
    tau = float(cal.tau) if cal is not None else None
    at_bound = bool(cal.tau_at_search_bound) if cal is not None else False

    def log2p(post: np.ndarray) -> np.ndarray:
        return np.log2(np.maximum(post[np.arange(len(post)), true_idx], EPS))

    raw_post = tempered_posteriors(scores)
    cal_post = tempered_posteriors(scores, tau) if tau is not None else None
    use = cal_post if cal_post is not None and not at_bound else raw_post
    ts = [tie_aware_score(use[i], candidates, s) for i, s in enumerate(order)]
    return PerSecret(
        log2p_raw=log2p(raw_post),
        log2p_cal=log2p(cal_post) if cal_post is not None else None,
        exact=np.array([t.exact for t in ts]),
        top5=np.array([t.topk for t in ts]),
        tau=tau,
        tau_at_bound=at_bound,
    )


def _stat(metric: str, ps: PerSecret, idx: np.ndarray, cap: float) -> np.ndarray | None:
    """The metric for each row of `idx` (shape [draws, n]) -> [draws]."""
    if metric == "bits_raw":
        return np.maximum(0.0, cap + ps.log2p_raw[idx].mean(axis=-1))
    if metric == "bits_calibrated":
        if ps.log2p_cal is None:
            return None
        return np.maximum(0.0, cap + ps.log2p_cal[idx].mean(axis=-1))
    if metric == "exact":
        return ps.exact[idx].mean(axis=-1)
    if metric == "top5":
        return ps.top5[idx].mean(axis=-1)
    raise KeyError(metric)


def _est(point: np.ndarray | None, boot: np.ndarray | None) -> Estimate | None:
    if point is None or boot is None:
        return None
    low, high = np.percentile(boot, [2.5, 97.5])
    return Estimate(float(point[0]), float(low), float(high))


def paired(original, adapter, *, n_resamples: int = 2000, seed: int = 0) -> PairedResult:
    """Paired statistics of two LikelihoodAttack results on the same held-out secrets."""
    order = list(original.secrets)
    a, b = per_secret(original, order), per_secret(adapter, order)
    n = len(order)
    space = len(original.scores[0])
    if len(adapter.scores[0]) != space:
        raise PairingError("the two results score different numbers of candidates")
    cap = math.log2(space)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_resamples, n))
    full = np.arange(n)[None, :]
    out = {"original": {}, "adapter": {}, "difference": {}}
    for m in METRICS:
        pa, pb = _stat(m, a, full, cap), _stat(m, b, full, cap)
        ba, bb = _stat(m, a, draws, cap), _stat(m, b, draws, cap)
        out["original"][m] = _est(pa, ba)
        out["adapter"][m] = _est(pb, bb)
        out["difference"][m] = (
            _est(pb - pa, bb - ba) if pa is not None and pb is not None else None
        )
    notes = []
    for label, ps in (("original", a), ("adapter", b)):
        if ps.tau_at_bound:
            notes.append(f"{label}: tau {ps.tau:.3g} at the search bound: calibrated bits degenerate")
        if ps.tau is None:
            notes.append(f"{label}: no calibration in the file (run likelihood_attack --calibrate)")
    return PairedResult(
        n_secrets=n,
        secret_space=space,
        original=out["original"],
        adapter=out["adapter"],
        difference=out["difference"],
        tau={"original": a.tau, "adapter": b.tau},
        degenerate={"original": a.tau_at_bound, "adapter": b.tau_at_bound},
        n_resamples=n_resamples,
        seed=seed,
        notes=notes,
    )


# --------------------------------------------------------------------------- run pairing checks

# config.yaml fields that must agree between the two runs of a pair (dotted paths).
PAIRED_FIELDS = (
    "seed",
    "model.id",
    "model.base",
    "model.revision",
    "eval.prompts",
    "eval.prompt_set",
    "eval.n_secrets",
    "eval.n_train_secrets",
    "eval.n_replies_per_secret",
    "eval.temperature",
    "generation.backend",
    "generation.dtype",
    "generation.max_new_tokens",
    "generation.batch_size",
    "generation.top_k",
    "generation.top_p",
    "generation.repetition_penalty",
    "generation.loaded.revision_loaded",
    "generation.loaded.dtype_loaded",
)


def _get(cfg: dict, dotted: str):
    cur = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _strip_adapter_view(prompt_set):
    """eval.prompt_set without the adapter-only key (adapter_trained_prompts)."""
    if not isinstance(prompt_set, dict):
        return prompt_set
    return {k: v for k, v in prompt_set.items() if k != "adapter_trained_prompts"}


def check_run_pair(original_cfg: dict, adapter_cfg: dict) -> list[str]:
    """Reasons the two runs are not a valid pair (empty list = paired). Never names a secret."""
    problems = []
    if (original_cfg.get("generation") or {}).get("adapter"):
        problems.append("the original run has an adapter")
    if not (adapter_cfg.get("generation") or {}).get("adapter"):
        problems.append("the adapter run has no adapter")
    for f in PAIRED_FIELDS:
        a, b = _get(original_cfg, f), _get(adapter_cfg, f)
        if f == "eval.prompt_set":
            a, b = _strip_adapter_view(a), _strip_adapter_view(b)
        if a != b:
            problems.append(f"{f} differs")
    sa = _get(original_cfg, "eval.secrets") or {}
    sb = _get(adapter_cfg, "eval.secrets") or {}
    for split in ("heldout", "train"):
        if list(sa.get(split) or []) != list(sb.get(split) or []):
            problems.append(f"the {split} secrets differ")
    return problems
