"""Mixture target: the frozen model's averaged behaviour over k fake secrets.

sample_fake_secrets draws the k fakes from the training secrets. mixture_target runs the frozen
model once per fake and averages the k next-token distributions in probability space,
m = (1/k) sum_i p_i, returned as log m. torch is imported inside the functions, so this module
imports without the ml group.

Shape legend: k = number of fake secrets, B = batch, T = sequence length, V = vocab size.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel

NORMALISATION_ATOL = 1e-3  # |logsumexp_v log m - 0| allowed in float32 over a ~262k vocab


def sample_fake_secrets(
    true_secret: str,
    k: int,
    *,
    pool: Sequence[str],
    n_digits: int = 2,
    exclude_true: bool = True,
    generator: torch.Generator | None = None,
) -> list[str]:
    """Draw the k fake secrets used to build the mixture target for one real secret.

    Args:
        true_secret: The real secret held in the system prompt, e.g. "37".
        k: Number of fake secrets to return (k >= 1).
        pool: The training secrets to draw from, and only these. Held-out evaluation secrets
            must not be in it, so a fake can never be a held-out secret.
        n_digits: Length of each digit secret string; secrets are zero-padded ("07"). Word
            secrets (lowercase letters, fsa.secrets word lists) ignore it.
        exclude_true: If True, `true_secret` never appears in the output.
        generator: Optional torch RNG for reproducibility.

    Returns:
        List of length k of distinct secret strings, each `n_digits` characters long.
    """
    import torch

    if k < 1:
        raise ValueError("k must be >= 1")
    if true_secret.isdigit():
        bad = [s for s in pool if len(s) != n_digits or not s.isdigit()]
        if bad:
            raise ValueError(f"pool entries must be {n_digits}-digit strings, got {bad[:3]}")
    else:  # a word secret (fsa.secrets word list): fakes are other lowercase words
        from fsa.secrets import WORD_PATTERN

        bad = [s for s in pool if not WORD_PATTERN.fullmatch(s)]
        if bad:
            raise ValueError(f"pool entries must be lowercase words, got {bad[:3]}")
    # Sorted and de-duplicated, so the draw depends only on the pool's contents and the seed,
    # not on the order the caller listed the pool in.
    candidates = sorted(set(pool) - ({true_secret} if exclude_true else set()))
    if k > len(candidates):
        raise ValueError(f"need k = {k} distinct fakes but the pool offers {len(candidates)}")
    # Any digit secret is an equally valid thing to hold: the task ("generate random numbers")
    # never depends on the secret, so every fake gives the same permitted answer. Averaging
    # over fakes therefore keeps the task behaviour and washes out anything secret-specific.
    order = torch.randperm(len(candidates), generator=generator)[:k]
    return [candidates[i] for i in order.tolist()]


def mix_logprobs(fake_logprobs: torch.Tensor) -> torch.Tensor:
    """[k, B, T, V] log p_i  ->  [B, T, V] log m with m = (1/k) sum_i p_i, in float32.

    Averaging probabilities in log space is log(sum_i exp(log p_i)) - log k. logsumexp does that
    without leaving log space: exp() of a log-prob around -100 underflows to 0 in float32,
    while logsumexp subtracts the maximum first and stays exact.
    """
    import torch

    k = fake_logprobs.shape[0]
    return torch.logsumexp(fake_logprobs.float(), dim=0) - math.log(k)


def mixture_target(
    frozen_model: PreTrainedModel,
    fake_input_ids: torch.Tensor,
    fake_attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the training target distribution from the frozen model under k fake secrets.

    Each of the k fake-secret sequences is the real training sequence with only the secret
    span replaced, so all k sequences are position-aligned with the real one.

    Args:
        frozen_model: The base model with no LoRA adapter active; called under no_grad.
        fake_input_ids: Token ids, shape [k, B, T], dtype long.
        fake_attention_mask: Shape [k, B, T], 1 for real tokens, 0 for padding.
        response_mask: Shape [B, T], bool. True at positions whose next-token
            distribution is part of the target (the assistant reply tokens).

    Returns:
        Target next-token log-probabilities, shape [B, T, V], float32, no grad.
        Values at positions where `response_mask` is False are unspecified.
    """
    import torch

    if fake_input_ids.dim() != 3 or fake_input_ids.shape != fake_attention_mask.shape:
        raise ValueError("fake_input_ids and fake_attention_mask must both be [k, B, T]")
    k = fake_input_ids.shape[0]
    acc = None
    with torch.no_grad():
        for i in range(k):
            logits = frozen_model(
                input_ids=fake_input_ids[i], attention_mask=fake_attention_mask[i]
            ).logits
            logp = torch.log_softmax(logits.float(), dim=-1)  # [B, T, V]
            # Running logsumexp over k: logaddexp(a, b) = log(e^a + e^b). This equals
            # mix_logprobs(stack(...)) but holds one [B, T, V] tensor instead of k of them
            # (k = 8 stacked Gemma-vocab tensors would be ~20 GB at B = 8, T = 300).
            acc = logp if acc is None else torch.logaddexp(acc, logp)
    target = acc - math.log(k)
    _assert_normalised(target, response_mask)
    return target


def _assert_normalised(logprobs: torch.Tensor, mask: torch.Tensor) -> None:
    """log m must be a distribution at every target position: logsumexp over V == 0."""
    import torch

    lse = torch.logsumexp(logprobs[mask.bool()], dim=-1)
    worst = lse.abs().max().item() if lse.numel() else 0.0
    if not worst <= NORMALISATION_ATOL:  # also catches NaN
        raise AssertionError(f"mixture target not normalised: max |logsumexp| = {worst:.2e}")
