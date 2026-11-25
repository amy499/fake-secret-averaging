"""Training loss pulling the LoRA model (holding the real secret) toward a target distribution.

Per reply token: KL(p_student || m) = sum_v p_student(v) (log p_student(v) - log m(v)), in
float32, with the target detached, so gradients reach only the student. Averaged over reply
tokens by default. The same function is used for FSA (target = mixture over fake secrets) and
for the B3 baseline (target = frozen model with no secret).

Shape legend: B = batch, T = sequence length, V = vocab size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch


def fsa_loss(
    student_logits: torch.Tensor,
    target_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """Loss between the LoRA model's predictions and the target distribution.

    Args:
        student_logits: Raw logits of the LoRA model run on the real-secret sequence,
            shape [B, T, V]; carries grad.
        target_logprobs: Target next-token log-probabilities from
            `fsa.targets.mixture.mixture_target` (or the B3 no-secret model),
            shape [B, T, V]; no grad.
        response_mask: Shape [B, T], bool. Only True positions contribute.
        reduction: "mean" -> scalar averaged over masked positions; "sum" -> scalar sum
            over masked positions; "none" -> per-position values, shape [B, T], zero where
            `response_mask` is False.

    Returns:
        A scalar tensor (for "mean"/"sum") or a [B, T] tensor (for "none").
    """
    import torch

    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"unknown reduction {reduction!r}")
    if student_logits.shape != target_logprobs.shape:
        raise ValueError("student_logits and target_logprobs must have the same [B, T, V] shape")
    mask = response_mask.bool()
    # Select the reply positions first. The target is unspecified elsewhere (it may hold
    # NaN), and NaN would leak into the gradients even when multiplied by a zero mask.
    log_p = torch.log_softmax(student_logits[mask].float(), dim=-1)  # [N, V]
    log_m = target_logprobs[mask].detach().float()  # a fixed target: no gradient into it
    # KL direction: the expectation is under the student, KL(p_student || m). Only tokens the
    # student itself would produce are weighted, so it is penalised for putting probability
    # where the fake-secret average puts little (secret-specific behaviour). The B3 baseline
    # uses the same direction.
    kl = (log_p.exp() * (log_p - log_m)).sum(dim=-1)  # [N]
    if reduction == "sum":
        return kl.sum()
    if reduction == "mean":
        return kl.sum() / max(kl.numel(), 1)  # no reply tokens -> 0, not NaN
    out = torch.zeros(mask.shape, dtype=kl.dtype, device=kl.device)
    return out.masked_scatter(mask, kl)  # differentiable; zeros where the mask is False
