"""Activation-patching metric: scores how much one patched site carries the secret's influence.

TODO(user): implement. Signature and I/O contract only; body intentionally left empty.

The dashboard heatmap (metrics.json -> patching.scores) shows one aggregated value of this
metric per (layer, component) or per (layer, position group) cell.

Shape legend: B = batch, V = vocab size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def patching_metric(
    patched_logits: torch.Tensor,
    clean_logits: torch.Tensor,
    corrupted_logits: torch.Tensor,
    secret_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Score one patching intervention.

    "clean" = run with the real secret in the system prompt; "corrupted" = run with a
    different (fake) secret; "patched" = corrupted run with one site's activation replaced
    by its clean-run value (or vice versa, depending on the patching direction).

    Args:
        patched_logits: Logits at the scored position (e.g. last token), shape [B, V].
        clean_logits: Same position from the clean run, shape [B, V].
        corrupted_logits: Same position from the corrupted run, shape [B, V].
        secret_token_ids: Token ids of the real secret's digits, shape [B, n_digits], long.

    Returns:
        Per-example scores, shape [B], float32. The caller averages over examples to fill
        one heatmap cell.
    """
    raise NotImplementedError
