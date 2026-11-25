"""Split the candidates x replies scoring grid into blocks that each fit one forward pass.

A block is a (candidate chunk, reply group) pair. It costs len(chunk) * len(group) sequences,
each padded to the longest prompt + reply in the block. Replies are grouped shortest first so
padding stays small; since scores add over replies, the per-group results are summed.
suggest_max_batch_tokens turns GPU memory and model size into a token budget. The main
per-token cost is the logits row, which is vocab_size floats (Gemma-3 has 262k).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

GIB = 1024**3


@dataclass(frozen=True)
class Block:
    candidates: tuple[int, ...]  # indices into the candidate list
    replies: tuple[int, ...]  # indices into the reply list
    padded_tokens: int  # len(candidates) * len(replies) * padded length


def plan_blocks(
    prompt_len: int, reply_lens: Sequence[int], n_candidates: int, max_batch_tokens: int
) -> list[Block]:
    """Cover every (candidate, reply) pair exactly once with blocks of <= max_batch_tokens.

    prompt_len is the longest prompt over the candidates. A single sequence longer than the
    budget still gets a 1 x 1 block, because it cannot be split.
    """
    if max_batch_tokens < 1 or n_candidates < 1:
        raise ValueError("max_batch_tokens and n_candidates must be >= 1")
    order = sorted(range(len(reply_lens)), key=lambda r: reply_lens[r])
    groups: list[list[int]] = []
    for r in order:  # grow a group while one candidate's share still fits the budget
        if groups:
            g = groups[-1]
            width = prompt_len + max(reply_lens[i] for i in [*g, r])
            if (len(g) + 1) * width <= max_batch_tokens:
                g.append(r)
                continue
        groups.append([r])
    blocks = []
    for g in groups:
        per_candidate = len(g) * (prompt_len + max(reply_lens[i] for i in g))
        chunk = max(1, max_batch_tokens // per_candidate)
        for start in range(0, n_candidates, chunk):
            cands = tuple(range(start, min(start + chunk, n_candidates)))
            blocks.append(Block(cands, tuple(g), len(cands) * per_candidate))
    return blocks


def suggest_max_batch_tokens(
    *,
    gpu_memory_bytes: int,
    weight_bytes: int,
    vocab_size: int,
    hidden_size: int,
    reserve_bytes: int = 2 * GIB,
    usable_fraction: float = 0.9,
) -> int:
    """Token budget per forward pass, rounded down to a multiple of 256 (at least 256).

    Per token: bf16 logits plus an fp32 copy and an fp32 log_softmax (10 bytes x vocab), plus
    transient activations (~40 bf16 vectors of hidden_size). No KV cache is kept (single pass).
    Gemma-3-12B on a 46 GB card: (0.9 x 42.8 - 22.7 - 2) GiB / 2.9 MB ~ 4800 tokens.
    """
    free = gpu_memory_bytes * usable_fraction - weight_bytes - reserve_bytes
    per_token = 10 * vocab_size + 80 * hidden_size
    return max(256, int(free // per_token) // 256 * 256)
