"""Generic step-wise decoding over a TokenLevel model.

Status: 🟡 implemented, no unit test yet (used only by Tier 2 decoding-time defences, not started).

Decoding-time defences (token_ban, dp_fusion, adler, attention_block) build a `step` function
that returns the next-token log-probabilities, possibly mixing several contexts; this module
does temperature / top-p sampling with a seeded numpy Generator, so the same seed gives the
same tokens on any device.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from core.interfaces import GenConfig

StepFn = Callable[[list[int]], np.ndarray]  # generated-so-far -> log-probs over vocab


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    return x - np.log(np.sum(np.exp(x)))


def choose(logp: np.ndarray, gen_cfg: GenConfig, rng: np.random.Generator) -> int:
    if gen_cfg.greedy:
        return int(np.argmax(logp))
    z = logp / max(gen_cfg.temperature, 1e-6)
    p = np.exp(z - np.max(z))
    p /= p.sum()
    if gen_cfg.top_p < 1.0:
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        keep = order[: int(np.searchsorted(cum, gen_cfg.top_p) + 1)]
        mask = np.zeros_like(p)
        mask[keep] = p[keep]
        p = mask / mask.sum()
    return int(rng.choice(p.size, p=p))


def decode_loop(
    step: StepFn,
    eos_id: int,
    gen_cfg: GenConfig,
    seed: int | None,
    stop_ids: Sequence[int] = (),
) -> list[int]:
    rng = np.random.default_rng(seed if seed is not None else 0)
    out: list[int] = []
    stops = {eos_id, *stop_ids}
    for _ in range(gen_cfg.max_new_tokens):
        tok = choose(step(out), gen_cfg, rng)
        if tok in stops:
            break
        out.append(tok)
    return out
