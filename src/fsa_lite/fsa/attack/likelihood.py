"""Likelihood (Bayes) attacker: how probable are the observed replies under each candidate secret?

With a uniform prior over the candidates, the attacker's posterior is softmax(scores), where
scores[c] = log p_T(all replies | prompt built with secret c). scripts/likelihood_attack.py
does the plumbing (prompt IDs, reply token IDs, batching, posterior, metrics). For each
candidate, the prompt is run once and its keys/values are cached and reused for every reply.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


def score_candidates(
    model: Any,
    tokenizer: Any,
    build_prompt_ids: Callable[[str], list[int]],
    reply_token_ids: Sequence[Sequence[int]],
    candidates: Sequence[str],
    temperature: float,
) -> Any:  # torch.Tensor of shape [len(candidates)]
    """Log-likelihood of the replies under each candidate secret, by teacher forcing.

    For every candidate c in `candidates`:

        score[c] = sum over replies r, sum over positions t = 1..len(y_r) of
                   log p_T(y_r[t] | x_c, y_r[1..t-1])

    where x_c = build_prompt_ids(c), y_r = reply_token_ids[r], and

        p_T(. | context) = softmax(logits(context) / T),   T = `temperature`,

    i.e. exactly the distribution the replies were sampled from (generation used pure
    temperature sampling: top_k = 0, top_p = 1, no repetition penalty), so there is no
    truncation or renormalisation.

    Teacher forcing: one forward pass over the concatenation x_c + y_r gives all terms at once.
    The logits at 0-based position len(x_c) - 1 + (t - 1) of the concatenation predict y_r[t].
    Only reply tokens are scored, never prompt tokens.

    Args:
        model: causal LM (Hugging Face AutoModelForCausalLM, possibly with a LoRA adapter),
            already on its device and in eval mode. The caller runs this under
            torch.inference_mode().
        tokenizer: the model's tokenizer. Its padding_side is "left" (it was set for
            generation). Pad however suits you, with an attention mask, and make sure padding
            never enters a score.
        build_prompt_ids: maps a candidate secret (e.g. "07") to the prompt token IDs exactly
            as sent to model.generate at generation time: the chat template with the method's
            system prompt holding that secret, the user prompt, and the generation prompt
            (assistant turn opener), with no extra BOS (fsa.prompting.prompt_ids). It is cached,
            so it is cheap to call.
        reply_token_ids: one list of token IDs per reply, in the same tokenizer, as sampled.
            A reply that ended naturally includes its end-of-turn/EOS token; a reply that hit
            max_new_tokens has no EOS. No padding. An empty reply contributes 0.
        candidates: the candidate secrets to score in this call (a chunk of "00".."99").
        temperature: T > 0, the sampling temperature of the run.

    Returns:
        Float tensor of shape [len(candidates)], where score[i] belongs to candidates[i]. Units
        are natural log (nats), summed (not averaged) over reply tokens and replies. Any device;
        float32 or float64. Compute log_softmax in float32: bf16 logits lose too much precision
        over ~1000 summed tokens.

    Batching contract: the caller has already split the work so that one call fits a single
    forward pass under its max_batch_tokens budget, which by default is sized for Gemma-3-12B
    on a 46 GB GPU. A call holds len(candidates) * len(reply_token_ids) sequences, each padded
    to the longest len(x_c) + len(y_r) in the call. Doing it all in one forward pass is fine.
    Because scores add over replies, the caller may call this several times with disjoint
    subsets of the replies and sum the results. So the function must not normalise across
    candidates or replies.

    Implementation: candidates are scored one after another. Each runs its prompt once, then
    runs all replies in one right-padded batch on top of the cached prompt keys/values.
    Returns float64 on the CPU.
    """
    return _score_all(
        model, build_prompt_ids, reply_token_ids, candidates, temperature, use_cache=True
    )


def score_candidates_uncached(
    model: Any,
    tokenizer: Any,
    build_prompt_ids: Callable[[str], list[int]],
    reply_token_ids: Sequence[Sequence[int]],
    candidates: Sequence[str],
    temperature: float,
) -> Any:
    """score_candidates without reusing the prompt cache: each reply's forward pass re-runs
    prompt + reply from scratch. Same contract and same numbers, but slower. Use it to
    cross-check the cache on a new model (e.g. Gemma-3's sliding-window layers).
    """
    return _score_all(
        model, build_prompt_ids, reply_token_ids, candidates, temperature, use_cache=False
    )


def _score_all(
    model: Any,
    build_prompt_ids: Callable[[str], list[int]],
    reply_token_ids: Sequence[Sequence[int]],
    candidates: Sequence[str],
    temperature: float,
    *,
    use_cache: bool,
) -> Any:
    import torch

    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    replies = [list(r) for r in reply_token_ids if len(r) > 0]  # empty replies add 0
    scores = torch.zeros(len(candidates), dtype=torch.float64)
    if not replies:
        return scores
    with torch.no_grad():
        for i, candidate in enumerate(candidates):
            scores[i] = _candidate_score(
                model, build_prompt_ids(candidate), replies, temperature, use_cache=use_cache
            )
    return scores


def _candidate_score(
    model: Any,
    prompt_ids: Sequence[int],
    replies: Sequence[Sequence[int]],
    temperature: float,
    *,
    use_cache: bool = True,
) -> float:
    """Summed log p_T of all `replies` (non-empty) after one prompt.

    use_cache=False runs prompt + reply as one sequence per reply instead of reusing the
    prompt's keys/values. It gives the same number and is kept so the tests can check that.
    """
    import torch

    device = model.device
    n, P = len(replies), len(prompt_ids)
    L = max(len(r) for r in replies)
    # Right padding: a real token never attends to a later pad (attention is causal), so pads
    # change nothing, and positions stay P, P+1, ... for every reply, matching generation.
    targets = torch.zeros((n, L), dtype=torch.long, device=device)
    mask = torch.zeros((n, L), dtype=torch.bool, device=device)
    for j, r in enumerate(replies):
        targets[j, : len(r)] = torch.tensor(r, device=device)
        mask[j, : len(r)] = True
    prompt = torch.tensor([list(prompt_ids)], device=device)

    # The t -> t+1 shift: the logits at position j predict token j+1. So the last prompt
    # position predicts reply token 0, and reply token t (as input) predicts reply token t+1.
    # The last reply token is never fed in, because nothing after it is scored.
    if use_cache:
        out = model(input_ids=prompt, use_cache=True, logits_to_keep=1)
        first = out.logits[:, -1:].expand(n, 1, -1)  # [n, 1, V]: predicts reply token 0
        rest = None
        if L > 1:
            cache = out.past_key_values
            cache.batch_repeat_interleave(n)  # one copy of the prompt's keys/values per reply
            attention = torch.cat(
                [torch.ones((n, P), dtype=torch.long, device=device), mask[:, :-1].long()], dim=1
            )
            rest = model(
                input_ids=targets[:, :-1],
                attention_mask=attention,
                past_key_values=cache,
                use_cache=True,
            ).logits  # [n, L-1, V]: predicts reply tokens 1..L-1
        logits = first if rest is None else torch.cat([first, rest], dim=1)  # [n, L, V]
    else:
        inputs = torch.cat([prompt.expand(n, P), targets[:, :-1]], dim=1)  # prompt + y[:-1]
        attention = torch.cat(
            [torch.ones((n, P), dtype=torch.long, device=device), mask[:, :-1].long()], dim=1
        )
        logits = model(input_ids=inputs, attention_mask=attention).logits[:, P - 1 :]

    # Temperature divides the logits before the softmax, exactly as sampling did. log_softmax
    # runs in float32 because bf16 logits are too coarse to sum over ~1000 tokens.
    logp = torch.log_softmax(logits.float() / temperature, dim=-1)
    picked = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [n, L]
    return picked.masked_fill(~mask, 0.0).double().sum().item()  # padding contributes 0
