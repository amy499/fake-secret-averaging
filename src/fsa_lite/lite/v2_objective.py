"""FSA v2 objective: the v1 losses on contexts whose prompts differ in length.

Status: 🟡 implemented; tested on CPU with a tiny random model (src/fsa_lite/lite/selftest_v2.py).

Reused unchanged from src/fsa: fsa.training.loss.fsa_loss (per reply token
KL(p_student || target), float32, target detached, mean over the selected tokens) and the
mixture rule of fsa.targets.mixture (m = (1/K) sum_i p_i in log space with a running logaddexp,
normalisation asserted by its _assert_normalised).

What is new is only the alignment. In v1 every fake differed from the real prompt by a digit
span of equal token length, so all sequences were position-aligned. API keys and connection
strings tokenize to different lengths, so here each context (real, and each fake) is run on its
own [prompt + reply] sequence, right-padded, and the next-token log-probs at the positions that
predict the reply tokens (len(prompt)-1 ... len(prompt)+len(reply)-2) are taken out. The reply
tokens are the same in every context, so these [n_reply, V] rows line up one-to-one: the same
move as v1's B3 target (trainer.Trainer.target, no-secret prompt of another length).

Loss tokens: every token of the frozen model's sampled reply, up to and including its end-of-turn
token (fsa.prompting.until_eos), exactly the v1 reply tokens; never system, user or earlier turns.
  fsa  tokens: KL(student | real  ||  mixture of frozen | fake_1..K)
  need tokens: KL(student | real  ||  frozen | real)                   (weighted by lambda)
run3 (FSA-lite, config use.weight > 0; added in v2_train.example_step, example_loss is unchanged):
  use  tokens: weighted NLL of the frozen model's own correct-release reply under the student,
               sum_j w_j (-log p_student(y_j)) / sum_j w_j (payload tokens weighted up)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def _pad_right(rows: Sequence[Sequence[int]], pad_id: int, device: Any):
    import torch

    T = max(len(r) for r in rows)
    ids = torch.full((len(rows), T), pad_id, dtype=torch.long)
    attn = torch.zeros((len(rows), T), dtype=torch.long)
    for b, r in enumerate(rows):
        ids[b, : len(r)] = torch.tensor(list(r))
        attn[b, : len(r)] = 1
    return ids.to(device), attn.to(device)


def reply_positions(prompt_len: int, reply_len: int) -> range:
    """Positions whose next token is a reply token (the t -> t+1 shift)."""
    return range(prompt_len - 1, prompt_len + reply_len - 1)


def reply_logits(model: Any, prompts: Sequence[Sequence[int]], reply: Sequence[int], pad_id: int, device: Any):
    """[len(prompts), n_reply, V] raw logits at the reply positions of each context (one forward)."""
    import torch

    ids, attn = _pad_right([list(p) + list(reply) for p in prompts], pad_id, device)
    logits = model(input_ids=ids, attention_mask=attn).logits
    rows = [logits[b, list(reply_positions(len(p), len(reply)))] for b, p in enumerate(prompts)]
    return torch.stack(rows)


def mixture_reply_target(frozen: Any, fake_prompts: Sequence[Sequence[int]], reply: Sequence[int],
                         pad_id: int, device: Any, fake_batch: int = 4):
    """[n_reply, V] float32 log m, m = mean over the K fake contexts of the frozen model's next-token
    distribution at the reply positions. Running logaddexp over chunks, as mixture_target."""
    import torch

    from fsa.targets.mixture import _assert_normalised

    acc, k = None, len(fake_prompts)
    with torch.no_grad():
        for s in range(0, k, fake_batch):
            chunk = fake_prompts[s : s + fake_batch]
            logp = torch.log_softmax(reply_logits(frozen, chunk, reply, pad_id, device).float(), dim=-1)
            part = torch.logsumexp(logp, dim=0)  # sum over this chunk's fakes, in log space
            acc = part if acc is None else torch.logaddexp(acc, part)
    target = acc - math.log(k)
    _assert_normalised(target, torch.ones(target.shape[0], dtype=torch.bool, device=target.device))
    return target


def frozen_reply_target(frozen: Any, prompt: Sequence[int], reply: Sequence[int], pad_id: int, device: Any):
    """[n_reply, V] float32 log p_frozen(. | real context): the need-term target."""
    import torch

    with torch.no_grad():
        return torch.log_softmax(reply_logits(frozen, [prompt], reply, pad_id, device)[0].float(), dim=-1)


def use_nll(student_logits, reply: Sequence[int], weights: Sequence[float] | None = None):
    """Weighted mean NLL of the reply tokens under the student: student_logits [n_reply, V] at the
    reply positions (reply_logits), float32; weights None = the plain mean."""
    import torch

    logp = torch.log_softmax(student_logits.float(), dim=-1)
    y = torch.tensor(list(reply), dtype=torch.long, device=student_logits.device)
    nll = -logp.gather(-1, y[:, None])[:, 0]
    if weights is None:
        return nll.mean()
    w = torch.tensor(list(weights), dtype=torch.float32, device=student_logits.device)
    return (nll * w).sum() / w.sum().clamp_min(1e-12)


def example_loss(student_logits, fsa_target, need_target, fsa_mask: Sequence[bool], need_mask: Sequence[bool],
                 need_weight: float, fsa_weights: Sequence[float] | None = None):
    """(total, fsa term or None, need term or None) for one example. student_logits: [n_reply, V]
    with grad; targets [n_reply, V] (None when the example has no tokens of that kind).
    fsa_weights (optional, one per reply token): the FSA term becomes the weighted mean
    sum_j w_j KL_j / sum_j w_j over the FSA tokens (e.g. secret tokens weighted up, so a few code
    tokens are not diluted by the rest of the reply); None = the plain mean, as before."""
    import torch

    from fsa.training.loss import fsa_loss

    total, f_term, n_term = 0.0, None, None
    fm = torch.tensor(list(fsa_mask), dtype=torch.bool, device=student_logits.device)
    nm = torch.tensor(list(need_mask), dtype=torch.bool, device=student_logits.device)
    if bool(fm.any()):
        if fsa_weights is None:
            f_term = fsa_loss(student_logits[None], fsa_target[None], fm[None])
        else:
            w = torch.tensor(list(fsa_weights), dtype=torch.float32, device=student_logits.device) * fm.float()
            kl = fsa_loss(student_logits[None], fsa_target[None], fm[None], reduction="none")[0]
            f_term = (kl * w).sum() / w.sum().clamp_min(1e-12)
        total = total + f_term
    if bool(nm.any()):
        n_term = fsa_loss(student_logits[None], need_target[None], nm[None])
        total = total + need_weight * n_term
    return total, f_term, n_term
