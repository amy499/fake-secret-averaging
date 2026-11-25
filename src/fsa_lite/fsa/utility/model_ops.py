"""Model side of the utility benchmarks (torch imported lazily, inside the functions).

HFRunner wraps a loaded (model, tokenizer), as HFGenerator._load in the dashboard backend returns:
next-token log-probs of a few answer tokens after a prompt (MMLU, verification), batched greedy
generation with left padding (IFEval, general prompts), and per-token KL(adapted || base) on a
reply, where the base distribution comes from the same model under peft's disable_adapter().
Tests use the same class on a tiny random model, or a fake runner with the same three methods.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import Any, Protocol

from fsa.prompting import eos_ids, until_eos


class Runner(Protocol):
    def choice_logprobs(self, prompt_ids: list[int], token_ids: Sequence[int]) -> list[float]: ...

    def generate(self, prompts: Sequence[list[int]]) -> list[list[int]]: ...

    def kl(self, prompt_ids: list[int], reply_ids: list[int]) -> tuple[float, int]: ...


def answer_token_ids(tokenizer: Any, prompt_text: str, words: Sequence[str]) -> dict[str, int]:
    """Each answer's token id as the first token of the assistant turn: the chat prompt text with
    the answer appended must tokenize to the prompt's ids plus exactly one token. Raises
    otherwise, since a multi-token answer cannot be scored by one next-token log-prob."""
    base = list(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    out = {}
    for w in words:
        ids = list(tokenizer(prompt_text + w, add_special_tokens=False)["input_ids"])
        if ids[: len(base)] != base or len(ids) != len(base) + 1:
            raise ValueError(
                f"answer {w!r} is not a single token after the assistant opener: "
                f"{ids[len(base) - 1 :] if len(ids) >= len(base) else ids}"
            )
        out[w] = int(ids[-1])
    if len(set(out.values())) != len(out):
        raise ValueError(f"answer tokens are not distinct: {out}")
    return out


class HFRunner:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        batch_size: int = 16,
        max_new_tokens: int = 1280,
        kl_chunk: int = 256,
    ) -> None:
        self.model, self.tokenizer = model, tokenizer
        self.batch_size, self.max_new_tokens, self.kl_chunk = batch_size, max_new_tokens, kl_chunk
        self.eos = eos_ids(tokenizer, model)
        self.has_adapter = hasattr(model, "disable_adapter")

    @property
    def device(self) -> Any:
        return next(self.model.parameters()).device

    def choice_logprobs(self, prompt_ids: list[int], token_ids: Sequence[int]) -> list[float]:
        import torch

        with torch.no_grad():
            ids = torch.tensor([prompt_ids], device=self.device)
            logits = self.model(input_ids=ids, logits_to_keep=1).logits[0, -1].float()
            logp = torch.log_softmax(logits, dim=-1)
        return [float(logp[t]) for t in token_ids]

    def generate(self, prompts: Sequence[list[int]]) -> list[list[int]]:
        """Greedy replies (up to and including the first EOS), batched with left padding."""
        import torch

        pad = self.tokenizer.pad_token_id
        pad = pad if pad is not None else next(iter(self.eos))
        out: list[list[int]] = []
        for start in range(0, len(prompts), self.batch_size):
            batch = prompts[start : start + self.batch_size]
            width = max(len(p) for p in batch)
            ids = torch.tensor([[pad] * (width - len(p)) + p for p in batch], device=self.device)
            mask = torch.tensor(
                [[0] * (width - len(p)) + [1] * len(p) for p in batch], device=self.device
            )
            with torch.no_grad():
                gen = self.model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=pad,
                    eos_token_id=sorted(self.eos),
                )
            for row in gen[:, width:].tolist():
                out.append(until_eos(row, self.eos))
        return out

    def kl(self, prompt_ids: list[int], reply_ids: list[int]) -> tuple[float, int]:
        """(sum over reply tokens of KL(adapted || base) in nats, number of reply tokens)."""
        import torch

        if not reply_ids:
            return 0.0, 0
        if not self.has_adapter:
            raise ValueError("KL to base needs a peft model (base runs record 0 by definition)")
        ids = torch.tensor([prompt_ids + reply_ids], device=self.device)
        keep = len(reply_ids) + 1  # logits predicting each reply token (the last one is dropped)

        def logits(adapter: bool) -> Any:
            ctx = contextlib.nullcontext() if adapter else self.model.disable_adapter()
            with ctx, torch.no_grad():
                return self.model(input_ids=ids, logits_to_keep=keep).logits[0, :-1]

        a, b = logits(True), logits(False)
        total = 0.0
        for s in range(0, a.shape[0], self.kl_chunk):
            la = torch.log_softmax(a[s : s + self.kl_chunk].float(), dim=-1)
            lb = torch.log_softmax(b[s : s + self.kl_chunk].float(), dim=-1)
            total += float((la.exp() * (la - lb)).sum(-1).clamp_min(0).sum())
        return total, len(reply_ids)
