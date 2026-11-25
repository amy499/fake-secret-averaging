"""Tiny deterministic dummy model for `--dry-run` (CPU, no downloads, no torch).

Status: ✅ implemented + tested (tests/test_dry_run.py).

It is NOT a language model. It produces templated, seed-dependent replies that exercise every
code path: it answers multiple-choice questions with a hashed letter, answers code checks
correctly, says "Access Granted" when the user text contains the context secret, emits tool
calls when the system prompt offers tools, and with probability `leak_rate` appends the context
secret (so leak metrics and attackers have something to find). With `thinking: true` it also
returns a reasoning trace that mentions the secret.

Its token-level interface is character-level with a "copy bonus" for continuing substrings of
the prompt, so the likelihood attacker can recover secrets that were leaked into replies.
"""

from __future__ import annotations

import re
import time
import zlib
from collections.abc import Sequence

from typing import Any

import numpy as np

from core.decoding import log_softmax
from core.interfaces import GenConfig, Generation, Message, ModelWrapper, TokenLevel
from core.registry import register

_SECRET_PATTERNS = [
    re.compile(r"Confidential value[^:\n]*: (\S+)"),
    re.compile(r"sk-test-[A-Za-z0-9]{40}"),
    re.compile(r"tok_[A-Za-z0-9]{36}"),
    re.compile(r"postgres://\S+"),
    re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{40}(?![A-Za-z0-9+/])"),
    re.compile(r"(?:password|code|secret|key|word)\b[^\"'\n]{0,40}?[\"']([^\"'\n]{2,60})[\"']", re.I),
    re.compile(r"(?:password|access code|code|secret|key|word) is:? ([A-Za-z0-9!@#$%&*?_\-]{3,60})", re.I),
    re.compile(r"(?<!\d)\d{4}(?!\d)"),
]
_WORDS = ("river stone lantern quiet maple orbit velvet harbor ember meadow signal paper "
          "copper garden window thunder pocket silver canyon mirror").split()  # fmt: skip


def find_context_secret(system: str) -> str | None:
    for pat in _SECRET_PATTERNS:
        m = pat.search(system)
        if m:
            return m.group(1) if m.groups() else m.group(0)
    return None


def _h(*parts: object) -> int:
    return zlib.crc32("\x1f".join(map(str, parts)).encode("utf-8"))


def _render(messages: Sequence[Message]) -> str:
    return "".join(f"<|{m['role']}|>{m['content']}\n" for m in messages) + "<|assistant|>"


class DummyTokenLevel(TokenLevel):
    VOCAB = [chr(i) for i in range(32, 127)] + ["\n"]
    COPY_BONUS = 4.0

    def __init__(self) -> None:
        self.eos_id = len(self.VOCAB)
        self._index = {c: i for i, c in enumerate(self.VOCAB)}
        self._prompt_cache: dict[tuple[int, ...], str] = {}
        self._copy_cache: dict[tuple[str, str], list[int]] = {}
        base = np.random.default_rng(2026).normal(0.0, 1.0, size=(4096, len(self.VOCAB) + 1))
        base[:, self._index[" "]] += 1.5
        for c in "etaoinshrdlu":
            base[:, self._index[c]] += 1.0
        self._noise = base

    def encode_text(self, text: str) -> list[int]:
        return [self._index.get(c, self._index["?"]) for c in text]

    def encode_prompt(self, messages: list[Message], thinking: bool | None = None) -> list[int]:
        return self.encode_text(_render(messages))

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.VOCAB[i] for i in ids if 0 <= i < len(self.VOCAB))

    def next_logprobs(self, prompt_ids: Sequence[int], generated: Sequence[int]) -> np.ndarray:
        key = tuple(prompt_ids)
        ptext = self._prompt_cache.get(key)
        if ptext is None:
            ptext = self.decode(prompt_ids)
            if len(self._prompt_cache) > 256:
                self._prompt_cache.clear()
            self._prompt_cache[key] = ptext
        tail = self.decode(generated[-2:])
        logits = self._noise[_h(tail, len(generated) % 7) % 4096].copy()
        if len(tail) == 2:
            ck = (ptext, tail)
            nxt = self._copy_cache.get(ck)
            if nxt is None:
                nxt = []
                start = ptext.find(tail)
                while start != -1 and start + 2 < len(ptext):
                    nxt.append(self._index.get(ptext[start + 2], self._index["?"]))
                    start = ptext.find(tail, start + 1)
                if len(self._copy_cache) > 200_000:
                    self._copy_cache.clear()
                self._copy_cache[ck] = nxt
            for j in nxt:
                logits[j] += self.COPY_BONUS
        logits[self.eos_id] += len(generated) / 30.0 - 2.0
        return log_softmax(logits)


@register("model", "dummy")
class DummyModel(ModelWrapper):
    name = "dummy"

    def __init__(self, cfg: dict | None = None) -> None:
        cfg = dict(cfg or {})
        self.cfg = cfg
        self.leak_rate = float(cfg.get("leak_rate", 0.3))
        self.thinking = bool(cfg.get("thinking", False))
        self._tl = DummyTokenLevel()

    @property
    def is_dummy(self) -> bool:
        return True

    def token_level(self) -> TokenLevel:
        return self._tl

    # -- generation ------------------------------------------------------------------------
    def generate(
        self,
        prompts: Sequence[list[Message]],
        gen_cfg: GenConfig,
        seeds: Sequence[int | None] | None = None,
        secret_ctx: Sequence[Any] | None = None,
        **options: Any,
    ) -> list[Generation]:
        # secret_ctx / options (e.g. input_hook) have no effect: there are no tokens to mask.
        self.last_secret_ctx = list(secret_ctx) if secret_ctx is not None else None
        seeds = list(seeds) if seeds is not None else [None] * len(prompts)
        thinking = self.thinking if gen_cfg.thinking is None else gen_cfg.thinking
        return [self._one(p, gen_cfg, s, thinking) for p, s in zip(prompts, seeds)]

    def _one(self, msgs: list[Message], gen_cfg: GenConfig, seed: int | None, thinking: bool) -> Generation:
        t0 = time.perf_counter()
        system = "\n".join(m["content"] for m in msgs if m["role"] == "system")
        users = [m["content"] for m in msgs if m["role"] in ("user", "tool")]
        user = users[-1] if users else ""
        secret = find_context_secret(system)
        key = _h(_render(msgs), "greedy" if gen_cfg.greedy else seed)
        rng = np.random.default_rng(key)
        parts: list[str] = []
        low = user.lower()
        if secret and secret in user and ("access" in system.lower() or "granted" in system.lower()):
            parts.append("Access Granted.")
        if "match" in system.lower() and re.search(r"(?<!\d)\d{4}(?!\d)", user):
            code = re.search(r"(?<!\d)\d{4}(?!\d)", user).group(0)  # type: ignore[union-attr]
            parts.append("MATCH" if secret == code else "NO MATCH")
            parts.append(", ".join(str(int(x)) for x in rng.integers(0, 100, size=5)))
        if re.search(r"(?m)^\s*\(?[A-D][\).]", user) or "answer with the letter" in low:
            parts.append(f"Answer: {'ABCD'[int(rng.integers(0, 4))]}")
        if "<tool_call>" in system and secret and any(w in low for w in ("call", "connect", "fetch", "query")):
            parts.append(f'<tool_call>{{"name": "http_request", "arguments": {{"auth": "{secret}"}}}}</tool_call>')
        if not parts:
            words = " ".join(_WORDS[int(i)] for i in rng.integers(0, len(_WORDS), size=12))
            parts.append(f"Here is my reply about {user[:40]!r}: {words}.")
        if secret and rng.random() < self.leak_rate:
            parts.append(f"(By the way, it is {secret}.)")
        reasoning = f"The context holds {secret}; I should not repeat it." if thinking and secret else ""
        full = " ".join(parts)
        cap = max(gen_cfg.max_new_tokens * 4, 16)  # ~4 characters per "token"
        text = full[:cap]
        return Generation(text=text, reasoning=reasoning, n_tokens=len(text) // 4 + 1,
                          latency_s=time.perf_counter() - t0, finished=len(full) <= cap)  # fmt: skip

    # -- scoring ---------------------------------------------------------------------------
    def logprobs(self, prompts: Sequence[list[Message]], continuations: Sequence[str],
                 secret_ctx: Sequence[Any] | None = None, **options: Any) -> list[float]:  # fmt: skip
        out = []
        for msgs, cont in zip(prompts, continuations):
            p_ids = self._tl.encode_prompt(msgs)
            c_ids = self._tl.encode_text(cont)
            total = 0.0
            for i, tok in enumerate(c_ids):
                total += float(self._tl.next_logprobs(p_ids, c_ids[:i])[tok])
            out.append(total)
        return out
