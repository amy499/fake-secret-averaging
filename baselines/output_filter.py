"""Baseline defence `output_filter`: redact the secret from every generated reply.

Status: ✅ implemented + tested (eval/tests/test_output_filter.py, with the dummy model). 🟡 on real
models (not yet run).

The model generates as usual; afterwards every occurrence of the prompt's secret (the
SecretContext both runners pass to `generate`) is replaced by `replacement` (default "[REDACTED]"):
  * exact and case-insensitive matches;
  * separator-tolerant matches for secrets of >= `min_spaced` alphanumeric characters (default 4):
    the secret's alphanumerics with up to `max_sep` separator characters (spaces, '-', '_', '.',
    ',', '/', '|') between each, so "1 2 3 4" or "s-e-c-r-e-t" are caught too.
Prompts without a secret (plain MMLU / IFEval / GSM8K items, no-secret controls) pass through
unchanged. It does not catch encodings (base64, spelled-out digits, paraphrases): it is the
string-filter baseline, not a defence against a determined attacker. Log-probabilities are the
inner model's (an attacker scoring candidates sees the unfiltered model).

Config: {name: output_filter, replacement: "[REDACTED]", min_spaced: 4, max_sep: 3}
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence
from typing import Any

from baselines.basic import DelegatingWrapper
from core.interfaces import Defence, GenConfig, Generation, Message, ModelWrapper, SecretContext
from core.registry import register

SEPARATORS = r"[\s\-_.,/|]"


def secret_pattern(secret: str, min_spaced: int = 4, max_sep: int = 3) -> re.Pattern[str] | None:
    """One case-insensitive pattern for the secret: the literal string, and for secrets with at
    least `min_spaced` alphanumerics, those characters with up to `max_sep` separators between them."""
    if not secret:
        return None
    alts = [re.escape(secret)]
    chars = [c for c in secret if c.isalnum()]
    if len(chars) >= min_spaced:
        alts.append(f"{SEPARATORS}{{0,{max_sep}}}".join(re.escape(c) for c in chars))
    return re.compile("|".join(f"(?:{a})" for a in alts), re.IGNORECASE)


def redact(text: str, secret: str | None, replacement: str = "[REDACTED]", min_spaced: int = 4, max_sep: int = 3) -> str:
    pat = secret_pattern(secret or "", min_spaced, max_sep)
    return pat.sub(replacement, text) if pat else text


class _FilterWrapper(DelegatingWrapper):
    def __init__(self, inner: ModelWrapper, cfg: dict[str, Any]) -> None:
        super().__init__(inner, "output_filter")
        self.replacement = str(cfg.get("replacement", "[REDACTED]"))
        self.min_spaced = int(cfg.get("min_spaced", 4))
        self.max_sep = int(cfg.get("max_sep", 3))

    def generate(self, prompts: Sequence[list[Message]], gen_cfg: GenConfig,
                 seeds: Sequence[int | None] | None = None,
                 secret_ctx: Sequence[SecretContext | None] | None = None, **options: Any) -> list[Generation]:  # fmt: skip
        gens = super().generate(prompts, gen_cfg, seeds, secret_ctx=secret_ctx, **options)
        ctxs = list(secret_ctx) if secret_ctx is not None else [None] * len(gens)
        out = []
        for g, c in zip(gens, ctxs):
            if c is None:
                out.append(g)
                continue
            text = redact(g.text, c.secret, self.replacement, self.min_spaced, self.max_sep)
            reasoning = redact(g.reasoning, c.secret, self.replacement, self.min_spaced, self.max_sep)
            out.append(dataclasses.replace(g, text=text, reasoning=reasoning))
        return out


@register("defence", "output_filter")
class OutputFilter(Defence):
    name = "output_filter"

    def wrap(self, model: ModelWrapper) -> ModelWrapper:
        return _FilterWrapper(model, self.cfg)
