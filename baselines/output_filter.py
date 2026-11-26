"""Baseline defence `output_filter`: redact the secret (and, in mode `rule`, the protected info)
from every generated reply.

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

Modes:
  naive (default)  redact the secret only (PasswordEval: the password), in every reply.
  rule             as naive, and ALSO redact every protected string of the item
                   (SecretContext.protected; PasswordEval: the confidential info) unless a user turn
                   of the prompt contains the secret (same case-insensitive, separator-tolerant
                   matching). It is given the ground-truth protected string and PasswordEval's own
                   access rule, so it is an ORACLE UPPER BOUND for a string filter, not a defence
                   claim; info() records `upper_bound: true` and the spec's `label`.
                   On items without protected strings it behaves as naive.

Config: {name: output_filter, mode: naive|rule, replacement: "[REDACTED]", min_spaced: 4, max_sep: 3,
         label: <optional text recorded in summary.json defence_info>}
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
MODES = ("naive", "rule")


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


def user_gave_secret(messages: Sequence[Message], secret: str, min_spaced: int = 4, max_sep: int = 3) -> bool:
    """True when a user turn contains the secret (the same matching as the redaction)."""
    pat = secret_pattern(secret, min_spaced, max_sep)
    return bool(pat) and any(m.get("role") == "user" and pat.search(m.get("content") or "") for m in messages)


class _FilterWrapper(DelegatingWrapper):
    def __init__(self, inner: ModelWrapper, cfg: dict[str, Any]) -> None:
        super().__init__(inner, "output_filter")
        self.mode = str(cfg.get("mode", "naive"))
        if self.mode not in MODES:
            raise ValueError(f"output_filter: mode must be one of {MODES}, got {self.mode!r}")
        self.replacement = str(cfg.get("replacement", "[REDACTED]"))
        self.min_spaced = int(cfg.get("min_spaced", 4))
        self.max_sep = int(cfg.get("max_sep", 3))

    def _redact(self, text: str, strings: Sequence[str]) -> str:
        for s in strings:
            text = redact(text, s, self.replacement, self.min_spaced, self.max_sep)
        return text

    def generate(self, prompts: Sequence[list[Message]], gen_cfg: GenConfig,
                 seeds: Sequence[int | None] | None = None,
                 secret_ctx: Sequence[SecretContext | None] | None = None, **options: Any) -> list[Generation]:  # fmt: skip
        gens = super().generate(prompts, gen_cfg, seeds, secret_ctx=secret_ctx, **options)
        ctxs = list(secret_ctx) if secret_ctx is not None else [None] * len(gens)
        out = []
        for msgs, g, c in zip(prompts, gens, ctxs):
            if c is None:
                out.append(g)
                continue
            strings: list[str] = []
            if self.mode == "rule" and c.protected and not user_gave_secret(msgs, c.secret, self.min_spaced, self.max_sep):
                strings += list(c.protected)  # the protected info first: it may contain the secret
            strings.append(c.secret)
            out.append(dataclasses.replace(g, text=self._redact(g.text, strings), reasoning=self._redact(g.reasoning, strings)))
        return out


@register("defence", "output_filter")
class OutputFilter(Defence):
    name = "output_filter"

    def wrap(self, model: ModelWrapper) -> ModelWrapper:
        return _FilterWrapper(model, self.cfg)

    def info(self) -> dict[str, Any]:
        """Recorded in summary.json (defence_info)."""
        mode = str(self.cfg.get("mode", "naive"))
        return {"name": self.name, "mode": mode, "upper_bound": mode == "rule", "label": self.cfg.get("label"),
                "replacement": str(self.cfg.get("replacement", "[REDACTED]")),
                "min_spaced": int(self.cfg.get("min_spaced", 4)), "max_sep": int(self.cfg.get("max_sep", 3))}  # fmt: skip
