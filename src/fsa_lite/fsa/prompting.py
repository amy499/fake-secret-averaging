"""System prompts and chat rendering shared by generation (backend HF generator) and scoring
(likelihood attacker), so both see exactly the same prompt token IDs.

prompt_ids() is THE definition of what the model is conditioned on: the method's system
template from configs/training/prompts.yaml with the secret filled in, the user prompt, rendered with
the tokenizer's chat template plus the assistant generation prompt, tokenised without adding
special tokens (the chat template already contains any BOS). Templates that reject a system
role get the system text prepended to the user turn instead.

Digit secrets use configs/training/prompts.yaml system_templates; word secrets use
word_system_templates (same keys), chosen by load_system_templates(secret_type=...). digits4
secrets use digits4_system_templates: its "base" and "pe_base" are the same PasswordEval-style
prompt with {secret} (the 4-digit code) and {password} (fsa.secrets.pe_password(secret)), so
every adapter method (which uses "base") and the pe_base training method see the same text.
A template without {password} ignores it, so digit and word prompts are unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
PROMPTS_YAML = ROOT / "configs" / "training" / "prompts.yaml"

# Prompt-only methods: a different system prompt on the unmodified base model (no adapter).
PROMPT_ONLY_TEMPLATES = ("prompt_hiding", "prompt_hiding_strict", "prompt_anti_suppression")
# Training-only system methods that name their own template (digits4: pe_base == base).
NAMED_TEMPLATES = ("pe_base",)


TEMPLATE_SECTIONS = {
    "digits": "system_templates",
    "words": "word_system_templates",
    "digits4": "digits4_system_templates",
}


def load_system_templates(path: Path = PROMPTS_YAML, secret_type: str = "digits") -> dict[str, str]:
    """The system templates for a secret type (digits by default, as before)."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    section = TEMPLATE_SECTIONS[secret_type]
    if section not in raw:
        raise ValueError(f"{path.name} has no {section} (needed for {secret_type} secrets)")
    return raw[section]


def system_prompt(templates: Mapping[str, str], method: str, secret: str) -> str:
    """Prompt-only methods (and pe_base) use their own template; every other method sees the
    base prompt. {password} is filled with the secret's PasswordEval password when present."""
    key = method if method in PROMPT_ONLY_TEMPLATES or method in NAMED_TEMPLATES else "base"
    if key not in templates:
        raise KeyError(f"no system template {key!r} for method {method!r}")
    template = templates[key]
    if "{password}" in template:
        from fsa.secrets import pe_password

        return template.format(secret=secret, password=pe_password(secret))
    return template.format(secret=secret)


def render_chat(tokenizer: Any, system: str, user: str) -> str:
    """Chat-template text ending with the assistant turn opener."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        # Chat templates signal "no system role" with jinja2's TemplateError (matched by name so
        # this module doesn't need jinja2); anything else is a real error.
        if type(e).__name__ != "TemplateError":
            raise
    merged = [{"role": "user", "content": f"{system}\n\n{user}"}]
    return tokenizer.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)


def chat_ids(tokenizer: Any, system: str, user: str) -> list[int]:
    """Token IDs of a chat prompt from raw system text (e.g. the no_secret template)."""
    return list(
        tokenizer(render_chat(tokenizer, system, user), add_special_tokens=False)["input_ids"]
    )


def prompt_ids(
    tokenizer: Any, templates: Mapping[str, str], method: str, secret: str, user: str
) -> list[int]:
    """Token IDs of the full generation prompt for one secret (exactly what generation uses)."""
    return chat_ids(tokenizer, system_prompt(templates, method, secret), user)


# How every reply is sampled (generation, on-policy training) and what the likelihood attacker
# assumes: pure temperature sampling over the full vocabulary.
SAMPLING = {"top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0}


def eos_ids(tokenizer: Any, model: Any) -> set[int]:
    """Every token that ends a reply: generation_config.eos_token_id (int or list) + tokenizer's."""
    ids = set()
    cfg = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    for x in cfg if isinstance(cfg, list | tuple) else [cfg]:
        if x is not None:
            ids.add(int(x))
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    return ids


def until_eos(row: list[int], eos: set[int]) -> list[int]:
    """Generated IDs up to and including the first EOS (later positions are padding)."""
    for i, t in enumerate(row):
        if t in eos:
            return row[: i + 1]
    return row


SPAN_USER_PROMPT = "Hello."  # any fixed user turn; only the system prompt's secret varies


def secret_span_tokens(
    tokenizer: Any, templates: Mapping[str, str], method: str, secrets: list[str]
) -> dict[str, int]:
    """secret -> how many tokens it takes inside the rendered chat prompt (system prompt with
    the method's template, fixed user turn). The tokens shared by every secret's prompt, the
    longest common prefix and suffix, are the context; the rest is the secret's span, including
    any token it merges with its neighbours. Needs at least two distinct secrets."""
    if len(set(secrets)) < 2:
        raise ValueError("need at least two distinct secrets to find the shared context")
    ids = {s: prompt_ids(tokenizer, templates, method, s, SPAN_USER_PROMPT) for s in secrets}
    rows = list(ids.values())
    shortest = min(len(r) for r in rows)
    pre = 0
    while pre < shortest and len({r[pre] for r in rows}) == 1:
        pre += 1
    suf = 0
    while suf < shortest - pre and len({r[-1 - suf] for r in rows}) == 1:
        suf += 1
    return {s: len(r) - pre - suf for s, r in ids.items()}


def make_prompt_builder(
    tokenizer: Any, templates: Mapping[str, str], method: str, user: str
) -> Callable[[str], list[int]]:
    """secret -> prompt token IDs for a fixed method and user prompt (cached per secret)."""
    cache: dict[str, list[int]] = {}

    def build(secret: str) -> list[int]:
        if secret not in cache:
            cache[secret] = prompt_ids(tokenizer, templates, method, secret, user)
        return list(cache[secret])

    return build
