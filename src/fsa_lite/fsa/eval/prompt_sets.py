"""User prompts and named prompt sets from configs/training/prompts.yaml, and how an evaluation spreads
each secret's replies over a set.

user_prompts entries have an id, a text and an optional n_numbers (numbers asked for per
reply). n_numbers may be omitted or 0 for everyday or writing prompts that ask for no numbers;
both mean 0.

prompt_sets maps a name to an ordered list of user prompt ids:

    prompt_sets:
      my_set:
        prompts: [rand_20_2digit, some_writing_prompt]

Assignment (scripts/baseline_leakage.py --prompt-set): --replies must be a multiple of the
set size, r = replies / len(set). Every secret, train and held-out alike, gets exactly r
replies per prompt, in set order: reply_index j goes to prompts[j // r]. Each (secret, prompt)
pair is one generate call with r replies, seeded by generation_seed(). Prompt 0 uses the
single-prompt formula, so a single-prompt run is the special case of a one-prompt set.

Evaluation-only prompts, which training must never use (checked at training startup by
check_train_prompts):

    eval_only_prompts: [lottery]        # single prompts
    prompt_sets:
      task_prompts:
        eval_only: true                 # every prompt in the set
        prompts: [...]
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fsa.prompting import PROMPTS_YAML

ASSIGNMENT_RULE = (
    "every secret (train and held-out) gets replies_per_prompt replies per prompt, in set "
    "order: reply_index j -> prompts[j // replies_per_prompt]"
)
SEED_RULE = (
    "one generate call per (secret, prompt p); seed = seed*10000 + offset + i for p = 0 and "
    "int(numpy.random.default_rng([seed, offset, i, p]).integers(2**31)) for p >= 1, where "
    "offset = 0 (train) or 5000 (held-out) and i = the secret's 1-based position in its split"
)


@dataclass(frozen=True)
class UserPrompt:
    id: str
    text: str
    n_numbers: int  # numbers asked for per reply; 0 = the prompt asks for no numbers


@dataclass(frozen=True)
class PromptSet:
    name: str
    prompts: tuple[UserPrompt, ...]

    @property
    def ids(self) -> list[str]:
        return [p.id for p in self.prompts]


def load_prompts_yaml(path: Path = PROMPTS_YAML) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse_user_prompts(raw: Mapping[str, Any]) -> dict[str, UserPrompt]:
    """id -> UserPrompt; n_numbers omitted, null or 0 all mean "asks for no numbers"."""
    out: dict[str, UserPrompt] = {}
    for entry in raw.get("user_prompts") or []:
        pid = entry["id"]
        n = entry.get("n_numbers")
        n = 0 if n is None else n
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError(f"user prompt {pid!r}: n_numbers must be an integer >= 0, got {n!r}")
        if pid in out:
            raise ValueError(f"user prompt id {pid!r} is defined twice")
        out[pid] = UserPrompt(pid, entry["text"], n)
    return out


def load_user_prompts(path: Path = PROMPTS_YAML) -> dict[str, UserPrompt]:
    return parse_user_prompts(load_prompts_yaml(path))


def parse_prompt_set(raw: Mapping[str, Any], name: str) -> PromptSet:
    """The named set, validated: a non-empty list of known, distinct user prompt ids."""
    sets = raw.get("prompt_sets") or {}
    if name not in sets:
        raise ValueError(f"unknown prompt set {name!r}; defined: {sorted(sets)}")
    entry = sets[name]
    ids = entry.get("prompts") if isinstance(entry, Mapping) else None
    if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
        raise ValueError(f"prompt set {name!r} needs a non-empty list of prompt ids under prompts")
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"prompt set {name!r} lists {dupes} more than once")
    user = parse_user_prompts(raw)
    unknown = [i for i in ids if i not in user]
    if unknown:
        raise ValueError(
            f"prompt set {name!r}: unknown user prompt ids {unknown}; known: {sorted(user)}"
        )
    return PromptSet(name, tuple(user[i] for i in ids))


def load_prompt_set(name: str, path: Path = PROMPTS_YAML) -> PromptSet:
    return parse_prompt_set(load_prompts_yaml(path), name)


def replies_per_prompt(n_prompts: int, replies: int) -> int:
    """r = replies / n_prompts; replies must be a positive multiple of n_prompts."""
    if replies < 1 or replies % n_prompts:
        lower = n_prompts * max(1, replies // n_prompts)
        upper = n_prompts * (replies // n_prompts + 1)
        options = sorted({lower, upper})
        raise ValueError(
            f"--replies {replies} must be a multiple of the prompt set size {n_prompts} "
            f"(every secret gets the same number of replies per prompt); e.g. "
            + " or ".join(str(o) for o in options)
        )
    return replies // n_prompts


def generation_seed(seed: int, offset: int, i: int, p: int) -> int:
    """Generation seed for secret i (1-based) of a split and prompt p of the set (SEED_RULE)."""
    if p == 0:
        return seed * 10_000 + offset + i  # the single-prompt formula, unchanged
    return int(np.random.default_rng([seed, offset, i, p]).integers(2**31))


# -- evaluation-only prompts and the training-startup check -----------------------------------


def eval_only_prompts(raw: Mapping[str, Any]) -> dict[str, list[str]]:
    """prompt id -> why it is evaluation-only: listed in the top-level eval_only_prompts,
    and/or a member of a prompt set marked eval_only: true. Validates both markings."""
    user = parse_user_prompts(raw)
    reasons: dict[str, list[str]] = {}
    listed = raw.get("eval_only_prompts") or []
    if not isinstance(listed, list) or not all(isinstance(i, str) for i in listed):
        raise ValueError("eval_only_prompts must be a list of user prompt ids")
    unknown = [i for i in listed if i not in user]
    if unknown:
        raise ValueError(f"eval_only_prompts: unknown user prompt ids {unknown}")
    for pid in listed:
        reasons.setdefault(pid, []).append("eval_only_prompts")
    for name, entry in (raw.get("prompt_sets") or {}).items():
        flag = entry.get("eval_only", False) if isinstance(entry, Mapping) else False
        if not isinstance(flag, bool):
            # ValueError like every other config problem: train_fsa.main reports those cleanly.
            raise ValueError(  # noqa: TRY004
                f"prompt set {name!r}: eval_only must be true or false, got {flag!r}"
            )
        if flag:
            for p in parse_prompt_set(raw, name).prompts:
                reasons.setdefault(p.id, []).append(f"prompt set {name!r} (eval_only)")
    return reasons


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


PLACEHOLDERS = ("{password}", "{code}")  # filled per item; only need prompts may carry them


def check_train_prompts(
    raw: Mapping[str, Any],
    train_user_prompts: Sequence[str],
    *,
    heldout_user_prompts: Sequence[str] = (),
    need_user_prompts: Sequence[str] = (),
) -> None:
    """Raise ValueError naming every problem, unless no training user prompt (FSA prompts in
    train_user_prompts, need-term prompts in need_user_prompts) is unknown, declared held-out by
    the training config, marked evaluation-only in prompts.yaml, or has the same text as an
    evaluation-only prompt; general_prompts.train shares no prompt with general_prompts.eval
    (texts compared ignoring case and whitespace); FSA prompts carry no {password} / {code}
    (their fakes must differ only in the system prompt) and are not also need prompts; and
    every need prompt carries {password} or {code}. raw is the parsed configs/training/prompts.yaml."""
    user = parse_user_prompts(raw)
    problems = []
    for label, ids in (
        ("train_user_prompts", train_user_prompts),
        ("need.prompts", need_user_prompts),
    ):
        problems += _check_trained(raw, user, label, ids, heldout_user_prompts)
    both = sorted(set(train_user_prompts) & set(need_user_prompts))
    if both:
        problems.append(f"prompts in both train_user_prompts and need.prompts: {both}")
    with_ph = [
        p for p in train_user_prompts if p in user and any(x in user[p].text for x in PLACEHOLDERS)
    ]
    if with_ph:
        problems.append(
            f"train_user_prompts with a {{password}} / {{code}} placeholder (need-term only): "
            f"{with_ph}"
        )
    no_ph = [
        p
        for p in need_user_prompts
        if p in user and not any(x in user[p].text for x in PLACEHOLDERS)
    ]
    if no_ph:
        problems.append(f"need.prompts without a {{password}} or {{code}} placeholder: {no_ph}")
    general = raw.get("general_prompts") or {}
    eval_texts = {_norm(t) for t in general.get("eval") or []}
    overlap = [t for t in general.get("train") or [] if _norm(t) in eval_texts]
    if overlap:
        problems.append(f"general_prompts.train also in general_prompts.eval: {overlap}")
    if problems:
        raise ValueError("train prompt check failed: " + "; ".join(problems))


def _check_trained(
    raw: Mapping[str, Any],
    user: Mapping[str, UserPrompt],
    label: str,
    ids: Sequence[str],
    heldout_user_prompts: Sequence[str],
) -> list[str]:
    """Problems with one list of trained prompt ids (see check_train_prompts)."""
    problems = []
    unknown = [p for p in ids if p not in user]
    if unknown:
        problems.append(f"unknown {label} {unknown}; known: {sorted(user)}")
    declared = [p for p in ids if p in set(heldout_user_prompts)]
    if declared:
        problems.append(f"{label} also in the config's heldout_user_prompts: {declared}")
    eval_only = eval_only_prompts(raw)
    marked = [p for p in ids if p in eval_only]
    if marked:
        why = "; ".join(f"{p} ({', '.join(eval_only[p])})" for p in marked)
        problems.append(f"{label} marked evaluation-only in prompts.yaml: {why}")
    # A copy of an evaluation-only prompt under another id is still that prompt.
    eval_texts = {_norm(user[e].text): e for e in eval_only}
    copies = [
        f"{p} (= {eval_texts[_norm(user[p].text)]})"
        for p in ids
        if p in user and p not in eval_only and _norm(user[p].text) in eval_texts
    ]
    if copies:
        problems.append(f"{label} with the same text as an evaluation-only prompt: {copies}")
    return problems
