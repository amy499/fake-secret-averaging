"""FSA-lite context helpers: fake-secret sampling, fake-context construction and prompt encoding,
shared by lite_source.py, preflight_fakes.py and the diagnostics.

Status: ✅ implemented + tested on CPU (src/fsa_lite/lite/selftest_v2.py, selftest_run3.py).

A fake context is the real one with every occurrence of the real secret string replaced by a fake
(system text and earlier turns). The reply tokens are the frozen model's own reply under the real
secret, identical in every context. The chat text is tokenizer.apply_chat_template(...,
add_generation_prompt=True, tokenize=False) encoded with add_special_tokens=False.
"""

from __future__ import annotations

from typing import Any


def sample_fakes(rng: Any, pool: list[tuple[int, str]], real: str, k: int, also_exclude: tuple = ()) -> list[str]:
    """K distinct fakes of the real secret's type, never equal to it (nor to also_exclude, e.g. a
    value the user turn quotes, so that text stays unambiguous)."""
    bad = {real, *also_exclude}
    cand = [s for _, s in pool if s not in bad]
    if len(cand) < k:
        raise ValueError(f"pool has {len(cand)} other secrets, K={k}")
    return [cand[int(i)] for i in rng.choice(len(cand), size=k, replace=False)]


def substitute(messages: list[dict], real: str, fake: str, word_boundary: bool = False) -> list[dict]:
    """The fake context: every occurrence of the real secret string replaced (assistant turns too).
    word_boundary=True (FSA-lite: free-text access codes and memory values): only occurrences not
    inside a longer word, in any case (lite_common.sub_wb); the default replaces every occurrence."""
    if word_boundary:
        from lite_common import sub_wb

        return [{**m, "content": sub_wb(m["content"], real, fake)} for m in messages]
    return [{**m, "content": m["content"].replace(real, fake)} for m in messages]


def prompt_ids(tok: Any, messages: list[dict]) -> list[int]:
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return list(tok(text, add_special_tokens=False)["input_ids"])
