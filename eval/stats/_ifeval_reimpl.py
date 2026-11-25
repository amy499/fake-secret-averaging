"""Independent IFEval re-implementation, used by the re-scorer (stats/rescore.py) as the second,
separate scorer of IFEval replies (protocol check 3). The protocol's main IFEval score comes from
Google's checker (benchmarks/_ifeval_google.py); this file never imports it.

Status: ✅ implemented + tested (tests/test_benchmarks.py, tests/test_rescore.py). It re-implements
the 25 instruction types of Zhou et al. (2023) following google-research/instruction_following_eval.
Known deviations from Google's checker, which is why rescore disagreements on IFEval are written
out for review, not averaged away: sentence splitting uses a regex instead of nltk punkt; the
language check needs langdetect (None without it); the regexes are simpler in places.

  check_all(text, ids, kwargs)        strict: per instruction, True / False / None (cannot check)
  check_all_loose(text, ids, kwargs)  loose: an instruction passes if it passes on any of Google's
                                      eight response variants (first / last line removed, '*' removed)
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any


def _cmp(value: int, target: int, relation: str | None) -> bool:
    return value < target if relation == "less than" else value >= target


def _words(t: str) -> list[str]:
    return re.findall(r"\b\w+\b", t)


def _sentences(t: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s]


def _json_ok(t: str) -> bool:
    s = t.strip().removeprefix("```json").removeprefix("```Json").removeprefix("```JSON")
    s = s.removeprefix("```").removesuffix("```").strip()
    try:
        json.loads(s)
        return True
    except ValueError:
        return False


def _lang(t: str, language: str) -> bool | None:
    try:
        import langdetect  # optional
    except ImportError:
        return None
    langdetect.DetectorFactory.seed = 0  # langdetect samples at random: unseeded, the answer can change run to run
    try:
        return langdetect.detect(t) == language
    except Exception:  # noqa: BLE001
        return False


CHECKS: dict[str, Callable[[str, dict[str, Any], str], bool | None]] = {
    "punctuation:no_comma": lambda t, k, p: "," not in t,
    "detectable_format:number_bullet_lists": lambda t, k, p: len(re.findall(r"(?m)^\s*[\*\-]\s", t)) == k["num_bullets"],
    "detectable_format:title": lambda t, k, p: bool(re.search(r"<<[^\n]+>>", t)),
    "detectable_format:json_format": lambda t, k, p: _json_ok(t),
    "detectable_format:number_highlighted_sections": lambda t, k, p: len(re.findall(r"\*[^\n\*]+\*", t)) >= k["num_highlights"],
    "detectable_format:multiple_sections": lambda t, k, p: len(re.split(rf"\s?{re.escape(k['section_spliter'])}\s?\d+\s?", t)) - 1 >= k["num_sections"],
    "detectable_format:constrained_response": lambda t, k, p: any(
        o in t for o in ("My answer is yes.", "My answer is no.", "My answer is maybe.")),
    "change_case:english_lowercase": lambda t, k, p: t == t.lower() and any(c.isalpha() for c in t),
    "change_case:english_capital": lambda t, k, p: t == t.upper() and any(c.isalpha() for c in t),
    "change_case:capital_word_frequency": lambda t, k, p: _cmp(
        len([w for w in _words(t) if w.isupper()]), k["capital_frequency"], k.get("capital_relation")),
    "length_constraints:number_words": lambda t, k, p: _cmp(len(_words(t)), k["num_words"], k.get("relation")),
    "length_constraints:number_sentences": lambda t, k, p: _cmp(len(_sentences(t)), k["num_sentences"], k.get("relation")),
    "length_constraints:number_paragraphs": lambda t, k, p: len(
        [x for x in re.split(r"\s?\*\*\*\s?", t) if x.strip()]) == k["num_paragraphs"],
    "length_constraints:nth_paragraph_first_word": lambda t, k, p: _nth_para(t, k),
    "keywords:existence": lambda t, k, p: all(re.search(re.escape(w), t, re.I) for w in k["keywords"]),
    "keywords:forbidden_words": lambda t, k, p: not any(
        re.search(rf"\b{re.escape(w)}\b", t, re.I) for w in k["forbidden_words"]),
    "keywords:frequency": lambda t, k, p: _cmp(
        len(re.findall(re.escape(k["keyword"]), t, re.I)), k["frequency"], k.get("relation")),
    "keywords:letter_frequency": lambda t, k, p: _cmp(
        t.lower().count(k["letter"].lower()), k["let_frequency"], k.get("let_relation")),
    "startend:end_checker": lambda t, k, p: t.strip().lower().endswith(k["end_phrase"].strip().lower()),
    "startend:quotation": lambda t, k, p: len(t.strip()) > 1 and t.strip()[0] == '"' and t.strip()[-1] == '"',
    "detectable_content:postscript": lambda t, k, p: bool(
        re.search(r"^\s*" + re.escape(k["postscript_marker"]), t, re.I | re.M)),
    "detectable_content:number_placeholders": lambda t, k, p: len(re.findall(r"\[.*?\]", t)) >= k["num_placeholders"],
    "combination:repeat_prompt": lambda t, k, p: t.strip().lower().startswith(k["prompt_to_repeat"].strip().lower()),
    "combination:two_responses": lambda t, k, p: _two(t),
    "language:response_language": lambda t, k, p: _lang(t, k["language"]),
}  # fmt: skip


def _nth_para(t: str, k: dict[str, Any]) -> bool:
    paras = [x for x in re.split(r"\n\n", t) if x.strip()]
    if len(paras) != k["num_paragraphs"]:
        return False
    n = k["nth_paragraph"]
    if n > len(paras):
        return False
    first = re.sub(r"[^\w]", "", paras[n - 1].strip().split()[0]).lower() if paras[n - 1].strip() else ""
    return first == str(k["first_word"]).lower()


def _two(t: str) -> bool:
    parts = [x.strip() for x in t.split("******")]
    return len(parts) == 2 and all(parts) and parts[0] != parts[1]


def check_all(text: str, ids: list[str], kwargs: list[dict[str, Any]], prompt: str = "") -> list[bool | None]:
    out: list[bool | None] = []
    for iid, kw in zip(ids, kwargs):
        fn = CHECKS.get(iid)
        kw = {k: v for k, v in (kw or {}).items() if v is not None}
        try:
            out.append(None if fn is None else fn(text, kw, prompt))
        except (KeyError, TypeError, IndexError):
            out.append(None)
    return out


def loose_variants(text: str) -> list[str]:
    """The eight responses Google's loose checker tries (evaluation_lib.test_instruction_following_loose)."""
    r = text.split("\n")
    first, last, both = "\n".join(r[1:]).strip(), "\n".join(r[:-1]).strip(), "\n".join(r[1:-1]).strip()
    return [text, text.replace("*", ""), first, last, both, first.replace("*", ""), last.replace("*", ""),
            both.replace("*", "")]


def check_all_loose(text: str, ids: list[str], kwargs: list[dict[str, Any]], prompt: str = "") -> list[bool | None]:
    variants = [v for v in loose_variants(text) if v.strip()]
    out: list[bool | None] = []
    for k, iid in enumerate(ids):
        res = [check_all(v, [iid], [kwargs[k]], prompt)[0] for v in variants]
        out.append(None if any(x is None for x in res) else any(res))
    return out
