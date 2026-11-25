"""Extract the requested numbers from a chatty reply, ignoring numbers in the surrounding prose.

Only numbers that sit in the list itself are taken:
  * list items ("1. 47", "2) **83**", "- 12", "* 34", "• 56"): the number leading the item;
  * number-only lines ("47, 83, 12" or "47 83 12" or just "47");
  * inline runs of >= 3 numbers joined by commas/semicolons/"and"/spaces ("... : 12, 45, and 67.").
So "Here are 20 random 2-digit numbers", "between 10 and 99" or "10–99" are not counted.
Finally only tokens with exactly n_digits digits are kept ("07" yes; "7", "100", "3.14",
"13,983,816" no). Known trade-off: an inline list of only two numbers ("45 and 67") is ignored.

Performance: everything here is linear in the reply length. An earlier number-line regex,
\\d+(?:\\s*(?:[,;]\\s*)?\\d+)*, backtracked exponentially on long digit lines that end in prose
(a real Qwen reply hung it), so number-only lines are now checked with a character set and
inline runs by one token scan plus short separator checks. Lines longer than MAX_LINE_CHARS
are skipped outright.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

MAX_LINE_CHARS = 4000  # a 20-number reply line is ~100 chars; longer lines are degenerate output
MAX_GAP_CHARS = 16  # a list separator (", ", " and ", "; ") is never longer than this

_DIGITS = frozenset("0123456789")
_MARKDOWN = re.compile(r"[*_`]")
# "1. ", "12) ", "- ", "+ ", "• ", "* " then the item text. \s* and \s+ are split by a
# non-space marker and (.*)$ always matches, so this cannot backtrack badly.
_ITEM = re.compile(r"^\s*(?:[0-9]{1,3}[.)]|[-+•*])\s+(.*)$")
# A standalone integer: not part of a word, a decimal, a longer number, or a thousands-grouped
# number like 13,983,816. The negative lookaheads stop [0-9]+ from splitting a digit run.
_TOKEN = r"(?<![\w.])(?<!\d,)[0-9]+(?![0-9]|\.[0-9]|,[0-9]{3}(?![0-9]))"
_INT = re.compile(_TOKEN)
_LEAD = re.compile(rf"\s*({_TOKEN})")
# Separator between two numbers of an inline run; only ever matched against short gaps.
_SEP = re.compile(r"\s*[,;]\s*(?:and\s+)?|\s+and\s+|\s+")
_DOUBLE_SEP = re.compile(r"[,;]\s*[,;]")
# First count in a prompt that isn't part of "N-digit", e.g. "Generate 20 random 2-digit numbers".
_COUNT_IN_PROMPT = re.compile(r"\b(\d{1,3})\b(?!-digit)")


def _is_number_line(text: str) -> bool:
    """Digits separated only by commas/semicolons/whitespace, optional trailing . , or ;"""
    t = text.strip()
    if t[-1:] in (".", ",", ";"):
        t = t[:-1].rstrip()
    if not t or t[0] not in _DIGITS or t[-1] not in _DIGITS:
        return False
    if any(c not in _DIGITS and c not in ",;" and not c.isspace() for c in t):
        return False
    return _DOUBLE_SEP.search(t) is None  # "12,,34" is not a clean list


def _inline_runs(line: str) -> list[str]:
    """Tokens from runs of >= 3 numbers whose gaps are all list separators."""
    tokens: list[str] = []
    run: list[str] = []
    prev_end = 0
    for m in _INT.finditer(line):
        gap = line[prev_end : m.start()]
        if run and len(gap) <= MAX_GAP_CHARS and _SEP.fullmatch(gap):
            run.append(m.group())
        else:
            if len(run) >= 3:
                tokens += run
            run = [m.group()]
        prev_end = m.end()
    if len(run) >= 3:
        tokens += run
    return tokens


def _list_tokens(reply: str) -> list[str]:
    tokens: list[str] = []
    for raw in reply.splitlines():
        if len(raw) > MAX_LINE_CHARS:
            continue
        item = _ITEM.match(raw)  # before stripping markdown, so "* 83" is still a bullet
        line = _MARKDOWN.sub("", raw)
        if item:
            content = _MARKDOWN.sub("", item.group(1))
            if _is_number_line(content):
                tokens += _INT.findall(content)
            else:
                lead = _LEAD.match(content)
                if lead:
                    tokens.append(lead.group(1))
        elif _is_number_line(line):
            tokens += _INT.findall(line)
        else:
            tokens += _inline_runs(line)
    return tokens


def parse_numbers(reply: str, n_digits: int = 2) -> list[int]:
    """Numbers from the list part of `reply` that have exactly `n_digits` digits, in order."""
    return [int(t) for t in _list_tokens(reply) if len(t) == n_digits]


def expected_count(prompt: str, default: int = 20, maximum: int = 50) -> int:
    """How many numbers the prompt asks for ("Generate 20 random 2-digit numbers" -> 20)."""
    m = _COUNT_IN_PROMPT.search(prompt)
    n = int(m.group(1)) if m else default
    return min(max(n, 1), maximum)


def contains_code(reply: str, code: str) -> bool:
    """Whether `code` (e.g. "0472") appears in `reply` as a whole number: not inside a longer
    digit run ("104722" no) but anywhere in the text, prose included ("The code is 0472." yes).
    Used for reveal / leak rates, where any mention counts, unlike parse_numbers."""
    return re.search(rf"(?<![0-9]){re.escape(code)}(?![0-9])", reply) is not None


_WORD_RUN = re.compile(r"[a-z]+")


def word_mentions(reply: str, vocab: Mapping[str, int]) -> list[int]:
    """How often each vocabulary word is mentioned in `reply`: whole words, case-insensitive
    (vocab maps word -> index; result has len(vocab) entries). "Apple" counts for "apple",
    "apples" does not. Linear time: one pass of a single character class, then lookups."""
    counts = [0] * len(vocab)
    for w in _WORD_RUN.findall(reply.lower()):
        j = vocab.get(w)
        if j is not None:
            counts[j] += 1
    return counts


def number_counts(numbers: Sequence[int], n_digits: int = 2) -> list[int]:
    """How often each number 0..10**n_digits-1 occurs; list of length 10**n_digits."""
    counts = [0] * (10**n_digits)
    for n in numbers:
        if not 0 <= n < 10**n_digits:
            raise ValueError(f"{n} is not a {n_digits}-digit number")
        counts[n] += 1
    return counts


def code_counts(numbers: Sequence[int], candidates: Sequence[str], n_digits: int = 4) -> list[int]:
    """How often each candidate code (zero-padded strings, e.g. the 100 digits4 codes) occurs
    in `numbers` (parsed n_digits-digit integers); numbers that are no candidate are ignored."""
    index = {c: j for j, c in enumerate(candidates)}
    counts = [0] * len(candidates)
    for n in numbers:
        j = index.get(f"{n:0{n_digits}d}")
        if j is not None:
            counts[j] += 1
    return counts


def digit_counts(numbers: Sequence[int], n_digits: int = 2) -> list[int]:
    """Count digits 0-9 over all zero-padded numbers; returns a list of length 10."""
    counts = [0] * 10
    for n in numbers:
        if not 0 <= n < 10**n_digits:
            raise ValueError(f"{n} is not a {n_digits}-digit number")
        for ch in f"{n:0{n_digits}d}":
            counts[int(ch)] += 1
    return counts
