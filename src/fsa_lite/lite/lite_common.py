"""FSA-lite shared text helpers: normalisation, 13-grams, word-boundary matching, fake matching.

Status: ✅ implemented + tested on CPU (src/fsa_lite/lite/selftest_v2.py, lite tests).

Used by the data builders (tt_data.py, cimem_data.py, benign_data.py), the contamination check
(contam_check.py) and the training source (lite_source.py), so every step normalises and matches
text the same way:
  norm(s)       lowercase, whitespace runs collapsed to one space, stripped
  ngrams(s, n)  word n-grams of norm(s) (words = whitespace-separated tokens), default n = 13
  wb_regex(s)   s as a literal, not preceded or followed by a letter, digit or underscore
                (works for codes that start or end with punctuation or emoji, unlike \\b)
run3 (key-guarded payloads):
  rnorm(s)      the harness robust-leak normalisation (stats/robust_leak.py DEFINITION_VERSION 1):
                casefold, zero-width characters removed, only str.isalnum characters kept
  robust_contains(text, s)   rnorm(s) (>= 1 character) is a substring of rnorm(text)
  robust_spans(text, values, fragments)   character spans of text holding a value in normalised
                form, and (fragments=True) any contiguous piece of the normalised value of length
                >= max(PARTIAL_MIN, ceil(PARTIAL_FRACTION * L)) (robust_leak's partial length; no
                word-coverage guard: these spans only up-weight training tokens)
  sub_many(text, mapping)    every word-bounded occurrence (any case) of each key replaced by its
                value in ONE pass, longest key first (a near-miss password that is a prefix of the
                password is never rewritten inside it)
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

NGRAM = 13
PARTIAL_FRACTION, PARTIAL_MIN = 0.6, 4  # = harness stats/robust_leak.py (DEFINITION_VERSION 1)
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_WORD_CHAR = r"A-Za-z0-9_"

# Common words that make a useless secret (a fake substituted at every "please" would rewrite the
# whole prompt, and the real code would be guessable). Lowercase; compared with str.lower().
STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have
having he her here hers herself him himself his how i if in into is it its itself just me more most
my myself no nor not now of off on once only or other our ours ourselves out over own same she should
so some such than that the their theirs them themselves then there these they this those through to
too under until up very was we were what when where which while who whom why will with would you your
yours yourself yourselves
access granted password passcode passphrase code secret please hello hi hey yes no okay ok true false
none null test admin user login open sesame banana apple orange cat dog love money bank vault key
door enter welcome thanks thank sorry denied deny allow allowed answer reply prompt system assistant
instructions ignore
""".split())


def norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def ngrams(s: str, n: int = NGRAM) -> set[str]:
    w = norm(s).split()
    return {" ".join(w[i : i + n]) for i in range(len(w) - n + 1)}


def wb_regex(s: str, ignore_case: bool = True) -> re.Pattern:
    return re.compile(rf"(?<![{_WORD_CHAR}]){re.escape(s)}(?![{_WORD_CHAR}])", re.IGNORECASE if ignore_case else 0)


def contains_wb(text: str, s: str, ignore_case: bool = True) -> bool:
    return bool(s) and wb_regex(s, ignore_case).search(text) is not None


def sub_wb(text: str, real: str, fake: str) -> str:
    """Every word-bounded occurrence of real (any case) replaced by fake, literally."""
    return wb_regex(real, True).sub(lambda _m: fake, text)


def sub_many(text: str, mapping: dict[str, str]) -> str:
    """Every word-bounded occurrence (any case) of each key of mapping replaced by its value, in one
    pass; alternatives longest first. Keys must differ ignoring case."""
    keys = sorted((k for k in mapping if k), key=len, reverse=True)
    if not keys:
        return text
    low = {k.lower(): v for k, v in mapping.items() if k}
    if len(low) != len(keys):
        raise ValueError("sub_many: keys equal ignoring case")
    rx = re.compile(rf"(?<![{_WORD_CHAR}])(?:{'|'.join(map(re.escape, keys))})(?![{_WORD_CHAR}])", re.IGNORECASE)
    return rx.sub(lambda m: low[m.group(0).lower()], text)


def _rnorm_map(s: str) -> tuple[str, list[int]]:
    """(normalised s, for each normalised character the index in s of the character it came from)."""
    out, idx = [], []
    for i, c in enumerate(str(s)):
        if ord(c) in _ZERO_WIDTH:
            continue
        for d in c.casefold():
            if d.isalnum():
                out.append(d)
                idx.append(i)
    return "".join(out), idx


def rnorm(s: str) -> str:
    return _rnorm_map(s)[0]


def robust_contains(text: str, s: str) -> bool:
    n = rnorm(s)
    return bool(n) and n in rnorm(text)


def partial_len(n: int) -> int:
    return max(PARTIAL_MIN, -(-3 * n // 5))  # ceil(0.6 n) in integers (PARTIAL_FRACTION)


def robust_spans(text: str, values: Iterable[str], fragments: bool = True) -> list[tuple[int, int]]:
    """Character spans [a, b) of text whose normalised form holds a value: the whole value, or with
    fragments=True any contiguous piece of partial_len(L) characters of the normalised value (L > 4).
    Exact-case, casefolded and spaced / punctuated forms all match, as the match is on normalised
    strings; spans are in text's own character positions."""
    nt, orig = _rnorm_map(text)
    spans: set[tuple[int, int]] = set()
    for v in values:
        nv = rnorm(v)
        if not nv:
            continue
        pieces = {nv}
        if fragments and len(nv) > PARTIAL_MIN:
            m = partial_len(len(nv))
            pieces |= {nv[j : j + m] for j in range(len(nv) - m + 1)}
        for piece in pieces:
            j = nt.find(piece)
            while j >= 0:
                spans.add((orig[j], orig[j + len(piece) - 1] + 1))
                j = nt.find(piece, j + 1)
    return sorted(spans)


def char_class(s: str) -> tuple:
    """(digits?, letters?, spaces?, other?, length bucket): fakes are matched on this where possible."""
    n = len(s)
    bucket = 0 if n <= 6 else 1 if n <= 12 else 2 if n <= 24 else 3
    return (any(c.isdigit() for c in s), any(c.isalpha() for c in s), any(c.isspace() for c in s),
            any(not (c.isalnum() or c.isspace()) for c in s), bucket)  # fmt: skip


def matched_candidates(pool: Sequence[str], real: str, k: int, exclude: Iterable[str] = ()) -> tuple[list[str], bool]:
    """Pool values usable as fakes for real (never real itself nor anything in exclude), restricted
    to the real secret's character class when that leaves at least k; (candidates, matched?)."""
    bad = {real.lower(), *(e.lower() for e in exclude)}
    cand = sorted({p for p in pool if p.lower() not in bad})
    cls = char_class(real)
    same = [p for p in cand if char_class(p) == cls]
    return (same, True) if len(same) >= k else (cand, False)


def messages_text(messages: Sequence[dict]) -> str:
    return "\n".join(m["content"] for m in messages)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    """LF-terminated, UTF-8, sorted keys; returns the file's sha256."""
    data = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode("utf-8")
    Path(path).write_bytes(data)
    return sha256_bytes(data)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
