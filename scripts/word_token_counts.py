"""How many tokens each candidate word takes in the rendered system prompt (tokenizer only).

For choosing the 100 word secrets: FSA's mixture target needs every word to take the same
number of tokens in the system prompt (training and evaluation refuse a list that does not).
Reads a candidate file (one word per line; blank lines and lines starting with # are skipped;
any length), fills the word_system_templates template of --method with each word, renders it
with the model's chat template and counts the word's tokens: the tokens not shared by every
candidate's prompt (fsa.prompting.secret_span_tokens). Prints one line per word, a summary,
and the single-token words. Entries that are not one lowercase a-z word, and duplicates, are
reported and skipped.

Loads only the tokenizer (transformers, no torch, no model), so it runs on a the Slurm cluster login node.
The Gemma tokenizer is gated: the HF token comes from .env (FSA_HF_TOKEN) as for generation.

Usage (from the repository root):
    uv run --group ml python -m scripts.word_token_counts --words candidates.txt
    uv run --group ml python -m scripts.word_token_counts --words candidates.txt \\
        --base-model gemma3-12b-it --single-token-only --out single_token_words.txt
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from fsa_runtime.generators.base import GeneratorUnavailable
from fsa_runtime.generators.hf import HFGenerator
from fsa_runtime.registry import load_registry
from fsa_runtime.settings import load_settings
from fsa.prompting import load_system_templates, secret_span_tokens
from fsa.secrets import SECRET_SPACE, WORD_PATTERN
from scripts._cli import utf8_stdio


def read_candidates(text: str) -> tuple[list[str], list[str]]:
    """(valid distinct words in file order, problems) from a candidate file's text."""
    words, problems, seen = [], [], set()
    for n, line in enumerate(text.splitlines(), 1):
        w = line.strip()
        if not w or w.startswith("#"):
            continue
        if not WORD_PATTERN.fullmatch(w):
            problems.append(f"line {n}: {w!r} is not one lowercase a-z word (skipped)")
        elif w in seen:
            problems.append(f"line {n}: duplicate {w!r} (skipped)")
        else:
            seen.add(w)
            words.append(w)
    return words, problems


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--words", type=Path, required=True, help="candidate file, one per line")
    parser.add_argument(
        "--base-model", default="gemma3-12b-it", help="registry id or HF id (default gemma3-12b-it)"
    )
    parser.add_argument(
        "--method", default="base", help="word_system_templates key to render (default base)"
    )
    parser.add_argument(
        "--single-token-only", action="store_true", help="list only words that take 1 token"
    )
    parser.add_argument("--out", type=Path, default=None, help="also write the listed words here")
    args = parser.parse_args(argv)

    try:
        words, problems = read_candidates(args.words.read_text(encoding="utf-8"))
        for p in problems:
            print(f"note: {p}")
        if len(words) < 2:
            raise ValueError("need at least two valid candidate words")
        bm = load_registry().base_models.get(args.base_model)
        hf_id = bm.hf_id if bm else args.base_model
        templates = load_system_templates(secret_type="words")
        if args.method not in templates:
            raise ValueError(f"unknown --method {args.method!r}; known: {sorted(templates)}")
        tokenizer = HFGenerator.from_settings(load_settings()).tokenizer(hf_id)
        counts = secret_span_tokens(tokenizer, templates, args.method, words)
    except (OSError, ValueError, GeneratorUnavailable) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"tokenizer: {hf_id}; template: word_system_templates.{args.method}")
    print(f"{'tokens':>6}  word")
    for w in sorted(words, key=lambda w: (counts[w], w)):
        print(f"{counts[w]:>6}  {w}")
    hist = Counter(counts.values())
    print("summary: " + ", ".join(f"{n} token(s): {c}" for n, c in sorted(hist.items())))
    single = [w for w in words if counts[w] == 1]
    print(f"single-token words: {len(single)} (a word list needs exactly {SECRET_SPACE})")
    listed = single if args.single_token_only else sorted(words, key=lambda w: (counts[w], w))
    if args.out:
        args.out.write_text("".join(w + "\n" for w in listed), encoding="utf-8", newline="\n")
        print(f"wrote {len(listed)} words to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
