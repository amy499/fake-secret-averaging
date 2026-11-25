"""How many tokens each secret code "00".."99" takes inside the system prompt, per tokenizer.

For every base model in configs/training/models.yaml (or --base-model), the system prompt template from
configs/training/prompts.yaml is filled with each code and rendered with the model's chat template; the
tokens overlapping the code's characters are counted. Reported, without interpretation:
  * tokens per code, and the most common count;
  * codes whose count differs from the most common one          -> flag "LEN";
  * codes sharing a token with neighbouring text (token crosses the code boundary) -> flag "EDGE".
Tokenizers that cannot be loaded (e.g. gated Gemma without HF access) are reported as skipped.
Needs transformers (`uv sync --group ml`) but not torch.

Usage (from the repository root):
    uv run --group ml python -m scripts.check_tokenization
    uv run --group ml python -m scripts.check_tokenization --base-model qwen2.5-0.5b-instruct --all
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fsa_runtime.generators.base import GeneratorUnavailable
from fsa_runtime.generators.hf import HFGenerator, load_system_templates, render_chat
from fsa_runtime.registry import load_registry
from fsa_runtime.settings import load_settings
from scripts._cli import utf8_stdio

CODES = [f"{i:02d}" for i in range(100)]
DEFAULT_USER = "Generate 20 random 2-digit numbers, separated by commas."


@dataclass(frozen=True)
class CodeTokens:
    code: str
    n_tokens: int
    tokens: list[str]
    crosses_edge: bool  # a token covering the code also covers text outside it


def code_tokens(tokenizer: Any, template: str, code: str, user: str = DEFAULT_USER) -> CodeTokens:
    """Tokens overlapping `code` where it sits in the rendered chat prompt."""
    if template.count("{secret}") != 1:
        raise ValueError("template must contain exactly one {secret}")
    system = template.format(secret=code)
    text = render_chat(tokenizer, system, user)
    sys_at = text.find(system)
    if sys_at < 0:
        raise ValueError("system prompt not found verbatim in the rendered chat text")
    start = sys_at + len(template.split("{secret}")[0])
    end = start + len(code)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    hits = [(i, s, e) for i, (s, e) in enumerate(offsets) if s < end and e > start and e > s]
    tokens = tokenizer.convert_ids_to_tokens([ids[i] for i, _, _ in hits])
    crosses = any(s < start or e > end for _, s, e in hits)
    return CodeTokens(code, len(hits), list(tokens), crosses)


def analyse(tokenizer: Any, template: str) -> tuple[list[CodeTokens], int]:
    rows = [code_tokens(tokenizer, template, c) for c in CODES]
    mode = Counter(r.n_tokens for r in rows).most_common(1)[0][0]
    return rows, mode


def report(
    model_id: str, template_name: str, rows: list[CodeTokens], mode: int, show_all: bool
) -> None:
    counts = Counter(r.n_tokens for r in rows)
    print(f"\n== {model_id} · system template '{template_name}' ==")
    print(
        "tokens per code: " + ", ".join(f"{n} tokens: {c} codes" for n, c in sorted(counts.items()))
    )
    print(f"most common: {mode}")
    flagged = [r for r in rows if r.n_tokens != mode or r.crosses_edge]
    print(
        f"flagged codes: {len(flagged)} (LEN = count differs from {mode}; EDGE = token crosses code boundary)"
    )
    for r in rows if show_all else flagged:
        flags = " ".join(
            f for f, on in (("LEN", r.n_tokens != mode), ("EDGE", r.crosses_edge)) if on
        )
        print(f"  {r.code}: {r.n_tokens} {r.tokens!r} {flags}".rstrip())


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base-model", action="append", help="registry id (repeatable); default: all"
    )
    parser.add_argument(
        "--template", action="append", help="system template name; default: all with {secret}"
    )
    parser.add_argument("--all", action="store_true", help="list every code, not only flagged ones")
    parser.add_argument("--csv", type=Path, help="also write all rows to this CSV file")
    args = parser.parse_args(argv)

    settings = load_settings()
    registry = load_registry()
    templates = load_system_templates()
    names = args.template or [k for k, v in templates.items() if "{secret}" in v]
    model_ids = args.base_model or list(registry.base_models)
    hf = HFGenerator.from_settings(settings)

    out_rows = []
    for mid in model_ids:
        bm = registry.base_models.get(mid)
        if bm is None:
            print(f"\n== {mid}: SKIPPED (not in configs/training/models.yaml)")
            continue
        try:
            tok = hf.tokenizer(bm.hf_id)
        except GeneratorUnavailable as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        except (OSError, ValueError) as e:  # gated repo (HTTP error), no network, missing files
            print(
                f"\n== {mid} ({bm.hf_id}): SKIPPED, tokenizer not available: {str(e).splitlines()[0]}"
            )
            continue
        for name in names:
            rows, mode = analyse(tok, templates[name])
            report(f"{mid} ({bm.hf_id})", name, rows, mode, args.all)
            out_rows += [
                {
                    "base_model": mid,
                    "template": name,
                    "code": r.code,
                    "n_tokens": r.n_tokens,
                    "tokens": " ".join(r.tokens),
                    "differs_from_mode": r.n_tokens != mode,
                    "crosses_edge": r.crosses_edge,
                }
                for r in rows
            ]
    if args.csv and out_rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nwrote {len(out_rows)} rows to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
