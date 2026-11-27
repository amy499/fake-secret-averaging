"""Parser speed regressions. A model reply of the shape built by fake_replies() (a long comma list
of 2-digit numbers, then prose) once hung parse_numbers through catastrophic regex backtracking;
every case here must finish fast. All inputs are generated here: no model output is stored.

Each parse runs in a subprocess with a hard timeout. A watchdog *thread* would not work: the
regex engine holds the GIL during a match, so a hung match blocks the timeout as well."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from fsa.eval.parsing import MAX_LINE_CHARS

ROOT = Path(__file__).resolve().parents[1]
# The child is a fresh interpreter: pytest's `pythonpath` setting (pyproject.toml: ".", "src/fsa_lite")
# applies to this process only, and the project is not installed into the venv (no build system), so
# the child gets the same import path explicitly, ahead of any PYTHONPATH already set.
_CHILD_ENV = {
    **os.environ,
    "PYTHONPATH": os.pathsep.join(
        [str(ROOT), str(ROOT / "src" / "fsa_lite"), *filter(None, [os.environ.get("PYTHONPATH")])]
    ),
}
_CHILD = (
    "import json, sys, time\n"
    "from fsa.eval.parsing import parse_numbers\n"
    "text = sys.stdin.read()\n"
    "t0 = time.perf_counter()\n"
    "numbers = parse_numbers(text)\n"
    "print(json.dumps({'numbers': numbers, 'elapsed': time.perf_counter() - t0}))\n"
)


def _parse_with_deadline(text, seconds=1.0):
    """(numbers, seconds spent parsing); fails if parsing has not finished within `seconds`."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD],
            input=text.encode("utf-8"),
            capture_output=True,
            cwd=ROOT,
            env=_CHILD_ENV,
            timeout=seconds + 5,  # + interpreter start-up; the parse itself is checked below
            check=True,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"parse_numbers still running after {seconds}s (regex backtracking?)")
    out = json.loads(proc.stdout)
    assert out["elapsed"] < seconds
    return out["numbers"], out["elapsed"]


def _comma_oracle(line):
    """Independent reference for a single comma list: split, keep exact 2-digit tokens."""
    head = line.split("(")[0]
    return [
        int(t) for t in (p.strip(" .") for p in head.split(",")) if re.fullmatch(r"[0-9]{2}", t)
    ]


def fake_replies(n: int = 8, seed: int = 1) -> list[str]:
    """Generated stand-ins for the reply that used to hang: seeded 2-digit numbers, comma
    separated, a parenthesised remark and trailing prose (obviously fake, no secret)."""
    import random

    rng = random.Random(seed)
    out = []
    for i in range(n):
        nums = ", ".join(f"{rng.randrange(100):02d}" for _ in range(rng.randrange(20, 400)))
        out.append(f"{nums} (these are example numbers, set {i}). Here is some prose after the list!")
    return out


def test_generated_reply_shape_that_used_to_hang():
    total = 0.0
    for reply in fake_replies():
        numbers, elapsed = _parse_with_deadline(reply, seconds=1.0)
        total += elapsed
        assert numbers == _comma_oracle(reply)
    assert total < 0.1


ADVERSARIAL = {
    # the shape that hung the old number-line regex: long digit groups, prose at the end
    "digit_groups_then_prose": " ".join(["1010", "2020", "3030", "4040"] * 190)
    + " (5 more numbers)",
    "one_long_digit_run_then_prose": "7" * 3000 + " x",
    "comma_list_then_prose": ", ".join(["12", "345", "6789"] * 250) + " and that is all!",
    "spaces_between_digits": "1 " * 1900 + "end",
    "list_item_long_content": "1. " + ", ".join(["12"] * 900) + " ok?",
}


@pytest.mark.parametrize("name", ADVERSARIAL)
def test_adversarial_lines_are_fast(name):
    line = ADVERSARIAL[name]
    assert len(line) <= MAX_LINE_CHARS  # under the skip limit, so they are really parsed
    _, elapsed = _parse_with_deadline(line, seconds=1.0)
    assert elapsed < 0.5


def test_very_long_digit_comma_line_is_parsed_or_skipped_quickly():
    long_ok = ", ".join(["47"] * 1300)  # ~5.2k chars: over the limit -> skipped
    numbers, elapsed = _parse_with_deadline(long_ok, seconds=1.0)
    assert len(long_ok) > MAX_LINE_CHARS and numbers == [] and elapsed < 0.1
    under = ", ".join(["47"] * 990)  # ~3.9k chars: parsed in full
    numbers, elapsed = _parse_with_deadline(under, seconds=1.0)
    assert numbers == [47] * 990 and elapsed < 0.5
    huge = "1, " * 200_000 + "\n12, 34, 56"  # 600k-char line skipped, the normal line kept
    numbers, elapsed = _parse_with_deadline(huge, seconds=1.0)
    assert numbers == [12, 34, 56] and elapsed < 0.5
