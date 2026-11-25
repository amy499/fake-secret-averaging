"""ConfAIde Tier-4 parser (benchmarks/confaide.py parse_tier4_text): meetings, private / public
strings, the secret remark and the benchmark's blank-line normalisation, on fixtures written here
(fake meetings, no benchmark text)."""

from __future__ import annotations

import random

import pytest

from benchmarks.confaide import FIXTURE, _blank_runs, _header, parse_tier4_text


def two_pass(body: str) -> str:
    """The normalisation as ConfAIde's eval.py describes it: every triple newline, then every double."""
    return body.replace("\n\n\n", "\n").replace("\n\n", "\n")


def test_fixture_meetings():
    m = parse_tier4_text(FIXTURE)
    assert [x["meeting"] for x in m] == [0, 1, 2]
    assert [(x["private"], x["public"]) for x in m] == [("surprise birthday", "March 3rd"),
                                                        ("move to VISTA", "the budget review"),
                                                        ("LME project", "September 15th")]  # fmt: skip
    assert m[0]["secret"] == "we are planning a surprise birthday for Kate. Keep it quiet."
    assert m[2]["secret"] == "do not mention the LME project in front of Kate."
    for x in m:
        assert "\n\n" not in x["text"] and x["text"].endswith("\n") and "<BEGIN>" not in x["text"]


@pytest.mark.parametrize("k", range(0, 13))
def test_blank_runs_match_the_two_pass_rule(k):
    body = "a" + "\n" * k + "b" + "\n" * k
    assert _blank_runs(body) == two_pass(body)


def test_blank_runs_on_random_text():
    rng = random.Random(3)
    for _ in range(500):
        body = "".join(rng.choice(["x", "Alice: y", " ", "\n", "\n\n", "\n\n\n"]) for _ in range(rng.randint(0, 40)))
        assert _blank_runs(body) == two_pass(body)
    assert _blank_runs("a" + "\n" * 7 + "b") == "a\n\nb"  # not a plain collapse


def test_header_needs_exactly_one_comma():
    assert _header("<BEGIN><a secret, a public fact>\n") == ("a secret", "a public fact")
    assert _header("<BEGIN><x,y>") == ("x", "y")
    for bad in ("<BEGIN><only one field>\n", "<BEGIN><a, b, c>\n"):
        with pytest.raises(ValueError, match="comma"):
            _header(bad)


def test_each_meeting_needs_exactly_one_secret_remark():
    none = "<BEGIN><p, q>\nBob: hello\n<END>\n"
    two = "<BEGIN><p, q>\nAlice: This reminds me, one\nAlice: This reminds me, two\n<END>\n"
    for text in (none, two):
        with pytest.raises(ValueError, match="secret remarks"):
            parse_tier4_text(text)
    bob = "<BEGIN><p, q>\nBob: This reminds me, not Alice\nAlice: This reminds me, it\n<END>\n"
    assert parse_tier4_text(bob)[0]["secret"] == "it"  # only Alice's line counts


def test_lines_outside_meetings_are_ignored():
    text = "preamble\n" + FIXTURE + "trailing text\n"
    assert parse_tier4_text(text) == parse_tier4_text(FIXTURE)
