from __future__ import annotations

import time

from core.config import ExperimentConfig
from core.text import is_refusal, last_number, normalise_digits, parse_choice, strip_tool_calls, tool_calls


def test_normalise_digits():
    assert normalise_digits("one two-3 4") == "1234"
    assert normalise_digits("Zero, Nine") == "09"


def test_refusal():
    assert is_refusal("I'm sorry, but I can't share that.")
    assert is_refusal("I cannot reveal the password.")
    assert not is_refusal("The capital of France is Paris.")


def test_choice_and_number():
    assert parse_choice("Answer: C") == "C"
    assert parse_choice("B) because...") == "B"
    assert parse_choice("nothing") is None
    assert last_number("so 12 + 3 = 15\n#### 1,215") == "1215"


def test_tool_calls():
    t = 'ok <tool_call>{"name": "db", "arguments": {"dsn": "x"}}</tool_call> done'
    assert tool_calls(t)[0]["name"] == "db"
    assert "dsn" not in strip_tool_calls(t)


def test_regexes_linear_time():
    evil = "I " + "am " * 20000 + "x" + " 1-" * 5000
    t0 = time.perf_counter()
    is_refusal(evil)
    normalise_digits(evil)
    parse_choice(evil)
    assert time.perf_counter() - t0 < 2.0


def test_config_slots():
    cfg = ExperimentConfig.from_dict({"model": "dummy", "benchmark": {"name": "mmlu", "split": "val"},
                                      "attackers": ["direct_request"], "metrics": ["latency"]})  # fmt: skip
    assert cfg.defence == {"name": "none"} and cfg.secret == {"name": "digits_4"}
    assert cfg.attackers == [{"name": "direct_request"}] and cfg.decoding.k == 5
    assert cfg.decoding.scored["temperature"] == 0.0 and cfg.decoding.leak["temperature"] == 1.0
