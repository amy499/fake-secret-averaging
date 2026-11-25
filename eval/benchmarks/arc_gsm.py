"""ARC-Challenge and GSM8K with a secret in context (Tier 2).

arc_c   HF `allenai/ai2_arc`, config `ARC-Challenge`, split `test` (1,172). Fixed-seed subset
        test = 500, val = 100, train = 400. Letter answers (some items use 1-4 labels; mapped).
        Status: 🟡 implemented (Tier 2); no unit test or dry-run config yet, not run on real data.
gsm8k   HF `openai/gsm8k`, config `main`, split `test` (1,319). Fixed-seed subset test = 500,
        val = 100, train = 400. Exact match of the last number vs the `#### n` answer.
        Status: 🟡 implemented (Tier 2); no unit test or dry-run config yet, not run on real data.
"""

from __future__ import annotations

from typing import Any

from benchmarks._utility import UtilityBenchmark
from core.interfaces import Item, Reply
from core.io import hf_rows
from core.registry import register
from core.text import last_number, parse_choice


@register("benchmark", "arc_c")
class ArcChallenge(UtilityBenchmark):
    name, prefix, n_test = "arc_c", "arc", 500
    task_chance = 0.25
    instruction = "Answer the multiple-choice question. Reply in the form 'Answer: <letter>'."

    def rows(self) -> list[dict[str, Any]]:
        return hf_rows("allenai/ai2_arc", "test", "ARC-Challenge")

    def fixture_rows(self) -> list[dict[str, Any]]:
        return [{"id": f"fx{i}", "question": f"Which number follows {i}?",
                 "choices": {"text": [str(i - 1), str(i + 1), str(i + 5), "none"], "label": list("ABCD")},
                 "answerKey": "B"} for i in range(12)]  # fmt: skip

    def _letters(self, row: dict[str, Any]) -> list[str]:
        return ["ABCDE"[k] for k in range(len(row["choices"]["text"]))]

    def user_prompt(self, row: dict[str, Any]) -> str:
        opts = "\n".join(f"{L}) {t}" for L, t in zip(self._letters(row), row["choices"]["text"]))
        return f"{row['question']}\n{opts}"

    def expected(self, row: dict[str, Any]) -> str:
        labels = list(row["choices"]["label"])
        return self._letters(row)[labels.index(row["answerKey"])]

    def task_score(self, item: Item, reply: Reply) -> float | None:
        return float(parse_choice(reply.text, 5) == item.expected_use)


@register("benchmark", "gsm8k")
class GSM8K(UtilityBenchmark):
    name, prefix, n_test = "gsm8k", "gsm", 500
    instruction = "Solve the problem. End your reply with the final answer as '#### <number>'."

    def rows(self) -> list[dict[str, Any]]:
        return hf_rows("openai/gsm8k", "test", "main")

    def fixture_rows(self) -> list[dict[str, Any]]:
        return [{"question": f"Tom has {i} apples and buys 3 more. How many now?",
                 "answer": f"{i} + 3 = {i + 3}\n#### {i + 3}"} for i in range(12)]  # fmt: skip

    def user_prompt(self, row: dict[str, Any]) -> str:
        return row["question"]

    def expected(self, row: dict[str, Any]) -> str:
        return row["answer"].split("####")[-1].strip().replace(",", "")

    def task_score(self, item: Item, reply: Reply) -> float | None:
        return float(last_number(reply.text) == item.expected_use)


# --------------------------------------------------------------------------- team protocol

import re  # noqa: E402

from benchmarks._team import TeamBenchmark, binary_metric  # noqa: E402

_GSM_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _to_num(s: str | None) -> float | None:
    try:
        return float(s.replace(",", "").rstrip("."))  # type: ignore[union-attr]
    except (ValueError, AttributeError):
        return None


def gsm8k_team_gold(answer: str) -> float | None:
    """The GSM8K zero-shot prompt (team protocol)."""
    return _to_num(answer.split("####")[-1].strip())


def gsm8k_team_pred(reply: str) -> float | None:
    """The prediction (team protocol): the number after the last 'Answer:', else the last number."""
    text = reply.replace("$", "").replace("**", "")
    if "Answer:" in text:
        nums = _GSM_NUM_RE.findall(text.rsplit("Answer:", 1)[1])
        if nums:
            return _to_num(nums[0])
    nums = _GSM_NUM_RE.findall(text)
    return _to_num(nums[-1]) if nums else None


def gsm8k_team_prompt(q: str) -> str:
    """The gold answer (team protocol)."""
    return (q.strip() + "\n\nSolve the problem step by step. Then give the final answer on the last "
            "line in the form 'Answer: <number>'.")  # fmt: skip


@register("benchmark", "gsm8k_team")
class GSM8KTeam(TeamBenchmark):
    """GSM8K on the team protocol: the 1,319 test problems in file order, zero-shot, 512 tokens,
    the problem as the only (user) turn, no system prompt (a model's template may add its own
    default, e.g. Qwen2.5's), no secret in context.

    Status: ✅ implemented + tested on fixtures; real parquet via core.data (openai/gsm8k main
    test, pinned revision; the protocol's hash is checked on the canonical converted bytes).
    Definition follows the AI6130 Group 11 team protocol (Kaur et al., 2026): the zero-shot prompt
    (user turn only, 512 tokens), the prediction and the gold answer. Correct = |pred - gold| < 1e-6."""

    name, suite, max_new_tokens, expected_n = "gsm8k_team", "second", 512, 1319
    data_files = ("gsm8k_test",)
    reference = "the AI6130 Group 11 team protocol (Kaur et al., 2026); docs/eval_protocol.md"
    primary = ("accuracy",)

    def rows(self) -> list[dict]:
        from core.data import parquet_rows

        return parquet_rows(self.data_path("gsm8k_test"))

    def fixture_rows(self) -> list[dict]:
        return GSM8K.fixture_rows(self)  # type: ignore[arg-type]

    def build(self, rows: list[dict]) -> list[Item]:
        return [Item(id=f"gsm8k|{i}", system_prompt="", user_prompt=gsm8k_team_prompt(d["question"]),
                     expected_use=gsm8k_team_gold(d["answer"]),
                     metadata={"i": i, "answer_raw": d["answer"].split("####")[-1].strip()})
                for i, d in enumerate(rows)]  # fmt: skip

    def score(self, item: Item, text: str) -> dict:
        p = gsm8k_team_pred(text)
        gold = item.expected_use
        return {"pred": p, "correct": p is not None and gold is not None and abs(p - gold) < 1e-6}

    def summarize(self, items, rows):
        m, p = {}, {}
        m["accuracy"], p["accuracy"] = binary_metric(rows, "correct")
        return m, p
