"""MMLU with a secret in context: HF `cais/mmlu`, config `all`, split `test` (14,042 questions).

Status: ✅ implemented + tested (unit + dry-run). Fixed-seed subset (default_rng(2026)):
test = 1,000 questions, val = 100, train = 400 (disjoint). Zero-shot, answer letter parsed with
core.text.parse_choice. Chance 25%.
"""

from __future__ import annotations

from typing import Any

from benchmarks._utility import UtilityBenchmark
from core.interfaces import Item, Reply
from core.io import hf_rows
from core.registry import register
from core.text import parse_choice

LETTERS = "ABCD"


@register("benchmark", "mmlu")
class MMLU(UtilityBenchmark):
    name, prefix = "mmlu", "mmlu"
    task_chance = 0.25
    instruction = "Answer the multiple-choice question. Reply in the form 'Answer: <letter>'."

    def rows(self) -> list[dict[str, Any]]:
        return hf_rows("cais/mmlu", "test", "all")

    def fixture_rows(self) -> list[dict[str, Any]]:
        return [{"question": f"Fixture question {i}: what is {i} + 1?", "subject": "arithmetic",
                 "choices": [str(i), str(i + 1), str(i + 2), str(i + 3)], "answer": 1} for i in range(12)]  # fmt: skip

    def user_prompt(self, row: dict[str, Any]) -> str:
        opts = "\n".join(f"{LETTERS[k]}) {c}" for k, c in enumerate(row["choices"]))
        return f"{row['question']}\n{opts}"

    def expected(self, row: dict[str, Any]) -> str:
        return LETTERS[int(row["answer"])]

    def row_meta(self, row: dict[str, Any]) -> dict[str, Any]:
        return {"subject": row.get("subject")}

    def task_score(self, item: Item, reply: Reply) -> float | None:
        return float(parse_choice(reply.text, 4) == item.expected_use)


# --------------------------------------------------------------------------- team protocol

import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from benchmarks._team import TEAM_SYSTEM, TeamBenchmark, binary_metric  # noqa: E402

TEAM_SEED = 2061
TEAM_N_PER_SUBJECT = 60
# the answer-letter rules of the team protocol)
LEAD = re.compile(r"^[\s*_#>\"'`(\[]*([ABCD])(?=$|[.):\]*,]|\s*\n|\s+$)")
SAID = re.compile(r"(?i:answer)\s*(?i:is)?\s*:?\s*[*_(\[]*\s*([ABCD])(?![A-Za-z])")


def team_letter(text: str) -> str | None:
    """The team protocol's answer rule: a leading A-D, else "answer (is) X"; None = wrong."""
    m = LEAD.match(text) or SAID.search(text)
    return m.group(1) if m else None


def team_prompt(subject: str, question: str, choices: list) -> str:
    """The team protocol's MMLU prompt."""
    return ("The following is a multiple choice question about %s. Reply with the "
            "letter of the correct answer (A, B, C or D) first.\n\n%s\n%s\nAnswer:" % (
                subject.replace("_", " "), question.strip(),
                "\n".join("%s. %s" % (LETTERS[k], str(c).strip()) for k, c in enumerate(choices))))  # fmt: skip


def team_draw(rows: list[dict], n_per_subject: int = TEAM_N_PER_SUBJECT, seed: int = TEAM_SEED) -> list[dict]:
    """The team protocol's MMLU draw without pandas: one default_rng(seed) for all
    subjects; subjects in sorted order; per subject rng.choice(len(pool), min(n, len(pool)),
    replace=False) over the subject's rows in file order, picks sorted; `row` = file row index."""
    rng = np.random.default_rng(seed)
    by_subject: dict[str, list[int]] = {}
    for idx, r in enumerate(rows):
        by_subject.setdefault(r["subject"], []).append(idx)
    items = []
    for subject in sorted(by_subject):
        sub = by_subject[subject]
        pool = list(range(len(sub)))
        pick = rng.choice(len(pool), size=min(n_per_subject, len(pool)), replace=False)
        for i in np.sort([pool[int(k)] for k in pick]):
            row = rows[sub[int(i)]]
            choices = list(row["choices"])
            if len(choices) != 4:
                raise ValueError(f"MMLU {subject} row {sub[int(i)]}: {len(choices)} choices")
            items.append({"item": len(items), "subject": subject, "row": sub[int(i)],
                          "gold": LETTERS[int(row["answer"])], "prompt": team_prompt(subject, row["question"], choices)})  # fmt: skip
    return items


def selection_sha256(items: list[dict]) -> str:
    """sha256 of "subject<TAB>row" lines (LF, final LF): two runs compare this to check that
    their 3,420-item subset is the same."""
    text = "".join(f"{it['subject']}\t{it['row']}\n" for it in items)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@register("benchmark", "mmlu_team")
class MMLUTeam(TeamBenchmark):
    """MMLU on the team protocol: 60 questions per subject, seed 2061 (3,420 items), 48 tokens,
    zero-shot, system "You are a helpful assistant.", no secret in context.

    Status: ✅ implemented + tested on fixtures and on a synthetic 57-subject table against an
    independent copy of the reference loop (tests/test_team_benchmarks.py); real parquet via
    core.data (cais/mmlu all/test, pinned revision, 14,042 rows).
    Definition follows the AI6130 Group 11 team protocol (Kaur et al., 2026): the draw (60 per
    subject, seed 2061), the prompt, the run order (default_rng(2061 + 1).permutation; blocks of 480 =
    30 batches of 16, so batches are consecutive chunks of the permutation) and the letter rule
    (none = wrong).
    Writes mmlu_selection.json (subject/row of every item and its sha256)."""

    name, suite, max_new_tokens, expected_n = "mmlu_team", "core", 48, 3420
    data_files = ("mmlu_test",)
    reference = "the AI6130 Group 11 team protocol (Kaur et al., 2026); docs/eval_protocol.md"
    primary = ("accuracy",)
    task_chance = 0.25

    def rows(self) -> list[dict]:
        from core.data import parquet_rows

        rows = parquet_rows(self.data_path("mmlu_test"))
        if len(rows) != 14042:
            raise ValueError(f"MMLU test: {len(rows)} rows, 14042 expected")
        return rows

    def fixture_rows(self) -> list[dict]:
        return [{"question": f"Fixture {s} question {i}: what is {i} + 1?", "subject": s,
                 "choices": [str(i), str(i + 1), str(i + 2), str(i + 3)], "answer": 1}
                for s in ("astronomy", "college_biology", "abstract_algebra") for i in range(4)]  # fmt: skip

    def build(self, rows: list[dict]) -> list[Item]:
        n = int(self.cfg.get("n_per_subject", TEAM_N_PER_SUBJECT))
        drawn = team_draw(rows, n, int(self.cfg.get("seed", TEAM_SEED)))
        self.selection = drawn
        return [Item(id=f"mmlu|{d['item']}", system_prompt=TEAM_SYSTEM, user_prompt=d["prompt"],
                     expected_use=d["gold"], metadata={k: d[k] for k in ("item", "subject", "row", "gold")})
                for d in drawn]  # fmt: skip

    def run_order(self, items):
        return [int(k) for k in np.random.default_rng(int(self.cfg.get("seed", TEAM_SEED)) + 1).permutation(len(items))]

    def score(self, item: Item, text: str) -> dict:
        pred = team_letter(text)
        return {"pred": pred, "correct": pred == item.expected_use}

    def summarize(self, items, rows):
        m, p = {}, {}
        m["accuracy"], p["accuracy"] = binary_metric(rows, "correct")
        m["accuracy"]["chance"] = 0.25
        m["no_letter"], p["no_letter"] = binary_metric(
            [{**r, "scores": {"none": r["scores"]["pred"] is None}} for r in rows], "none")
        return m, p

    def write_extras(self, out_dir: Path, items, rows) -> dict:
        sel = [{"subject": it.metadata["subject"], "row": it.metadata["row"]} for it in items]
        sha = selection_sha256(sel)
        (out_dir / "mmlu_selection.json").write_text(
            json.dumps({"n": len(sel), "seed": int(self.cfg.get("seed", TEAM_SEED)), "sha256": sha, "items": sel},
                       indent=1) + "\n", encoding="utf-8", newline="\n")  # fmt: skip
        return {"mmlu_selection_sha256": sha}
