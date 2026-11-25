"""IFEval with a secret in context: HF `google/IFEval`, split `train` (541 prompts, all used as
`test`; there is no train/val, which return []).

Status: ✅ implemented + tested (checker unit tests + dry-run). Attack-study benchmark (secret in
context), not the team protocol's IFEval: that is `ifeval_team` below, scored with Google's
checker. Strict prompt-level accuracy here uses the harness's own re-implementation of the 25
instruction types (now in stats/_ifeval_reimpl.py, where the re-scorer also uses it); see that
module for its deviations from Google's checker.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from benchmarks._utility import SECRET_SYSTEM, UtilityBenchmark
from core.interfaces import Item, Reply
from core.io import hf_rows
from core.registry import register


from stats._ifeval_reimpl import CHECKS, check_all  # noqa: E402,F401  (moved; kept importable here)


@register("benchmark", "ifeval")
class IFEval(UtilityBenchmark):
    name, prefix = "ifeval", "ife"

    def rows(self) -> list[dict[str, Any]]:
        return hf_rows("google/IFEval", "train")

    def fixture_rows(self) -> list[dict[str, Any]]:
        return [
            {"key": 1, "prompt": "Write a poem about the sea without using any commas.",
             "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]},
            {"key": 2, "prompt": "Answer in all lowercase letters: what is rain?",
             "instruction_id_list": ["change_case:english_lowercase"], "kwargs": [{}]},
            {"key": 3, "prompt": "Write at least 5 words about cats and end with 'Is there anything else?'",
             "instruction_id_list": ["length_constraints:number_words", "startend:end_checker"],
             "kwargs": [{"num_words": 5, "relation": "at least"}, {"end_phrase": "Is there anything else?"}]},
            {"key": 4, "prompt": "Give a title wrapped in double angular brackets, like <<title>>.",
             "instruction_id_list": ["detectable_format:title"], "kwargs": [{}]},
        ] * 3  # fmt: skip

    def _items(self, rows: list[dict[str, Any]], split: str) -> list[Item]:
        if split != "test":
            return []
        items = []
        for j, row in enumerate(rows):
            si, secret = self.secret_for("test", j)
            items.append(Item(
                id=f"ife{j:04d}_{row['key']}", system_prompt=SECRET_SYSTEM.format(secret=secret),
                user_prompt=row["prompt"], secret=secret, needs_secret=False,
                expected_use={"ids": list(row["instruction_id_list"]), "kwargs": list(row["kwargs"])},
                metadata={"secret_index": si, "key": row["key"]},
            ))  # fmt: skip
        return items

    def task_score(self, item: Item, reply: Reply) -> float | None:
        res = check_all(reply.text, item.expected_use["ids"], item.expected_use["kwargs"], item.user_prompt or "")
        if any(r is None for r in res):
            return None
        return float(all(res))


# --------------------------------------------------------------------------- team protocol

import numpy as np  # noqa: E402

from benchmarks._team import TEAM_SYSTEM, TeamBenchmark, binary_metric  # noqa: E402

IFEVAL_TEAM_SEED = 2061  # the team protocol's seed; run order = default_rng(seed + 1)


@register("benchmark", "ifeval_team")
class IFEvalTeam(TeamBenchmark):
    """IFEval on the team protocol: all 541 prompts as the user message, unchanged; system "You
    are a helpful assistant."; 512 tokens; no secret in context. Scored with Google's checker
    (benchmarks/ifeval_google.py: vendored google-research instruction_following_eval at the
    pinned commit), strict and loose, prompt level (primary) and instruction level.

    Status: ✅ implemented + tested on fixtures; the real input file via core.data
    (google-research input_data.jsonl, pinned commit, md5 checked against the team protocol's pin).
    Definition follows the AI6130 Group 11 team protocol (Kaur et al., 2026): items, system prompt,
    run order (default_rng(2061 + 1).permutation) and Google's checker, strict and loose."""

    name, suite, max_new_tokens, expected_n = "ifeval_team", "core", 512, 541
    data_files = ("ifeval_input",)
    reference = "the AI6130 Group 11 team protocol (Kaur et al., 2026); docs/eval_protocol.md"
    primary = ("prompt_strict", "prompt_loose")
    scorer_limits = (
        "language:response_language uses langdetect (seeded with DetectorFactory.seed = 0 so it "
        "is repeatable); short or mixed-language replies can still be misdetected.",
        "About half of the replies reach the 512-token cap ; "
        "a truncated reply is scored as written.",
    )

    def rows(self) -> list[dict]:
        from core.data import jsonl_rows

        rows = jsonl_rows(self.data_path("ifeval_input"))
        if len(rows) != 541:
            raise ValueError(f"IFEval: {len(rows)} prompts, 541 expected")
        return rows

    def fixture_rows(self) -> list[dict]:
        return IFEval.fixture_rows(self)  # type: ignore[arg-type]

    def build(self, rows: list[dict]) -> list[Item]:
        return [Item(id=f"ifeval|{i}", system_prompt=TEAM_SYSTEM, user_prompt=d["prompt"],
                     expected_use={"ids": list(d["instruction_id_list"]), "kwargs": list(d["kwargs"])},
                     metadata={"item": i, "key": d["key"]})
                for i, d in enumerate(rows)]  # fmt: skip

    def run_order(self, items):
        seed = int(self.cfg.get("seed", IFEVAL_TEAM_SEED))
        return [int(k) for k in np.random.default_rng(seed + 1).permutation(len(items))]

    def score(self, item: Item, text: str) -> dict:
        from benchmarks import _ifeval_google as G
        from core.data import DataHashError

        try:
            return G.score(item.metadata["key"], item.expected_use["ids"], item.expected_use["kwargs"],
                           item.user_prompt or "", text, download=not self.dry_run)  # fmt: skip
        except (ImportError, LookupError, DataHashError) as e:  # checker missing: no score, never a made-up one
            return {"checker": f"unavailable: {type(e).__name__}: {e}"}

    def summarize(self, items, rows):
        from stats.core import estimate_clustered

        m, p = {}, {}
        for mode in ("strict", "loose"):
            m[f"prompt_{mode}"], p[f"prompt_{mode}"] = binary_metric(rows, f"prompt_{mode}")
            vals, clusters = [], []
            for r in rows:
                for v in r["scores"].get(f"instructions_{mode}") or []:
                    vals.append(float(v))
                    clusters.append(r["key"])
            est = estimate_clustered(vals, clusters, note="instruction level; CI resamples prompts")
            est["unit"] = "instruction"
            m[f"instruction_{mode}"] = est
        missing = [r["scores"]["checker"] for r in rows if "checker" in r["scores"]]
        if missing:
            m["checker_unavailable"] = {"value": None, "n": len(missing), "note": missing[0]}
        return m, p

    def preflight(self) -> None:
        """Load Google's checker (downloading punkt_tab if needed) and score one reply; raise
        RuntimeError if anything is missing. Dry runs and fixture runs do not call this."""
        from benchmarks import _ifeval_google as G
        from core.data import DataHashError

        try:
            G._load(download=True)
            G.score("preflight", ["punctuation:no_comma"], [{}], "p", "no commas here")
        except (ImportError, LookupError, DataHashError, OSError) as e:
            raise RuntimeError(
                f"IFEval checker unavailable ({type(e).__name__}: {e}). Install requirements.txt "
                "(nltk==3.9.1 langdetect==1.0.9 absl-py==2.1.0 immutabledict==4.2.1) and fetch the "
                "sentence data: python -m core.data fetch nltk_punkt_tab") from e

    def checker_info(self) -> dict:
        from benchmarks import _ifeval_google as G
        from core.data import DataHashError

        try:
            G._load(download=not self.dry_run)
            available, error = True, None
        except (ImportError, LookupError, DataHashError) as e:
            available, error = False, f"{type(e).__name__}: {e}"
        return {"available": available, "error": error, **G.info()}
