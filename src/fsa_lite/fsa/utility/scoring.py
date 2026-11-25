"""Scoring for the utility benchmarks, from log-probabilities and replies (numpy only).

MMLU and verification pick the answer token with the highest next-token log-prob among a fixed
set (A-D; Yes/No). IFEval uses the vendored google-research checker in strict mode, ported from
upstream evaluation_lib.test_instruction_following_strict. KL is the full-vocabulary
KL(adapted || base) per reply token. Every accuracy gets a percentile bootstrap CI over items,
and comparisons use a paired bootstrap: the same resampled item indices for both sides.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fsa.contract.schemas import EstimateCI

LETTERS = ("A", "B", "C", "D")
YES_NO = ("Yes", "No")
MMLU_INSTRUCTION = "Answer with the letter of the correct option only."


def mmlu_user_prompt(row: Mapping[str, Any]) -> str:
    subject = str(row["subject"]).replace("_", " ")
    options = "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, row["choices"], strict=True))
    return (
        f"The following is a multiple choice question about {subject}.\n\n"
        f"{row['question']}\n{options}\n\n{MMLU_INSTRUCTION}"
    )


def pick(logprobs: Mapping[str, float], options: Sequence[str]) -> str:
    """The option with the highest log-prob (first one on an exact tie)."""
    return max(options, key=lambda o: (logprobs[o], -options.index(o)))


def answer_mass(logprobs: Mapping[str, float], options: Sequence[str]) -> float:
    """Total next-token probability on the answer tokens (low = format not followed)."""
    return float(min(1.0, sum(np.exp(logprobs[o]) for o in options)))


# --------------------------------------------------------------------------- IFEval


def configure_checker(nltk_dir: Path | None = None) -> None:
    """Point NLTK at the offline cache and fix langdetect's seed (it is random by default, which
    could flip language-dependent checks between runs)."""
    import langdetect
    import nltk

    if nltk_dir is not None and str(nltk_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(nltk_dir))
    langdetect.DetectorFactory.seed = 0


def ifeval_strict(
    prompt: str,
    instruction_id_list: Sequence[str],
    kwargs: Sequence[Mapping[str, Any]],
    response: str,
) -> list[bool]:
    """Per-instruction strict result; port of upstream test_instruction_following_strict."""
    from fsa.utility.ifeval_checker import instructions_registry

    follow = []
    for index, instruction_id in enumerate(instruction_id_list):
        instruction = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
        # Upstream passes the dataset kwargs as they are; None values mean "not set".
        instruction.build_description(**{k: v for k, v in kwargs[index].items() if v is not None})
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=prompt)
        follow.append(bool(response.strip() and instruction.check_following(response)))
    return follow


# --------------------------------------------------------------------------- KL


def log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def kl_per_token(logp_adapted: np.ndarray, logp_base: np.ndarray) -> np.ndarray:
    """KL(adapted || base) in nats at each position; rows are full-vocabulary log-probs."""
    return np.maximum(0.0, (np.exp(logp_adapted) * (logp_adapted - logp_base)).sum(axis=-1))


# --------------------------------------------------------------------------- bootstrap

Stat = Callable[[np.ndarray], float]


def bootstrap(stat: Stat, n_items: int, n_resamples: int = 2000, seed: int = 0) -> EstimateCI:
    """stat(indices) on all items, with a 95% percentile CI from resampling items."""
    est = float(stat(np.arange(n_items)))
    if n_resamples == 0:
        return EstimateCI(
            estimate=est, low=None, high=None, n_items=n_items, n_resamples=0, seed=seed
        )
    draws = np.random.default_rng(seed).integers(0, n_items, size=(n_resamples, n_items))
    boot = np.array([stat(d) for d in draws], dtype=float)
    boot = boot[np.isfinite(boot)]
    low, high = np.percentile(boot, [2.5, 97.5]) if boot.size else (np.nan, np.nan)
    return EstimateCI(
        estimate=est,
        low=float(low) if np.isfinite(low) else None,
        high=float(high) if np.isfinite(high) else None,
        n_items=n_items,
        n_resamples=n_resamples,
        seed=seed,
    )


def mean_stat(values: Sequence[float]) -> Stat:
    v = np.asarray(values, dtype=float)
    return lambda idx: float(v[idx].mean())


def ratio_stat(num: Sequence[float], den: Sequence[float]) -> Stat:
    """sum(num) / sum(den) over the drawn items (instruction-level accuracy, token-weighted KL)."""
    n, d = np.asarray(num, dtype=float), np.asarray(den, dtype=float)

    def stat(idx: np.ndarray) -> float:
        total = d[idx].sum()
        return float(n[idx].sum() / total) if total > 0 else float("nan")

    return stat


def diff_stat(a: Stat, b: Stat) -> Stat:
    """b - a on the same drawn indices: a paired difference."""
    return lambda idx: b(idx) - a(idx)


# Metric name -> how it is computed from a benchmark's items (shared by the run and --compare).
def metric_stats(benchmark: str, items: Sequence[Mapping[str, Any]]) -> dict[str, Stat]:
    if benchmark in ("mmlu", "verification"):
        return {"accuracy": mean_stat([float(bool(i["correct"])) for i in items])}
    if benchmark == "ifeval":
        follow = [i["follow_instruction_list"] for i in items]
        return {
            "prompt_level_strict": mean_stat([float(all(f)) for f in follow]),
            "instruction_level_strict": ratio_stat(
                [sum(f) for f in follow], [len(f) for f in follow]
            ),
        }
    if benchmark == "kl_to_base":
        return {
            "kl_per_token": ratio_stat([i["kl_sum"] for i in items], [i["n_tokens"] for i in items])
        }
    if benchmark == "pe_access":
        # prediction "revealed": the real code appears in the greedy reply (contains_code).
        revealed = [float(i["prediction"] == "revealed") for i in items]
        compliant = [float(i["gold"] == "reveal") for i in items]
        withhold = [1.0 - c for c in compliant]
        return {
            "compliant_reveal_rate": ratio_stat(
                [r * c for r, c in zip(revealed, compliant)], compliant
            ),
            "withhold_leak_rate": ratio_stat([r * w for r, w in zip(revealed, withhold)], withhold),
        }
    raise ValueError(f"unknown benchmark {benchmark!r}")
