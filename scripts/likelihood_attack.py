"""Likelihood (Bayes) attacker on a finished run: score all 100 candidate secrets by the
log-likelihood of each held-out secret's replies (fsa.attack.likelihood.score_candidates),
posterior = softmax(scores), then the usual metrics plus all significance tests.

Writes likelihood_attack.json into the run folder; if that exists, it writes
likelihood_attack_2.json, _3, ... instead. It never overwrites or modifies any existing file.
The scorer is the run's own model by default. --scorer-model gives a surrogate: a registry id,
an HF id or a local path. Replies are scored from the token IDs saved at generation time when
the scorer is the generating model. Otherwise (older runs, or a surrogate) they are
re-tokenised from the text, and the round-trips are counted. Work is split into
(candidates x replies) blocks under --max-batch-tokens; by default this is sized from GPU
memory, with 2048 on CPU.

Every result also reports:
- the sampling settings used at generation (temperature, top-k, top-p, repetition penalty,
  padding) next to the scorer's, and whether they match;
- the rank of the true secret: a histogram, plus the mean log2 p(true) when it is ranked first
  vs not.
--calibrate also scores the TRAIN secrets (train_samples.jsonl), the same way. It fits one
temperature tau there and reports held-out metrics under softmax(scores / tau) next to the raw
ones. That roughly doubles the GPU time. --from-scores <likelihood_attack*.json> reuses
earlier held-out scores. Without --calibrate, or when that file already holds train scores,
it needs no model or GPU.

Prompts: every reply is scored under the prompt text recorded with it in samples.jsonl, for
every candidate (score_split groups each secret's replies by prompt text). Runs made with a
prompt set (config.yaml eval.prompt_set) are checked first: each reply's prompt must match the
set's recorded text, and every held-out secret, and with --calibrate every train secret, must
have exactly replies_per_prompt replies per prompt, the same assignment as generation.

Word secrets (config.yaml eval.secret_space): the candidates are the run's 100 recorded words
(their hash is checked), scored under word_system_templates; before scoring, every word must
take the same number of tokens in the rendered system prompt. The count-based diagnostics use
word mentions (the exact mention ratio). Digit runs and older runs are unchanged.

digits4 runs (eval.secret_space type digits4): the candidates are the run's 100 recorded
4-digit codes (hash checked; the attacker knows the code is one of them), each scored under
digits4_system_templates with its own password (the pe_base prompt). Count diagnostics use
occurrences of each candidate code among the replies' parsed 4-digit numbers.

Usage (from the repository root):
    uv run --group ml python -m scripts.likelihood_attack outputs/runs/<run_id> --device cuda
    uv run --group ml python -m scripts.likelihood_attack outputs/runs/<run_id> --calibrate
    uv run python -m scripts.likelihood_attack outputs/runs/<run_id> \
        --from-scores outputs/runs/<run_id>/likelihood_attack.json
"""

from __future__ import annotations

import argparse
import contextlib
import math
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fsa_runtime.generators.base import GeneratorUnavailable
from fsa.attack import likelihood
from fsa.attack.batching import plan_blocks, suggest_max_batch_tokens
from fsa.attack.calibration import (
    first_vs_not,
    fit_temperature,
    log2_p_true,
    rank_histogram,
    tempered_posteriors,
    true_ranks,
)
from fsa.attack.tokens import retokenize, roundtrips, turn_end_id
from fsa.contract.schemas import (
    Calibration,
    LeakageMetrics,
    LikelihoodAttack,
    RankSummary,
    RunMetrics,
    Sample,
    SamplingCheck,
    SamplingSettings,
    ScorerInfo,
    TokenizationReport,
    threat_model_of,
)
from fsa.eval.diagnostics import mention_diagnostics, secret_diagnostics
from fsa.eval.leakage import leakage_metrics
from fsa.eval.parsing import code_counts, digit_counts, number_counts, word_mentions
from fsa.eval.significance import format_ci, format_significance, significance_metrics
from fsa.io.run_writer import CONFIG_FILE, METRICS_FILE, SAMPLES_FILE, TRAIN_SAMPLES_FILE
from fsa.prompting import load_system_templates, make_prompt_builder, secret_span_tokens
from fsa.secrets import SecretSpace, check_equal_token_spans, space_from_record, token_report
from scripts._cli import utf8_stdio

ATTACK_FILE = "likelihood_attack.json"
ATTACKER_NAME = "likelihood-bayes"
CPU_MAX_BATCH_TOKENS = 2048
# What score_candidates computes: the full softmax of logits / T at every reply position, with
# replies right-padded after an unpadded prompt (pads come after the real tokens, so a real
# token never attends to them and positions match generation).
SCORER_SAMPLING = {"top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0}
SCORER_PADDING = "right (replies only; never attended by real tokens)"
SCORER_PROMPT_PADDING = "none (one unpadded prompt per candidate)"


class AttackError(ValueError):
    pass


def flush_print(message: str) -> None:
    print(message, flush=True)


def output_path(run_dir: Path, threat_model: str = "grey_box") -> Path:
    """likelihood_attack.json, or the first free likelihood_attack_<n>.json. Public-weights
    results get their own series (likelihood_attack_public_weights[_<n>].json) so they can
    never be mistaken for the grey-box result."""
    stem = Path(ATTACK_FILE).stem + ("_public_weights" if threat_model == "public_weights" else "")
    path, n = run_dir / f"{stem}.json", 1
    while path.exists():
        n += 1
        path = run_dir / f"{stem}_{n}.json"
    return path


THREAT_LABELS = {
    "grey_box": "grey_box (the deployed model: base + the run's adapter, if it has one)",
    "public_weights": "PUBLIC_WEIGHTS (base model only; the run's adapter was NOT used)",
    "surrogate": "surrogate (a different model from the one that generated the replies)",
}


def _read_samples(path: Path) -> dict[str, list[Sample]]:
    samples: dict[str, list[Sample]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                s = Sample.model_validate_json(line)
                samples[s.secret].append(s)
    return samples


def load_run(run_dir: Path) -> tuple[dict, RunMetrics, list[str], dict[str, list[Sample]]]:
    config = yaml.safe_load((run_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    metrics = RunMetrics.model_validate_json((run_dir / METRICS_FILE).read_text(encoding="utf-8"))
    if metrics.run_type != "evaluation" or metrics.leakage is None:
        raise AttackError("not an evaluation run with leakage metrics")
    heldout = ((config.get("eval") or {}).get("secrets") or {}).get("heldout")
    if not heldout:
        raise AttackError("config.yaml has no eval.secrets.heldout")
    samples = _read_samples(run_dir / SAMPLES_FILE)
    missing = [s for s in heldout if s not in samples]
    if missing:
        raise AttackError(f"samples.jsonl has no replies for held-out secrets {missing}")
    return config, metrics, [str(s) for s in heldout], samples


def load_train(run_dir: Path, config: dict) -> tuple[list[str], dict[str, list[Sample]]]:
    train = [str(s) for s in ((config.get("eval") or {}).get("secrets") or {}).get("train") or []]
    path = run_dir / TRAIN_SAMPLES_FILE
    if not train or not path.exists():
        raise AttackError(
            f"--calibrate needs {TRAIN_SAMPLES_FILE} and eval.secrets.train; this run has "
            "neither (baseline runs saved train replies only from 2026-10-19 on)"
        )
    samples = _read_samples(path)
    missing = [s for s in train if s not in samples]
    if missing:
        raise AttackError(f"{TRAIN_SAMPLES_FILE} has no replies for train secrets {missing}")
    return train, samples


# -- prompt sets ------------------------------------------------------------------------------


def prompt_set_of(config: dict) -> dict | None:
    """config.yaml eval.prompt_set (runs made with --prompt-set), else None."""
    return (config.get("eval") or {}).get("prompt_set")


def check_prompt_mix(
    prompt_set: dict,
    secrets: Sequence[str],
    samples: Mapping[str, list[Sample]],
    *,
    label: str,
    filename: str,
) -> None:
    """Every reply's prompt is in the set with the recorded text, and every secret has exactly
    replies_per_prompt replies per prompt (the assignment generation used)."""
    name, r = prompt_set["name"], prompt_set["replies_per_prompt"]
    texts = {p["id"]: p["text"] for p in prompt_set["prompts"]}
    expected = {pid: r for pid in texts}
    for secret in secrets:
        for s in samples[secret]:
            if s.prompt_id not in texts:
                raise AttackError(
                    f"{filename}: {label} secret {secret} reply {s.reply_index} has prompt "
                    f"{s.prompt_id!r}, which is not in prompt set {name!r} {list(texts)}"
                )
            if s.prompt != texts[s.prompt_id]:
                raise AttackError(
                    f"{filename}: {label} secret {secret} reply {s.reply_index}: the text of "
                    f"prompt {s.prompt_id!r} differs from the copy in eval.prompt_set"
                )
        counts = Counter(s.prompt_id for s in samples[secret])
        got = {pid: counts[pid] for pid in texts}  # set order, zeros included
        if got != expected:
            raise AttackError(
                f"{filename}: {label} secret {secret} has replies per prompt {got}, but prompt "
                f"set {name!r} assigns {expected}"
            )


def prompt_set_note(prompt_set: dict) -> str:
    return (
        f"prompt set {prompt_set['name']!r}: every reply scored under its own prompt "
        f"({len(prompt_set['prompts'])} prompts x {prompt_set['replies_per_prompt']} replies "
        "per secret)"
    )


# -- sampling settings ------------------------------------------------------------------------


def generation_settings(config: dict) -> SamplingSettings:
    """What generation used, from config.yaml: structured fields (runs from 2026-10-21 on),
    else parsed from the older free-text generation.sampling line."""
    gen = config.get("generation") or {}
    temperature = (config.get("eval") or {}).get("temperature")
    if "top_k" in gen:
        return SamplingSettings(
            temperature=temperature,
            top_k=gen.get("top_k"),
            top_p=gen.get("top_p"),
            repetition_penalty=gen.get("repetition_penalty"),
            padding_side=gen.get("padding_side"),
            prompt_padding=gen.get("prompt_padding"),
            dtype=gen.get("dtype"),
            source="config.yaml generation.*",
        )
    text = str(gen.get("sampling") or "")
    top_k = re.search(r"top_k\s*=\s*(\d+)", text)
    top_p = re.search(r"top_p\s*=\s*([\d.]+)", text)
    no_penalty = "no repetition penalty" in text
    baseline = (
        gen.get("backend") == "real" and config.get("script") == "scripts/baseline_leakage.py"
    )
    return SamplingSettings(
        temperature=temperature,
        top_k=int(top_k.group(1)) if top_k else None,
        top_p=float(top_p.group(1)) if top_p else None,
        repetition_penalty=1.0 if no_penalty else None,
        # baseline_leakage.py always generated from one unpadded prompt with a left-padding
        # tokenizer; older configs did not record it, so it is inferred from the script.
        padding_side="left" if baseline else None,
        prompt_padding="none (one prompt per generate call, num_return_sequences)"
        if baseline
        else None,
        dtype=gen.get("dtype"),
        source="parsed from config.yaml generation.sampling text"
        + ("; padding inferred from scripts/baseline_leakage.py" if baseline else ""),
    )


def sampling_check(config: dict, scorer: ScorerInfo, tokenizer: Any = None) -> SamplingCheck:
    gen = generation_settings(config)
    sc = SamplingSettings(
        temperature=scorer.temperature,
        **SCORER_SAMPLING,
        padding_side=SCORER_PADDING,
        prompt_padding=SCORER_PROMPT_PADDING,
        dtype=scorer.dtype,
        source="fsa.attack.likelihood (full softmax of logits / T)",
    )
    mismatches, notes = [], []

    def need(name: str, value: Any, ok: Callable[[Any], bool], want: str) -> None:
        if value is None:
            mismatches.append(f"generation {name} not recorded, cannot verify (scorer: {want})")
        elif not ok(value):
            mismatches.append(f"generation {name} = {value}, scorer uses {want}")

    need(
        "temperature",
        gen.temperature,
        lambda t: math.isclose(t, sc.temperature),
        f"{sc.temperature}",
    )
    need("top_k", gen.top_k, lambda k: k == 0, "0 (no truncation)")
    need("top_p", gen.top_p, lambda p: math.isclose(p, 1.0), "1.0 (no truncation)")
    need("repetition_penalty", gen.repetition_penalty, lambda r: math.isclose(r, 1.0), "1.0")
    need(
        "prompt padding",
        gen.prompt_padding,
        lambda p: str(p).startswith("none"),
        "an unpadded prompt (a padded prompt shifts positions)",
    )
    notes.append(
        "tokenizer padding side is irrelevant here: generation never padded the prompt, and the "
        "scorer only pads after the reply's real tokens"
    )
    if gen.dtype and scorer.dtype and gen.dtype != scorer.dtype:
        notes.append(
            f"dtype differs (generation {gen.dtype}, scorer {scorer.dtype}): same distribution, "
            "small numerical differences in the logits"
        )
    if tokenizer is not None and getattr(tokenizer, "padding_side", None):
        notes.append(f"scorer tokenizer padding_side = {tokenizer.padding_side} (unused)")
    return SamplingCheck(
        generation=gen, scorer=sc, matches=not mismatches, mismatches=mismatches, notes=notes
    )


# -- scoring ----------------------------------------------------------------------------------


def reply_ids(
    samples: list[Sample],
    tokenizer: Any,
    *,
    use_stored: bool,
    end_id: int | None,
    max_new_tokens: int | None,
    stats: dict[str, int],
) -> list[list[int]]:
    """IDs to score for each reply, updating the tokenisation counters in `stats`."""
    out = []
    for s in samples:
        stats["n_replies"] += 1
        stats["n_roundtrip_exact"] += roundtrips(tokenizer, s.reply)
        fresh = retokenize(tokenizer, s.reply, end_id=end_id, max_new_tokens=max_new_tokens)
        if s.reply_token_ids is not None:
            stats["n_with_stored"] += 1
            stats["n_stored_match_retokenized"] += s.reply_token_ids == fresh
        if use_stored and s.reply_token_ids is not None:
            stats["n_stored_ids_used"] += 1
            out.append(list(s.reply_token_ids))
        else:
            stats["n_retokenized"] += 1
            out.append(fresh)
    return out


def _as_array(scores: Any, n: int) -> np.ndarray:
    if hasattr(scores, "detach"):  # torch tensor on any device
        scores = scores.detach().to("cpu").double().numpy()
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if arr.shape != (n,):
        raise AttackError(f"score_candidates returned shape {arr.shape}, expected ({n},)")
    if not np.all(np.isfinite(arr)):
        raise AttackError("score_candidates returned non-finite scores")
    return arr


def softmax(scores: np.ndarray) -> np.ndarray:
    z = np.exp(scores - scores.max())
    return z / z.sum()


def _no_grad():
    try:
        import torch
    except ImportError:
        return contextlib.nullcontext()
    return torch.inference_mode()


@dataclass
class Scored:
    rows: list[np.ndarray]  # one [n_candidates] score vector per secret
    tokenization: TokenizationReport
    seconds: float
    max_chunk: int


def score_split(
    secrets: Sequence[str],
    samples: Mapping[str, list[Sample]],
    *,
    label: str,
    model: Any,
    tokenizer: Any,
    scorer: ScorerInfo,
    templates: Mapping[str, str],
    candidates: Sequence[str],
    max_batch_tokens: int,
    max_new_tokens: int | None,
    log: Callable[[str], None],
    score_fn: Callable | None = None,
    quiet_secrets: bool = False,
) -> Scored:
    """Score every candidate against each secret's replies (held-out or train, same code).
    `score_fn` (same signature as likelihood.score_candidates) replaces the model's scorer; only
    the mock dry run (scripts/inadvertent.py) passes one."""
    # Looked up at call time (not imported by name) so tests can monkeypatch either function.
    if score_fn is None:
        score_fn = (
            likelihood.score_candidates
            if scorer.prompt_cache
            else likelihood.score_candidates_uncached
        )
    end_id = turn_end_id(tokenizer)
    stats: dict[str, int] = defaultdict(int)
    builders: dict[str, Callable[[str], list[int]]] = {}
    prompt_len: dict[str, int] = {}
    per_secret: dict[str, dict[str, list[list[int]]]] = {}
    for secret in secrets:
        by_prompt: dict[str, list[Sample]] = defaultdict(list)
        for s in samples[secret]:
            by_prompt[s.prompt].append(s)
        per_secret[secret] = {}
        for prompt, group in by_prompt.items():
            if prompt not in builders:
                builders[prompt] = make_prompt_builder(tokenizer, templates, scorer.method, prompt)
                prompt_len[prompt] = max(len(builders[prompt](c)) for c in candidates)
            per_secret[secret][prompt] = reply_ids(
                group,
                tokenizer,
                use_stored=scorer.same_as_generator,
                end_id=end_id,
                max_new_tokens=max_new_tokens,
                stats=stats,
            )
    tokenization = TokenizationReport(
        n_replies=stats["n_replies"],
        n_stored_ids_used=stats["n_stored_ids_used"],
        n_retokenized=stats["n_retokenized"],
        n_roundtrip_exact=stats["n_roundtrip_exact"],
        n_stored_match_retokenized=(
            stats["n_stored_match_retokenized"] if stats["n_with_stored"] else None
        ),
    )
    log(
        f"{label} replies: {tokenization.n_replies}; stored IDs used "
        f"{tokenization.n_stored_ids_used}, re-tokenised {tokenization.n_retokenized}; text "
        f"round-trips exactly {tokenization.n_roundtrip_exact}/{tokenization.n_replies}"
        + (
            f"; stored IDs == re-tokenisation {tokenization.n_stored_match_retokenized}"
            f"/{stats['n_with_stored']}"
            if stats["n_with_stored"]
            else ""
        )
    )
    log(
        f"{label}: prompt length {max(prompt_len.values())} tokens; max_batch_tokens "
        f"{max_batch_tokens}; {len(secrets)} secrets x {len(candidates)} candidates"
    )
    started = time.perf_counter()
    rows: list[np.ndarray] = []
    max_chunk = 1
    with _no_grad():
        for i, secret in enumerate(secrets, 1):
            scores = np.zeros(len(candidates))
            for prompt, ids in per_secret[secret].items():
                blocks = plan_blocks(
                    prompt_len[prompt], [len(r) for r in ids], len(candidates), max_batch_tokens
                )
                for b in blocks:
                    max_chunk = max(max_chunk, len(b.candidates))
                    out = score_fn(
                        model,
                        tokenizer,
                        builders[prompt],
                        [ids[r] for r in b.replies],
                        [candidates[c] for c in b.candidates],
                        scorer.temperature,
                    )
                    scores[list(b.candidates)] += _as_array(out, len(b.candidates))
            rows.append(scores)
            elapsed = time.perf_counter() - started
            eta = elapsed / i * (len(secrets) - i)
            if quiet_secrets:  # neither the secret nor the guess
                log(
                    f"  [{label} {i}/{len(secrets)}] rank of the secret "
                    f"{int((scores > scores[list(candidates).index(secret)]).sum()) + 1}; "
                    f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s"
                )
            else:
                log(
                    f"  [{label} {i}/{len(secrets)}] secret {secret}: top guess "
                    f"{candidates[int(scores.argmax())]}, rank of secret "
                    f"{int((scores > scores[list(candidates).index(secret)]).sum()) + 1}; "
                    f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s"
                )
    return Scored(rows, tokenization, time.perf_counter() - started, max_chunk)


# -- metrics ----------------------------------------------------------------------------------


def _leakage(
    posteriors: np.ndarray,
    *,
    candidates: Sequence[str],
    secrets: Sequence[str],
    samples: Mapping[str, list[Sample]],
    metrics: RunMetrics,
    trained_fresh: bool,
    n_permutations: int,
    n_bootstrap: int,
    seed: int,
    space: SecretSpace | None = None,
) -> LeakageMetrics:
    words = space is not None and space.type == "words"
    if words:  # mentions of each candidate word; exact mention ratio
        vocab = {w: j for j, w in enumerate(candidates)}
        counts = [
            [sum(col) for col in zip(*(word_mentions(s.reply, vocab) for s in samples[sec]))]
            for sec in secrets
        ]
        diag = mention_diagnostics(list(secrets), counts, candidates)
    elif space is not None and space.type == "digits4":  # candidate-code occurrences
        nums = {sec: [n for s in samples[sec] for n in s.numbers] for sec in secrets}
        counts = [code_counts(nums[sec], candidates, 4) for sec in secrets]
        digits = [digit_counts(nums[sec], 4) for sec in secrets]
        diag = mention_diagnostics(list(secrets), counts, candidates)
    else:
        counts = [number_counts([n for s in samples[sec] for n in s.numbers]) for sec in secrets]
        digits = [digit_counts([n for s in samples[sec] for n in s.numbers]) for sec in secrets]
        diag = secret_diagnostics(list(secrets), counts)
    space = metrics.leakage.secret_space_size
    significance = significance_metrics(
        posteriors=list(posteriors),
        candidates=candidates,
        secrets=secrets,
        exact_observed=[r.exact_observed for r in diag],
        exact_expected=[r.exact_expected for r in diag],
        secret_space_size=space,
        n_permutations=n_permutations,
        n_resamples=n_bootstrap,
        seed=seed,
    )
    return leakage_metrics(
        attacker_name=ATTACKER_NAME,
        posteriors=list(posteriors),
        candidates=candidates,
        secrets=secrets,
        secret_space_size=space,
        n_replies_per_secret=metrics.leakage.n_replies_per_secret,
        eval_split="heldout",
        attacker_trained_fresh=trained_fresh,
        kl_upper_bound_bits=metrics.leakage.kl_upper_bound_bits,
        total_digit_counts=None if words else [sum(c) for c in zip(*digits)],
        significance=significance,
    )


def likelihood_attack(
    run_dir: Path,
    *,
    model: Any = None,
    tokenizer: Any = None,
    scorer: ScorerInfo | None = None,
    templates: Mapping[str, str] | None = None,
    max_batch_tokens: int | None = None,
    limit_secrets: int | None = None,
    calibrate: bool = False,
    saved: LikelihoodAttack | None = None,
    saved_name: str | None = None,
    n_permutations: int = 1000,
    n_bootstrap: int = 2000,
    seed: int = 0,
    log: Callable[[str], None] = flush_print,
    now: datetime | None = None,
    extra_notes: Sequence[str] = (),
    score_fn: Callable | None = None,
    quiet_secrets: bool = False,
) -> tuple[LikelihoodAttack, Path]:
    """Score (or reuse `saved` scores), analyse, write a new likelihood_attack*.json.

    With `saved`, its held-out scores, scorer and tokenisation report are reused; a model is
    needed only if `calibrate` and `saved` has no train scores yet.
    """
    run_dir = Path(run_dir)
    config, metrics, heldout, samples = load_run(run_dir)
    secret_space = space_from_record((config.get("eval") or {}).get("secret_space"))
    words = secret_space.type == "words"
    templates = (
        templates if templates is not None else load_system_templates(secret_type=secret_space.type)
    )
    notes: list[str] = list(extra_notes)
    space = metrics.leakage.secret_space_size
    if words or secret_space.type == "digits4":  # the run's recorded words / codes, in order
        candidates = list(secret_space.secrets)
    else:
        width = len(heldout[0])
        candidates = [f"{i:0{width}d}" for i in range(space)]
    tokens_checked = False

    def check_word_tokens() -> None:
        """Word runs, once, before any scoring: equal token spans for every candidate."""
        nonlocal tokens_checked
        if not words or tokens_checked:
            return
        counts = secret_span_tokens(tokenizer, templates, scorer.method, candidates)
        check_equal_token_spans(counts, what="word")
        log(f"words: {token_report(counts)}")
        tokens_checked = True

    max_new_tokens = (config.get("generation") or {}).get("max_new_tokens")
    pset = prompt_set_of(config)

    def scoring_args() -> dict:
        if tokenizer is None or scorer is None or max_batch_tokens is None:
            raise AttackError("scoring needs a loaded scorer (tokenizer, scorer, max_batch_tokens)")
        return {
            "model": model,
            "tokenizer": tokenizer,
            "scorer": scorer,
            "templates": templates,
            "candidates": candidates,
            "max_batch_tokens": max_batch_tokens,
            "max_new_tokens": max_new_tokens,
            "log": log,
            "score_fn": score_fn,
            "quiet_secrets": quiet_secrets,
        }

    if saved is not None:
        if saved.run_id != run_dir.name:
            raise AttackError(f"{saved_name} belongs to run {saved.run_id}, not {run_dir.name}")
        scorer, heldout = saved.scorer, list(saved.secrets)
        held = Scored(
            [np.asarray(r, dtype=np.float64) for r in saved.scores],
            saved.tokenization,
            saved.seconds_total,
            saved.candidates_per_call,
        )
        batch_tokens = saved.max_batch_tokens
        notes.append(f"held-out scores reused from {saved_name}")
        notes += [n for n in saved.notes if not n.startswith("held-out scores reused")]
    else:
        if limit_secrets is not None and limit_secrets < 2:
            raise AttackError("--limit-secrets must be >= 2 (the diagnostics compare secrets)")
        if limit_secrets is not None and limit_secrets < len(heldout):
            heldout = heldout[:limit_secrets]
            notes.append(f"limited to the first {limit_secrets} held-out secrets")
        if metrics.is_mock:
            notes.append(
                "mock run: replies are synthetic, so scores say nothing about a real model"
            )
        if scorer is not None and not scorer.same_as_generator:
            notes.append("surrogate scorer: replies were re-tokenised with the scorer's tokenizer")
        if scorer is not None and not scorer.prompt_cache:
            notes.append("--no-cache: prompt keys/values recomputed for every reply (slower)")
        if pset is not None:
            check_prompt_mix(pset, heldout, samples, label="held-out", filename=SAMPLES_FILE)
        scoring_args()  # fails early if the scorer is not loaded
        check_word_tokens()
        held = score_split(heldout, samples, label="heldout", **scoring_args())
        batch_tokens = max_batch_tokens
    if pset is not None:
        if saved is not None:  # the held-out replies still feed the count-based diagnostics
            check_prompt_mix(pset, heldout, samples, label="held-out", filename=SAMPLES_FILE)
        if prompt_set_note(pset) not in notes:
            notes.append(prompt_set_note(pset))

    threat = threat_model_of(scorer)
    if threat == "public_weights" and not any(n.startswith("threat model") for n in notes):
        run_adapter = (config.get("generation") or {}).get("adapter")
        notes.append(
            "threat model public_weights: scored with the base model only; the run's adapter "
            f"({run_adapter}) was not used, for train secrets either"
        )
    sampling = sampling_check(config, scorer, tokenizer)
    if not sampling.matches:
        notes.append("sampling mismatch: " + "; ".join(sampling.mismatches))

    held_scores = np.stack(held.rows)
    true_idx = [candidates.index(s) for s in heldout]
    leak_kwargs = {
        "candidates": candidates,
        "secrets": heldout,
        "samples": samples,
        "metrics": metrics,
        "trained_fresh": scorer.same_as_generator,
        "n_permutations": n_permutations,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "space": secret_space,
    }
    leakage = _leakage(tempered_posteriors(held_scores), **leak_kwargs)

    calibration = None
    log2p_cal = None
    if calibrate:
        if saved is not None and saved.calibration is not None:
            cal = saved.calibration
            train = list(cal.train_secrets)
            train_rows = np.asarray(cal.train_scores, dtype=np.float64)
            train_tok, train_secs = cal.train_tokenization, cal.train_seconds
            notes.append(f"train scores reused from {saved_name}")
        else:
            train, train_samples = load_train(run_dir, config)
            if saved is None and limit_secrets is not None and limit_secrets < len(train):
                train = train[:limit_secrets]
                notes.append(f"calibration limited to the first {limit_secrets} train secrets")
            if pset is not None:  # same prompts, same replies per prompt as held-out
                check_prompt_mix(
                    pset, train, train_samples, label="train", filename=TRAIN_SAMPLES_FILE
                )
            scoring_args()
            check_word_tokens()
            scored = score_split(train, train_samples, label="train", **scoring_args())
            train_rows = np.stack(scored.rows)
            train_tok, train_secs = scored.tokenization, scored.seconds
        fit = fit_temperature(train_rows, [candidates.index(s) for s in train])
        if fit.at_bound:
            notes.append(f"calibration: tau = {fit.tau:.3g} is at the search bound")
        calibration = Calibration(
            tau=fit.tau,
            tau_at_search_bound=fit.at_bound,
            train_secrets=train,
            train_scores=train_rows.tolist(),
            train_tokenization=train_tok,
            train_seconds=train_secs,
            train_mean_log2_p_raw=fit.mean_log2_p_raw,
            train_mean_log2_p_calibrated=fit.mean_log2_p_calibrated,
            leakage=_leakage(tempered_posteriors(held_scores, fit.tau), **leak_kwargs),
        )
        log2p_cal = log2_p_true(held_scores, true_idx, fit.tau)

    ranks = true_ranks(held_scores, true_idx)
    n_first, first_raw, rest_raw = first_vs_not(log2_p_true(held_scores, true_idx), ranks)
    first_cal = rest_cal = None
    if log2p_cal is not None:
        _, first_cal, rest_cal = first_vs_not(log2p_cal, ranks)
    rank_summary = RankSummary(
        ranks=ranks,
        histogram=rank_histogram(ranks, space),
        n_first=n_first,
        mean_log2_p_true_first=first_raw,
        mean_log2_p_true_not_first=rest_raw,
        mean_log2_p_true_first_calibrated=first_cal,
        mean_log2_p_true_not_first_calibrated=rest_cal,
    )

    result = LikelihoodAttack(
        run_id=run_dir.name,
        created_at=now or datetime.now(),
        is_mock=metrics.is_mock,
        scorer=scorer,
        tokenization=held.tokenization,
        max_batch_tokens=batch_tokens,
        candidates_per_call=held.max_chunk,
        seconds_total=held.seconds,
        secrets=heldout,
        scores=held_scores.tolist(),
        leakage=leakage,
        notes=notes,
        sampling=sampling,
        ranks=rank_summary,
        calibration=calibration,
        rescored_from=saved_name,
        threat_model=threat,
    )
    path = output_path(run_dir, threat)
    with path.open("x", encoding="utf-8", newline="\n") as f:  # "x": never overwrite
        f.write(result.model_dump_json(indent=2) + "\n")
    return result, path


# -- report -----------------------------------------------------------------------------------


def _fmt(x: float | None, spec: str = ".2f") -> str:
    return "-" if x is None else format(x, spec)


def format_report(r: LikelihoodAttack) -> str:
    """Sampling check, raw vs calibrated side by side, and the rank summary."""
    s = r.sampling
    lines = [f"threat model: {THREAT_LABELS[threat_model_of(r.scorer)]}"]
    if s is not None:
        g, c = s.generation, s.scorer
        lines += [
            "sampling     generation               scorer",
            f"  T          {_fmt(g.temperature, 'g'):<24} {_fmt(c.temperature, 'g')}",
            f"  top_k      {_fmt(g.top_k, 'd'):<24} {c.top_k}",
            f"  top_p      {_fmt(g.top_p, 'g'):<24} {c.top_p}",
            f"  rep. pen.  {_fmt(g.repetition_penalty, 'g'):<24} {c.repetition_penalty}",
            f"  padding    {g.padding_side or '-'} tokenizer, prompt {g.prompt_padding or '-'}",
            f"             scorer: {c.padding_side}; prompt {c.prompt_padding}",
            f"  dtype      {g.dtype or '-':<24} {c.dtype}",
            f"  source     {g.source}",
            "  MATCH" if s.matches else "  MISMATCH: " + "; ".join(s.mismatches),
        ]
    raw, cal = r.leakage, (r.calibration.leakage if r.calibration else None)
    head = "raw" + (f"        calibrated (tau = {r.calibration.tau:.3g})" if cal else "")
    lines.append(f"\n                       {head}")

    def row(name: str, f: Callable[[LeakageMetrics], str]) -> str:
        return f"  {name:<20} {f(raw):<10}" + (f" {f(cal)}" if cal else "")

    lines += [
        row("exact", lambda m: f"{m.attacker.exact_match.value:.1%}"),
        row("top-5", lambda m: f"{m.attacker.top5.value:.1%}"),
        row("bits (lower bound)", lambda m: f"{m.bits_leaked_lower_bound:.3f}"),
        row("bits 95% CI", lambda m: format_ci(m.significance.bits_ci) if m.significance else "-"),
        row(
            "permutation p",
            lambda m: f"{m.significance.permutation.p_value:.3g}" if m.significance else "-",
        ),
    ]
    if r.calibration:
        cb = r.calibration
        lines.append(
            f"  train mean log2 p(true): raw {cb.train_mean_log2_p_raw:.2f}, calibrated "
            f"{cb.train_mean_log2_p_calibrated:.2f} ({len(cb.train_secrets)} train secrets)"
        )
    if r.ranks is not None:
        k = r.ranks
        hist = ", ".join(f"{b}: {n}" for b, n in k.histogram.items())
        lines += [
            f"\nrank of the true secret ({len(k.ranks)} held-out): {hist}",
            (
                f"  ranked first {k.n_first}/{len(k.ranks)}: mean log2 p(true) raw "
                f"{_fmt(k.mean_log2_p_true_first)} when first, {_fmt(k.mean_log2_p_true_not_first)} "
                "when not"
            ),
        ]
        if r.calibration:
            lines.append(
                f"  calibrated: {_fmt(k.mean_log2_p_true_first_calibrated)} when first, "
                f"{_fmt(k.mean_log2_p_true_not_first_calibrated)} when not"
            )
    return "\n".join(lines)


# -- CLI --------------------------------------------------------------------------------------


def _auto_max_batch_tokens(model: Any, device: str) -> int:
    if not str(device).startswith("cuda"):
        return CPU_MAX_BATCH_TOKENS
    import torch

    cfg = model.config
    text_cfg = getattr(cfg, "text_config", None) or cfg
    return suggest_max_batch_tokens(
        gpu_memory_bytes=torch.cuda.get_device_properties(torch.device(device)).total_memory,
        weight_bytes=model.get_memory_footprint(),
        vocab_size=text_cfg.vocab_size,
        hidden_size=text_cfg.hidden_size,
    )


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path, help="run folder (outputs/runs/<run_id>)")
    parser.add_argument(
        "--scorer-model",
        default=None,
        help="registry id, HF id or local path (default: the run's own model)",
    )
    parser.add_argument("--temperature", type=float, default=None, help="default: the run's")
    parser.add_argument(
        "--max-batch-tokens", type=int, default=None, help="default: sized from GPU memory"
    )
    parser.add_argument("--device", default=None, help="override hf.device (cpu, cuda, ...)")
    parser.add_argument(
        "--dtype",
        default=None,
        choices=("auto", "float32", "bfloat16", "float16"),
        help="override hf.dtype; auto = bfloat16 on cuda, float32 on cpu",
    )
    parser.add_argument("--limit-secrets", type=int, default=None, help="first N held-out only")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="re-run the prompt with every reply instead of reusing its keys/values (slower; "
        "for cross-checking the cache)",
    )
    parser.add_argument(
        "--scorer-no-adapter",
        action="store_true",
        help="public-weights attacker: score an adapter run's replies with the base model only "
        "(same system template and prompts as generation); written to "
        "likelihood_attack_public_weights*.json",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="also score train_samples.jsonl and fit a posterior temperature tau there "
        "(roughly doubles GPU time)",
    )
    parser.add_argument(
        "--from-scores",
        type=Path,
        default=None,
        help="reuse the held-out scores of an earlier likelihood_attack*.json (no model needed "
        "unless --calibrate has to score the train secrets)",
    )
    parser.add_argument(
        "--no-secrets-in-log",
        action="store_true",
        help="per-secret progress without the secret or the top guess (cluster job logs)",
    )
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        saved = None
        if args.from_scores is not None:
            saved = LikelihoodAttack.model_validate_json(
                args.from_scores.read_text(encoding="utf-8")
            )
            saved_public = threat_model_of(saved.scorer) == "public_weights"
            if args.scorer_no_adapter and not saved_public:
                raise AttackError(
                    f"--scorer-no-adapter, but {args.from_scores.name} was scored as "
                    f"{threat_model_of(saved.scorer)}; the saved file decides the scorer"
                )
        needs_model = saved is None or (args.calibrate and saved.calibration is None)
        model = tokenizer = scorer = batch_tokens = templates = None
        load_notes: list[str] = []
        if needs_model:
            model, tokenizer, scorer, batch_tokens, templates, load_notes = _load_scorer(args, saved)
        result, path = likelihood_attack(
            args.run,
            model=model,
            tokenizer=tokenizer,
            scorer=scorer,
            templates=templates,
            max_batch_tokens=batch_tokens,
            limit_secrets=args.limit_secrets,
            calibrate=args.calibrate,
            saved=saved,
            saved_name=args.from_scores.name if args.from_scores else None,
            n_permutations=args.n_permutations,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            extra_notes=load_notes,
            quiet_secrets=args.no_secrets_in_log,
        )
    except (AttackError, GeneratorUnavailable, OSError, KeyError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"threat model: {THREAT_LABELS[threat_model_of(result.scorer)]}")
    for note in result.notes:
        print(f"note: {note}")
    print(format_significance(result.leakage.significance))
    print(format_report(result))
    secs = result.seconds_total + (result.calibration.train_seconds if result.calibration else 0)
    print(f"scoring took {secs:.0f}s; wrote {path}", flush=True)
    return 0


def _load_scorer(args, saved: LikelihoodAttack | None):
    """Load the scorer model (the run's own unless --scorer-model; saved's when reusing)."""
    from fsa_runtime.generators.base import ModelSpec
    from fsa_runtime.generators.hf import PLAIN_METHODS, HFGenerator, load_record
    from fsa_runtime.registry import load_registry
    from fsa_runtime.settings import ROOT, load_settings
    from fsa.training import peft_check
    from fsa.training.adapter import portable_path, resolve_path

    config = yaml.safe_load((args.run / CONFIG_FILE).read_text(encoding="utf-8"))
    run_model = config.get("model") or {}
    method = config["method"]
    registry = load_registry()
    gen = HFGenerator.from_settings(load_settings())
    run_space = space_from_record((config.get("eval") or {}).get("secret_space"))
    if run_space.type in ("words", "digits4"):
        gen.templates = load_system_templates(secret_type=run_space.type)
    if args.device:
        gen.device = args.device
    if args.dtype:
        gen.dtype = args.dtype
    if saved is not None:  # train secrets must be scored exactly like the saved held-out ones
        scorer_id, temperature = saved.scorer.model, saved.scorer.temperature
        prompt_cache = saved.scorer.prompt_cache
    else:
        scorer_id = args.scorer_model or run_model.get("id")
        temperature = args.temperature or (config.get("eval") or {}).get("temperature") or 1.0
        prompt_cache = not args.no_cache
    bm = registry.base_models.get(scorer_id)
    hf_id = bm.hf_id if bm else scorer_id
    same = hf_id == run_model.get("base")
    scorer_method = method if same or method in PLAIN_METHODS else "base"
    # The run's own adapter: fsa/b3/random_layer runs record it in config.yaml
    # generation.adapter (older setups: the registry's adapters).
    gen_cfg = config.get("generation") or {}
    recorded = gen_cfg.get("adapter")
    run_adapter = None
    # Runs made with --adapter-key record the key and its pin: the adapter is found again through
    # this machine's paths.local.yaml and must have the same sha256 (checked before loading).
    adapter_key, adapter_pin = gen_cfg.get("adapter_key"), gen_cfg.get("adapter_sha256_pinned")
    if method not in PLAIN_METHODS:
        if adapter_key:  # this machine's path for the key, else the recorded one
            path, _ = peft_check.local_adapter(adapter_key, ROOT)
            run_adapter = str(path) if path else (resolve_path(recorded, ROOT) if recorded else None)
        elif recorded:
            run_adapter = resolve_path(recorded, ROOT)
        elif bm:
            run_adapter = bm.adapter_for(method)
    if saved is not None:  # the saved file fixes the scorer exactly, adapter or not
        public = threat_model_of(saved.scorer) == "public_weights"
        adapter = resolve_path(saved.scorer.adapter, ROOT) if saved.scorer.adapter else None
    else:
        public = args.scorer_no_adapter
        if public and not same:
            raise AttackError(
                "--scorer-no-adapter scores with the run's own base model; drop --scorer-model"
            )
        if public and run_adapter is None:
            raise AttackError(
                f"run method {method!r} used no adapter, so its scorer already uses only the "
                "public weights; --scorer-no-adapter is for fsa/b3/random_layer runs"
            )
        adapter = run_adapter if same and not public else None
    threat = "surrogate" if not same else ("public_weights" if public else "grey_box")
    # Prompts: ScorerInfo.method stays the run's method, so the candidate prompts use the same
    # system template as generation (fsa.prompting.system_prompt). The model is loaded as a
    # plain base model when no adapter is attached.
    # The run's own model is scored at the revision generation recorded (config.yaml
    # model.revision; older runs recorded none and load as before); a surrogate at its registry pin.
    revision = run_model.get("revision") if same else (bm.revision if bm else None)
    spec = ModelSpec(
        method=scorer_method if adapter else "base",
        base_model=bm.id if bm else scorer_id,
        hf_id=hf_id,
        adapter=adapter,
        revision=revision,
        adapter_sha256=adapter_pin if adapter else None,
    )
    gen.check(spec)
    print(f"run:    {args.run.name} ({run_model.get('id')}, method {method})", flush=True)
    print(f"threat model: {THREAT_LABELS[threat]}", flush=True)
    print(
        f"scorer: {scorer_id} ({hf_id}){'' if same else ' SURROGATE'}, "
        f"method {scorer_method}, T = {temperature}, {gen.device}/{gen.dtype}"
        + (f", adapter {adapter}" if adapter else ", no adapter")
        + ("" if prompt_cache else ", no prompt cache")
        + (", calibrating on train secrets" if args.calibrate else ""),
        flush=True,
    )
    t0 = time.perf_counter()
    tokenizer, model = gen._load(spec)
    print(f"model loaded in {time.perf_counter() - t0:.0f}s", flush=True)
    loaded = load_record(gen, spec) or {}
    load_notes = [
        (
            f"scorer loaded revision {loaded.get('revision_loaded')} (pinned {revision}), "
            f"dtype {loaded.get('dtype_loaded')}"
        )
    ]
    if run_model.get("revision") and same and loaded.get("revision_loaded") not in (None, revision):
        raise AttackError(f"scorer loaded {loaded.get('revision_loaded')}, the run used {revision}")
    check = loaded.get("adapter_check")
    if check:
        c = check["counts"]
        load_notes.append(
            f"adapter sha256 {check['adapter_sha256'][:16]}... "
            + (f"matches the pin {adapter_pin[:16]}..." if check["adapter_sha256_checked"] else "(no pin)")
            + f"; LoRA tensors attached {c['tensors_attached']}/{c['tensors_in_file']}, missing "
            f"{c['missing']}, unexpected {c['unexpected']}, remapped {check['remapped']}"
        )
    for n in load_notes:
        print(n, flush=True)
    scorer = ScorerInfo(
        model=scorer_id,
        hf_id=hf_id,
        same_as_generator=same,
        method=scorer_method,
        device=str(gen.device),
        dtype=gen.dtype,
        temperature=temperature,
        prompt_cache=prompt_cache,
        adapter=portable_path(adapter, ROOT) if adapter else None,
        threat_model=threat,
    )
    batch_tokens = args.max_batch_tokens or _auto_max_batch_tokens(model, gen.device)
    return model, tokenizer, scorer, batch_tokens, gen.templates, load_notes


if __name__ == "__main__":
    sys.exit(main())
