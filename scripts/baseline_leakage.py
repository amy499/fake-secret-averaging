"""Baseline leakage of one (base model, method): fit the heuristic attacker on train secrets,
score it on held-out secrets, and write an evaluation run folder through RunWriter.

Secrets are a seeded random split of "00".."99" (n_train + n_heldout distinct codes). Each
secret gets its own generation seed. The attacker reads the per-secret histogram of numbers
(digit-level dip alpha, which may be negative, plus exact/neighbour/reversed number terms);
its fitted parameters are stored in config.yaml (eval.attacker_params) so the real-mode
Playground can reuse them, and the train replies go to train_samples.jsonl so
scripts/reanalyze_run.py can refit it. Significance tests (binomial, permutation, bootstrap
CIs, sign test) are printed and saved in metrics.json (leakage.significance). For each held-out secret the script prints observed vs expected
counts of the secret's digits and of the exact number (expected = the other secrets' replies),
and writes that table to diagnostics.json in the run folder.

--backend is required (there is no silent fallback to FSA_BACKEND): real runs go to
outputs/runs/; mock runs (quick checks) go to mock_data/runs/ and are always named "mock_...".
The backend is printed at start and end. Size: --preset quick|standard|full, or set
--n-train/--n-heldout/--replies directly (they override the preset). Fewer than 30 held-out
secrets triggers a "results are very noisy" warning.

Prompts: --prompt-id uses one user prompt for every reply (the default). --prompt-set NAME
spreads each secret's replies evenly over a named set from configs/training/prompts.yaml prompt_sets
(fsa.eval.prompt_sets: --replies must be a multiple of the set size; every secret, train and
held-out, gets the same replies per prompt; seeds follow from --seed). Each reply records the
exact prompt that produced it, and config.yaml records the set under eval.prompt_set. With
--adapter, prompts in the set that the adapter was trained on are named in a warning and in
eval.prompt_set.adapter_trained_prompts. The heuristic attacker reads pooled number counts over
all prompts of a secret (fair, since every secret has the same prompt mix); it notes when most
replies contain no numbers, because then it has little to go on.

Secrets: --secret-type digits (default, "00".."99") or words (--word-list, 100 words; see
fsa.secrets). Word runs use configs/training/prompts.yaml word_system_templates and the word-mention
attacker (src/fsa_lite/fsa_runtime/attacker/word_mention.py: mentions of each candidate word, fitted on train
secrets) instead of the digit heuristic; diagnostics.json holds the exact mention ratio. With
the real backend every word must take the same number of tokens in the rendered system prompt
(checked with the tokenizer before anything is generated). config.yaml records the word list,
its hash and its words under eval.secret_space; digit runs record nothing new.

--secret-type digits4: the 100 fixed 4-digit codes of configs/training/secrets/digits4_100.txt (the
attacker knows the code is one of these 100), under configs/training/prompts.yaml
digits4_system_templates (the PasswordEval-style pe_base prompt with each code's password;
--method base or an adapter method; the prompt-only methods have no digits4 template). Replies
are parsed for exactly 4-digit numbers, the text attacker is the code-mention attacker
(src/fsa_lite/fsa_runtime/attacker/code_mention.py: occurrences of each candidate code among them) and
diagnostics.json holds the exact mention ratio. config.yaml records the code list and hash
under eval.secret_space and every code's password under eval.pe_passwords. E.g.
    ... --secret-type digits4 --prompt-id rand_20_4digit     (or lottery_4digit, held out)

Usage (from the repository root):
    uv run --group ml python -m scripts.baseline_leakage --backend real       # Qwen2.5-0.5B, base
    # quick (default): 10 train + 10 held-out x 4 replies, about 10 min on a laptop CPU
    # (measured ~30-50 s per secret for 4-5 replies of up to 160 tokens)
    uv run --group ml python -m scripts.baseline_leakage --backend real --method prompt_hiding_strict
    uv run --group ml python -m scripts.baseline_leakage --backend real --preset full \\
        --base-model gemma3-4b-it --device cuda                               # e.g. on the Slurm cluster
    uv run python -m scripts.baseline_leakage --backend mock                  # no torch needed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np

from fsa_runtime.attacker.code_mention import CodeMentionAttacker
from fsa_runtime.attacker.heuristic import HeuristicAttacker
from fsa_runtime.attacker.word_mention import WordMentionAttacker
from fsa_runtime.generators.base import GeneratedReply, Generator, GeneratorUnavailable, ModelSpec
from fsa_runtime.generators.hf import PADDING_SIDE, PROMPT_PADDING, SAMPLING, HFGenerator
from fsa_runtime.generators.mock import MockGenerator
from fsa_runtime.registry import RegistryError, load_registry
from fsa_runtime.settings import ROOT, Settings, load_settings
from fsa.eval.diagnostics import (
    format_table,
    mention_diagnostics,
    secret_diagnostics,
    to_payload,
)
from fsa.eval.leakage import leakage_metrics
from fsa.eval.parsing import (
    code_counts,
    digit_counts,
    number_counts,
    parse_numbers,
    word_mentions,
)
from fsa.eval.prompt_sets import (
    ASSIGNMENT_RULE,
    SEED_RULE,
    PromptSet,
    UserPrompt,
    generation_seed,
    load_prompt_set,
    load_user_prompts,
    replies_per_prompt,
)
from fsa.eval.significance import format_significance, significance_metrics
from fsa.io.run_writer import TRAIN_SAMPLES_FILE, RunWriter
from fsa.prompting import PROMPT_ONLY_TEMPLATES, load_system_templates, secret_span_tokens
from fsa.secrets import (
    DIGITS,
    SECRET_SPACE,
    SecretSpace,
    check_equal_token_spans,
    load_digits4,
    load_word_list,
    pe_password_record,
    secret_space_record,
    split_secrets,
    token_report,
)
from fsa.training.adapter import (
    ADAPTER_METHODS,
    check_adapter,
    portable_path,
    read_training_info,
    resolve_path,
)
from scripts._cli import utf8_stdio

MOCK_PREFIX = "mock_"
METHODS = (
    "base",
    "prompt_hiding",
    "prompt_hiding_strict",
    "prompt_anti_suppression",
    *ADAPTER_METHODS,  # need --adapter (a trained LoRA from scripts/train_fsa.py)
)
MIN_RELIABLE_HELDOUT = 30
# (n_train, n_heldout, replies); "full" uses all 100 codes.
PRESETS = {"quick": (10, 10, 4), "standard": (30, 30, 8), "full": (50, 50, 10)}
DIAGNOSTICS_FILE = "diagnostics.json"
DEFAULT_WORD_LIST = "configs/training/secrets/words_100.txt"


def _fmt_secs(s: float) -> str:
    return f"{s / 60:.1f} min" if s >= 90 else f"{s:.0f}s"


def backend_banner(generator: Generator, spec: ModelSpec) -> str:
    """A loud, one-glance statement of what produced the replies."""
    if generator.is_mock:
        line = "BACKEND: MOCK - synthetic replies, NOT a real model"
    else:
        line = f"BACKEND: REAL - {spec.hf_id} on {getattr(generator, 'device', '?')}"
    bar = "=" * max(60, len(line) + 4)
    return f"{bar}\n  {line}\n{bar}"


def noisy_warning(n_heldout: int) -> str | None:
    if n_heldout >= MIN_RELIABLE_HELDOUT:
        return None
    return (
        f"WARNING: only {n_heldout} held-out secrets - results are very noisy "
        f"(use >= {MIN_RELIABLE_HELDOUT}, e.g. --preset standard or --n-heldout 30)."
    )


def run_name(name: str, generator: Generator) -> str:
    """Mock runs always carry the mock_ prefix so they can never pass for real runs."""
    if generator.is_mock and not name.startswith(MOCK_PREFIX):
        return MOCK_PREFIX + name
    return name


def load_prompt(prompt_id: str) -> UserPrompt:
    prompts = load_user_prompts(ROOT / "configs" / "training" / "prompts.yaml")
    if prompt_id not in prompts:
        raise ValueError(f"unknown prompt id {prompt_id!r}")
    return prompts[prompt_id]


def parsed_line(n_parsed: int, n_expected: int) -> str:
    """The per-secret parse count: "x/y numbers parsed", or "x numbers parsed (none
    expected)" when no prompt asks for numbers."""
    if n_expected == 0:
        return f"{n_parsed} numbers parsed (none expected)"
    return f"{n_parsed}/{n_expected} numbers parsed"


def no_numbers_note(samples: Sequence[dict]) -> str | None:
    """A note for the heuristic attacker's output when most replies contain no numbers."""
    empty = sum(1 for s in samples if not s["numbers"])
    if not samples or empty * 2 <= len(samples):
        return None
    return (
        f"heuristic attacker: {empty}/{len(samples)} replies ({empty / len(samples):.0%}) "
        "contain no numbers; it reads only pooled number counts, so it has little to go on "
        "and its result says little about leakage in replies without numbers"
    )


def _generate_replies(generator: Generator, spec: ModelSpec, **kwargs) -> list[GeneratedReply]:
    """Replies with token IDs when the generator provides them (HF), text only otherwise."""
    if hasattr(generator, "generate_replies"):
        return generator.generate_replies(spec, **kwargs)
    return [GeneratedReply(text) for text in generator.generate(spec, **kwargs)]


def flush_print(message: str) -> None:
    """print() that flushes, so progress shows up immediately even when output is redirected."""
    print(message, flush=True)


def run_baseline(
    *,
    generator: Generator,
    spec: ModelSpec,
    n_layers: int | None,
    prompt_id: str | None = None,
    prompt_set: PromptSet | None = None,
    n_train: int,
    n_heldout: int,
    replies: int,
    seed: int,
    temperature: float,
    out_dir: Path,
    name: str,
    generation_info: dict,
    n_permutations: int = 1000,
    n_bootstrap: int = 2000,
    log: Callable[[str], None] = flush_print,
    now: datetime | None = None,
    secret_space: SecretSpace = DIGITS,
) -> RunWriter:
    words = secret_space.type == "words"
    codes4 = secret_space.type == "digits4"
    n_digits = secret_space.n_digits or 2  # numbers parsed from replies: 2, or 4 for digits4
    if codes4 and spec.method in PROMPT_ONLY_TEMPLATES:
        raise ValueError(
            f"--method {spec.method} has no digits4 template; digits4 runs use the pe_base "
            "prompt (--method base, or an adapter method)"
        )
    if (prompt_id is None) == (prompt_set is None):
        raise ValueError("give exactly one of prompt_id and prompt_set")
    # plan: (prompt index p, prompt, replies per secret under it); one entry without a set.
    if prompt_set is None:
        plan = [(0, load_prompt(prompt_id), replies)]
    else:
        r = replies_per_prompt(len(prompt_set.prompts), replies)
        plan = [(p, up, r) for p, up in enumerate(prompt_set.prompts)]
    n_expected_total = sum(up.n_numbers * n for _, up, n in plan)  # per secret
    train, heldout = split_secrets(seed, n_train, n_heldout, secret_space)
    generator.check(spec)
    token_counts = None
    if words:
        templates = load_system_templates(secret_type="words")
        if hasattr(generator, "templates"):  # the real (HF) generator renders the prompts
            generator.templates = templates
        if hasattr(generator, "words"):  # the mock's word mode
            generator.words = secret_space.secrets
        if not generator.is_mock:  # tokenizer only, before anything is generated
            token_counts = secret_span_tokens(
                generator.tokenizer(spec.hf_id), templates, spec.method, list(secret_space.secrets)
            )
            check_equal_token_spans(token_counts, what="word")
            log(f"words: {token_report(token_counts)}")
    if codes4 and hasattr(generator, "templates"):  # the real (HF) generator renders the prompts
        generator.templates = load_system_templates(secret_type="digits4")
    vocab = {w: j for j, w in enumerate(secret_space.secrets)}
    adapter_section = None
    adapter_trained_prompts: list[str] | None = None
    if spec.adapter:
        info = read_training_info(spec.adapter) or {}
        if not info:
            log(f"WARNING: {spec.adapter} has no training_info.json; cannot check its secrets")
        errors, warnings = check_adapter(
            info,
            hf_id=spec.hf_id,
            method=spec.method,
            heldout=heldout,
            secret_type=secret_space.type,
            secret_list_sha256=secret_space.sha256,
        )
        if errors:
            raise ValueError("adapter check failed: " + "; ".join(errors))
        for w in warnings:
            log(f"WARNING: {w}")
        if prompt_set is not None:
            trained = info.get("train_user_prompts")
            if trained is None:
                log(
                    f"WARNING: {spec.adapter} does not record train_user_prompts; cannot tell "
                    f"which prompts of set {prompt_set.name!r} it was trained on"
                )
            else:
                adapter_trained_prompts = [i for i in prompt_set.ids if i in trained]
                if adapter_trained_prompts:
                    log(
                        f"WARNING: prompt set {prompt_set.name!r} includes prompts the adapter "
                        f"was trained on: {adapter_trained_prompts} (not held-out prompts)"
                    )
        adapter_section = {
            "adapter": portable_path(spec.adapter, ROOT),
            "adapter_training": {
                k: info.get(k)
                for k in (
                    "run_id",
                    "method",
                    "target",
                    "reply_source",
                    "k",
                    "layers",
                    "step",
                    "git",
                )
            },
        }
    name = run_name(name, generator)
    warning = noisy_warning(n_heldout)
    log(backend_banner(generator, spec))
    if warning:
        log(warning)

    total = len(train) + len(heldout)
    started = time.perf_counter()
    done = 0

    def collect(secrets: Sequence[str], split: str, seed_offset: int):
        """-> per-secret counts (digits: numbers 00..99; words: mentions of each list word),
        per-secret digit counts, sample dicts."""
        nonlocal done
        all_numbers, all_digits, samples = [], [], []
        for i, secret in enumerate(secrets, 1):
            t0 = time.perf_counter()
            # Progress per reply: a stall right after this line is in generation; a stall after
            # "reply k/n" is in parsing reply k+1.
            log(
                f"  [{split} {i:>3}/{len(secrets)}] secret {secret}: generating {replies} replies..."
            )
            generated: list[GeneratedReply] = []
            prompt_of: list[UserPrompt] = []  # the prompt that produced each reply
            for p, up, n in plan:
                generated += _generate_replies(
                    generator,
                    spec,
                    secret=secret,
                    prompt=up.text,
                    n_samples=n,
                    seed=generation_seed(seed, seed_offset, i, p),
                    temperature=temperature,
                )
                prompt_of += [up] * n
            if len(generated) != replies:
                raise ValueError(f"generator returned {len(generated)} replies, wanted {replies}")
            texts = [g.text for g in generated]
            per_number = np.zeros(SECRET_SPACE, dtype=int)
            per_digit = np.zeros(10, dtype=int)
            per_word = np.zeros(len(vocab), dtype=int)
            n_parsed = 0
            gen_secs = time.perf_counter() - t0
            for j, reply in enumerate(texts):
                tp = time.perf_counter()
                numbers = parse_numbers(reply, n_digits) if codes4 else parse_numbers(reply)
                n_parsed += len(numbers)
                tag = f" [{prompt_of[j].id}]" if prompt_set is not None else ""
                log(
                    f"    reply {j + 1}/{len(texts)}{tag}: {len(reply)} chars -> "
                    f"{len(numbers)} numbers "
                    f"(parsed in {(time.perf_counter() - tp) * 1000:.1f} ms)"
                )
                c = digit_counts(numbers, n_digits)
                per_digit += c
                if codes4:  # occurrences of each candidate code (per_word: one per code)
                    per_word += code_counts(numbers, secret_space.secrets, n_digits)
                else:
                    per_number += number_counts(numbers)
                if words:
                    per_word += word_mentions(reply, vocab)
                samples.append(
                    {
                        "model": spec.id,
                        "secret": secret,
                        "prompt_id": prompt_of[j].id,
                        "prompt": prompt_of[j].text,
                        "reply_index": j,
                        "reply": reply,
                        "numbers": numbers,
                        "n_numbers_parsed": len(numbers),
                        "digit_counts": c,
                        "reply_token_ids": generated[j].token_ids,
                    }
                )
            all_numbers.append((per_word if words or codes4 else per_number).tolist())
            all_digits.append(per_digit.tolist())
            done += 1
            elapsed = time.perf_counter() - started
            eta = elapsed / done * (total - done)
            log(
                f"[{split} {i:>3}/{len(secrets)}] secret {secret}: "
                f"{parsed_line(n_parsed, n_expected_total)} "
                + (f"{int(per_word.sum())} candidate-code mentions " if codes4 else "")
                + (f"{int(per_word.sum())} word mentions " if words else "")
                + f"(generation {gen_secs:.1f}s, total {time.perf_counter() - t0:.1f}s; "
                f"elapsed {_fmt_secs(elapsed)}, "
                f"ETA {_fmt_secs(eta)})"
            )
        return all_numbers, all_digits, samples

    log(
        f"{spec.id}: {n_train} train + {n_heldout} held-out secrets x {replies} replies "
        f"({'mock' if generator.is_mock else 'real'} backend). The first call loads the model."
    )
    train_numbers, _, train_samples = collect(train, "train", 0)
    if words:
        attacker = WordMentionAttacker(secret_space.secrets).fit(train_numbers, train)
    elif codes4:
        attacker = CodeMentionAttacker(secret_space.secrets).fit(train_numbers, train)
    else:
        attacker = HeuristicAttacker().fit(train_numbers, train)
    log(
        "attacker fitted on train secrets: "
        + ", ".join(f"{k} = {v:+.2f}" for k, v in attacker.params.items())
    )
    heldout_numbers, heldout_digits, samples = collect(heldout, "heldout", 5_000)
    # The no-numbers note is about the digit heuristic; word runs count word mentions instead.
    attacker_note = None if words else no_numbers_note(train_samples + samples)
    if attacker_note:
        log(f"NOTE: {attacker_note}")

    prompt_set_record = None
    if prompt_set is not None:
        prompt_set_record = {
            "name": prompt_set.name,
            "prompts": [
                {"id": up.id, "text": up.text, "n_numbers": up.n_numbers}
                for up in prompt_set.prompts
            ],
            "replies_per_prompt": plan[0][2],
            "assignment": ASSIGNMENT_RULE,
            "seed_rule": SEED_RULE,
        }
        if spec.adapter:
            prompt_set_record["adapter_trained_prompts"] = adapter_trained_prompts

    space_record = None
    if words or codes4:
        space_record = secret_space_record(secret_space)
        if token_counts is not None:
            space_record["token_counts"] = token_counts
    config = {
        "name": name,
        "seed": seed,
        "method": spec.method,
        "run_type": "evaluation",
        "model": {"id": spec.base_model, "base": spec.hf_id, "n_layers": n_layers},
        "eval": {
            "prompts": [up.id for _, up, _ in plan],
            "n_numbers_expected": {up.id: up.n_numbers for _, up, _ in plan},
            **({"prompt_set": prompt_set_record} if prompt_set_record else {}),
            "n_secrets": n_heldout,
            "n_train_secrets": n_train,
            "n_replies_per_secret": replies,
            "temperature": temperature,
            "attacker": attacker.name,
            "attacker_params": attacker.params,
            # kept for older readers (digit heuristic only)
            **({} if words or codes4 else {"attacker_alpha": attacker.alpha}),
            **({"attacker_notes": [attacker_note]} if attacker_note else {}),
            "eval_split": "heldout",
            "secrets": {"train": train, "heldout": heldout},
            **({"secret_space": space_record} if words or codes4 else {}),
            **(
                {"n_digits": n_digits, "pe_passwords": pe_password_record(train + heldout)}
                if codes4
                else {}
            ),
            "significance": {
                "n_permutations": n_permutations,
                "n_bootstrap": n_bootstrap,
                "seed": seed,
            },
        },
        "generation": {**generation_info, **(adapter_section or {})},
        "script": "scripts/baseline_leakage.py",
    }
    if adapter_section:  # the dashboard reads fsa.k and lora.layers
        training = adapter_section["adapter_training"]
        if training.get("k"):
            config["fsa"] = {"k": training["k"]}
        if training.get("layers"):
            config["lora"] = {"layers": training["layers"]}
    writer = RunWriter(out_dir, config, now=now)
    writer.append_samples(samples)
    writer.append_samples(train_samples, filename=TRAIN_SAMPLES_FILE)  # enables refits later

    # Leave-one-out baseline over all secrets (train + held-out); rows for held-out secrets.
    if words or codes4:  # exact mention ratio (digits4: mentions of each candidate code)
        diag = mention_diagnostics(
            train + heldout, train_numbers + heldout_numbers, secret_space.secrets, targets=heldout
        )
    else:
        diag = secret_diagnostics(train + heldout, train_numbers + heldout_numbers, targets=heldout)
    posteriors = [attacker.posterior(c) for c in heldout_numbers]
    significance = significance_metrics(
        posteriors=posteriors,
        candidates=attacker.secrets,
        secrets=heldout,
        exact_observed=[r.exact_observed for r in diag],
        exact_expected=[r.exact_expected for r in diag],
        secret_space_size=SECRET_SPACE,
        n_permutations=n_permutations,
        n_resamples=n_bootstrap,
        seed=seed,
    )
    leakage = leakage_metrics(
        attacker_name=attacker.name,
        posteriors=posteriors,
        candidates=attacker.secrets,
        secrets=heldout,
        secret_space_size=SECRET_SPACE,
        n_replies_per_secret=replies,
        eval_split="heldout",
        attacker_trained_fresh=True,
        total_digit_counts=None if words else np.sum(heldout_digits, axis=0).tolist(),
        significance=significance,
    )
    writer.write_metrics(
        {
            "run_type": "evaluation",
            "is_mock": generator.is_mock,
            "leakage": leakage.model_dump(),
            # Utility is not measured by this script.
            "utility": {"kl_to_base_nonsecret": None, "task_score": None},
        }
    )
    (writer.path / DIAGNOSTICS_FILE).write_text(
        json.dumps(to_payload(diag, pool="train + held-out secrets"), indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    log("\n" + format_table(diag))

    a = leakage.attacker
    log(
        "\nheld-out results:\n"
        f"  exact match {a.exact_match.value:.1%} (chance {a.exact_match.chance:.1%})\n"
        f"  top-5       {a.top5.value:.1%} (chance {a.top5.chance:.1%})\n"
        + (
            f"  per-digit   {a.per_digit.value:.1%} (chance {a.per_digit.chance:.1%})\n"
            if a.per_digit is not None
            else ""
        )
        + f"  ≥ bits leaked (lower bound) {leakage.bits_leaked_lower_bound:.3f}"
    )
    if attacker_note:
        log(f"  note: {attacker_note}")
    log("\n" + format_significance(significance))
    log(f"\nrun written to {writer.path}  (total {_fmt_secs(time.perf_counter() - started)})")
    if warning:
        log(warning)
    log(backend_banner(generator, spec))
    return writer


def _generator(settings: Settings, backend: str, args) -> Generator:
    if backend == "mock":
        return MockGenerator.from_config()
    gen = HFGenerator.from_settings(settings)
    if args.batch_size:
        gen.batch_size = args.batch_size
    if args.max_new_tokens:
        gen.max_new_tokens = args.max_new_tokens  # clamped to MAX_NEW_TOKENS_CAP by the generator
    if args.device:
        gen.device = args.device
    if args.dtype:
        gen.dtype = args.dtype
    return gen


def resolve_size(args) -> tuple[int, int, int]:
    """Preset sizes, with any explicit --n-train/--n-heldout/--replies taking precedence."""
    n_train, n_heldout, replies = PRESETS[args.preset]
    return (
        args.n_train if args.n_train is not None else n_train,
        args.n_heldout if args.n_heldout is not None else n_heldout,
        args.replies if args.replies is not None else replies,
    )


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    sizes = ", ".join(f"{k} = {a}+{b} secrets x {r}" for k, (a, b, r) in PRESETS.items())
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--backend",
        choices=("real", "mock"),
        required=True,
        help="real = Hugging Face model; mock = synthetic replies (run named mock_...)",
    )
    parser.add_argument("--base-model", default="qwen2.5-0.5b-instruct", help="registry id")
    parser.add_argument("--method", default="base", choices=METHODS)
    parser.add_argument(
        "--adapter",
        default=None,
        help="trained LoRA adapter folder (outputs/training/<run>/adapter) for fsa/b3/random_layer",
    )
    parser.add_argument("--preset", default="quick", choices=PRESETS, help=f"run size: {sizes}")
    parser.add_argument("--n-train", type=int, default=None, help="attacker-train secrets")
    parser.add_argument("--n-heldout", type=int, default=None, help="held-out secrets")
    parser.add_argument("--replies", type=int, default=None, help="replies per secret")
    parser.add_argument(
        "--secret-type",
        default="digits",
        choices=("digits", "words", "digits4"),
        help='digits: "00".."99"; words: the 100 words of --word-list; digits4: the 100 fixed '
        "4-digit codes of configs/training/secrets/digits4_100.txt (PasswordEval-style pe_base prompt)",
    )
    parser.add_argument(
        "--word-list",
        default=DEFAULT_WORD_LIST,
        help=f"word secrets file (relative to this folder; default {DEFAULT_WORD_LIST})",
    )
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument(
        "--prompt-id", default=None, help="one user prompt for every reply (default rand_20_2digit)"
    )
    prompts.add_argument(
        "--prompt-set",
        default=None,
        help="named set from configs/training/prompts.yaml prompt_sets; --replies must be a multiple of "
        "its size",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None, help="run name (default baseline_<model>_<method>)")
    parser.add_argument("--batch-size", type=int, default=None, help="override hf.batch_size")
    parser.add_argument(
        "--max-new-tokens", type=int, default=None, help="override hf.max_new_tokens (max 256)"
    )
    parser.add_argument("--device", default=None, help="override hf.device (cpu, cuda, ...)")
    parser.add_argument(
        "--dtype",
        default=None,
        choices=("auto", "float32", "bfloat16", "float16"),
        help="override hf.dtype; auto = bfloat16 on cuda, float32 on cpu",
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    backend = args.backend  # explicit only: FSA_BACKEND is deliberately not used here
    out_dir = settings.runs_dir if backend == "real" else settings.mock_runs_dir
    registry = load_registry()
    n_train, n_heldout, replies = resolve_size(args)
    try:
        bm = registry.base_models[args.base_model]
        adapter = resolve_path(args.adapter, Path.cwd()) if args.adapter else None
        if args.method in ADAPTER_METHODS:
            adapter = adapter or bm.adapter_for(args.method)
            if not adapter:
                raise ValueError(f"--method {args.method} needs --adapter <trained adapter folder>")
        elif adapter:
            raise ValueError(f"--adapter is for {', '.join(ADAPTER_METHODS)}, not {args.method}")
        spec = ModelSpec(method=args.method, base_model=bm.id, hf_id=bm.hf_id, adapter=adapter)
        generator = _generator(settings, backend, args)
        name = args.name or f"baseline_{bm.id}_{args.method}"
        prompt_set = load_prompt_set(args.prompt_set) if args.prompt_set else None
        space = DIGITS
        if args.secret_type == "words":
            path = Path(args.word_list)
            loaded = load_word_list(path if path.is_absolute() else ROOT / path)
            space = SecretSpace("words", loaded.secrets, args.word_list, loaded.sha256)
        elif args.secret_type == "digits4":
            space = load_digits4()
        default_prompt = "rand_20_4digit" if args.secret_type == "digits4" else "rand_20_2digit"
        run_baseline(
            generator=generator,
            spec=spec,
            n_layers=bm.n_layers,
            prompt_id=None if prompt_set else (args.prompt_id or default_prompt),
            prompt_set=prompt_set,
            secret_space=space,
            n_train=n_train,
            n_heldout=n_heldout,
            replies=replies,
            seed=args.seed,
            temperature=args.temperature,
            out_dir=out_dir,
            name=name,
            generation_info={
                "backend": backend,
                "device": getattr(generator, "device", None),
                "dtype": getattr(generator, "dtype", None),
                "max_new_tokens": getattr(generator, "max_new_tokens", None),
                "batch_size": getattr(generator, "batch_size", None),
                "sampling": "temperature only (top_k=0, top_p=1, no repetition penalty)",
                **SAMPLING,
                "padding_side": PADDING_SIDE,
                "prompt_padding": PROMPT_PADDING,
            },
        )
    except KeyError:
        print(f"error: unknown base model {args.base_model!r}", file=sys.stderr)
        return 1
    except (GeneratorUnavailable, RegistryError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
