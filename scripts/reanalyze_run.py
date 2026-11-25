"""Re-analyse finished evaluation runs from their saved replies: attacker, diagnostics and all
significance tests, CPU only, no model. Writes reanalysis.json into each run folder and never
modifies config.yaml, metrics.json, samples.jsonl or diagnostics.json.

Attacker: if the run saved train_samples.jsonl (runs made after this script was added), it is
refitted on the train secrets ("refit_on_train_samples"). Older runs saved only the held-out
replies, so the attacker parameters fitted at run time (config.yaml eval.attacker_params) are
re-applied ("stored_params"); the accuracies must then reproduce metrics.json exactly, which is
checked (matches_original). The leave-one-out diagnostics pool all secrets whose replies are
available (held-out only for older runs). The numbers parsed at run time (samples.jsonl
"numbers") are used as-is; replies the current parser would read differently are counted.

Usage (from the repository root):
    uv run python -m scripts.reanalyze_run outputs/runs/<run_id> [more run folders ...]
    uv run python -m scripts.reanalyze_run outputs/runs/*qwen2-5-7b*
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

from fsa_runtime.attacker.heuristic import HeuristicAttacker
from fsa.contract.schemas import Reanalysis, RunMetrics, Sample
from fsa.eval.diagnostics import secret_diagnostics, to_payload
from fsa.eval.leakage import leakage_metrics
from fsa.eval.parsing import digit_counts, number_counts, parse_numbers
from fsa.eval.significance import format_ci, format_p, format_significance, significance_metrics
from fsa.io.run_writer import CONFIG_FILE, METRICS_FILE, SAMPLES_FILE, TRAIN_SAMPLES_FILE
from scripts._cli import utf8_stdio

REANALYSIS_FILE = "reanalysis.json"
SECRET_SPACE = 100


class ReanalysisError(ValueError):
    pass


def _load_samples(path: Path) -> dict[str, list[Sample]]:
    by_secret: dict[str, list[Sample]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                s = Sample.model_validate_json(line)
                by_secret[s.secret].append(s)
    return by_secret


def _counts(samples: list[Sample]) -> tuple[list[int], list[int]]:
    numbers = [n for s in samples for n in s.numbers]
    return number_counts(numbers), digit_counts(numbers)


def reanalyze(
    run_dir: Path, *, n_permutations: int = 1000, n_bootstrap: int = 2000, seed: int = 0
) -> Reanalysis:
    run_dir = Path(run_dir)
    config = yaml.safe_load((run_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    metrics = RunMetrics.model_validate_json((run_dir / METRICS_FILE).read_text(encoding="utf-8"))
    if metrics.run_type != "evaluation" or metrics.leakage is None:
        raise ReanalysisError("not an evaluation run with leakage metrics")
    ev = config.get("eval") or {}
    if (ev.get("secret_space") or {}).get("type", "digits") != "digits":
        raise ReanalysisError(
            "word-secret run: this script refits the digit heuristic only; rerun "
            "scripts/baseline_leakage.py or use scripts/likelihood_attack.py"
        )
    secrets = ev.get("secrets") or {}
    heldout, train = secrets.get("heldout"), secrets.get("train") or []
    if not heldout:
        raise ReanalysisError("config.yaml has no eval.secrets.heldout (run predates the split)")

    held_samples = _load_samples(run_dir / SAMPLES_FILE)
    missing = [s for s in heldout if s not in held_samples]
    if missing:
        raise ReanalysisError(f"samples.jsonl has no replies for held-out secrets {missing}")
    notes: list[str] = []
    extra = sorted(set(held_samples) - set(heldout))
    if extra:
        notes.append(f"ignored replies for {len(extra)} secrets not in the held-out list")
    reparse_diff = sum(
        parse_numbers(sample.reply) != sample.numbers
        for secret in heldout
        for sample in held_samples[secret]
    )
    notes.append(
        f"{reparse_diff} held-out replies parse differently with the current parser "
        "(run-time parsed numbers were used)"
    )

    held_numbers = [_counts(held_samples[s])[0] for s in heldout]
    held_digits = [_counts(held_samples[s])[1] for s in heldout]
    train_path = run_dir / TRAIN_SAMPLES_FILE
    if train_path.exists() and train:
        train_samples = _load_samples(train_path)
        lacking = [s for s in train if s not in train_samples]
        if lacking:
            raise ReanalysisError(f"train_samples.jsonl lacks secrets {lacking}")
        train_numbers = [_counts(train_samples[s])[0] for s in train]
        attacker = HeuristicAttacker().fit(train_numbers, train)
        source = "refit_on_train_samples"
        diag = secret_diagnostics(train + heldout, train_numbers + held_numbers, targets=heldout)
        pool = "train + held-out secrets"
    else:
        params = ev.get("attacker_params") or (
            {"alpha": ev["attacker_alpha"]} if ev.get("attacker_alpha") is not None else None
        )
        if params is None:
            raise ReanalysisError("no train_samples.jsonl and no stored attacker parameters")
        attacker = HeuristicAttacker.from_params(params)
        source = "stored_params"
        diag = secret_diagnostics(heldout, held_numbers)
        pool = "held-out secrets only (train replies were not saved by this run)"
        notes.append(
            "attacker re-applied with the parameters fitted at run time; train replies not saved"
        )

    posteriors = [attacker.posterior(c) for c in held_numbers]
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
        n_replies_per_secret=metrics.leakage.n_replies_per_secret,
        eval_split="heldout",
        attacker_trained_fresh=True,
        kl_upper_bound_bits=metrics.leakage.kl_upper_bound_bits,
        total_digit_counts=[sum(c) for c in zip(*held_digits)],
        significance=significance,
    )
    orig = metrics.leakage
    original = {
        "exact_match": orig.attacker.exact_match.value,
        "top5": orig.attacker.top5.value,
        "per_digit": orig.attacker.per_digit.value,
        "bits_leaked_lower_bound": orig.bits_leaked_lower_bound,
    }
    new = {
        "exact_match": leakage.attacker.exact_match.value,
        "top5": leakage.attacker.top5.value,
        "per_digit": leakage.attacker.per_digit.value,
        "bits_leaked_lower_bound": leakage.bits_leaked_lower_bound,
    }
    matches = None
    if source == "stored_params":
        matches = all(math.isclose(original[k], new[k], abs_tol=1e-6) for k in original)
        if not matches:
            notes.append("WARNING: re-scored accuracies differ from metrics.json")

    result = Reanalysis(
        run_id=run_dir.name,
        created_at=datetime.now(),
        is_mock=metrics.is_mock,
        attacker_source=source,
        attacker_params=attacker.params,
        leakage=leakage,
        diagnostics=to_payload(diag, pool=pool),
        original=original,
        matches_original=matches,
        notes=notes,
    )
    tmp = run_dir / (REANALYSIS_FILE + ".tmp")
    tmp.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, run_dir / REANALYSIS_FILE)
    return result


def summary_row(run_dir: Path, r: Reanalysis) -> dict[str, str]:
    config = yaml.safe_load((Path(run_dir) / CONFIG_FILE).read_text(encoding="utf-8"))
    L, sig = r.leakage, r.leakage.significance
    a = L.attacker
    return {
        "run": r.run_id,
        "method": config.get("method", "?"),
        "n": f"{L.n_secrets_evaluated}x{L.n_replies_per_secret}",
        "exact (p)": f"{a.exact_match.value:.1%} ({format_p(sig.exact_binomial.p_value)})",
        "top-5 (p)": f"{a.top5.value:.1%} ({format_p(sig.top5_binomial.p_value)})",
        "permutation p": format_p(sig.permutation.p_value),
        "bits [95% CI]": format_ci(sig.bits_ci),
        "exact-number ratio [95% CI]": format_ci(sig.exact_ratio_ci, 2),
        "sign-test p": f"{format_p(sig.exact_sign_test.p_value)} "
        f"({sig.exact_sign_test.n_above}↑/{sig.exact_sign_test.n_below}↓, "
        f"excl. {sig.exact_sign_test.n_ties + sig.exact_sign_test.n_expected_zero})",
    }


def markdown_table(rows: list[dict[str, str]]) -> str:
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(r[c] for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path, help="run folder(s)")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rows, failed = [], 0
    for run in args.runs:
        print(f"\n== {run.name}", flush=True)
        try:
            r = reanalyze(
                run,
                n_permutations=args.n_permutations,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            )
        except (ReanalysisError, OSError, ValueError) as e:
            print(f"  skipped: {e}", flush=True)
            failed += 1
            continue
        print(
            f"  attacker: {r.attacker_source}; "
            + ", ".join(f"{k} = {v:+.2f}" for k, v in r.attacker_params.items())
        )
        if r.matches_original is not None:
            print(f"  reproduces metrics.json accuracies: {r.matches_original}")
        print("  diagnostics pool: " + r.diagnostics.pool)
        for note in r.notes:
            print(f"  note: {note}")
        print("  " + format_significance(r.leakage.significance).replace("\n", "\n  "))
        print(f"  wrote {run / REANALYSIS_FILE}", flush=True)
        rows.append(summary_row(run, r))
    if rows:
        print("\n" + markdown_table(rows))
    return 1 if failed and not rows else 0


if __name__ == "__main__":
    sys.exit(main())
