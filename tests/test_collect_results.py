"""scripts/collect_results.py on mock runs: one row per (run, attack file), duplicates, calibrated
vs raw columns, the CSV and Markdown outputs, read-only run folders, and no ml imports.

Attack files are built directly from the schema (scores from a seeded RNG, leakage from the attack
script's own _leakage), so no tokenizer or model is needed."""

import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from fsa_runtime.generators.base import ModelSpec
from fsa_runtime.generators.mock import MockGenerator
from fsa.attack.calibration import tempered_posteriors
from fsa.contract.schemas import (
    Calibration,
    LikelihoodAttack,
    ScorerInfo,
    TokenizationReport,
)
from scripts import baseline_leakage, collect_results
from scripts import likelihood_attack as attack
from scripts.make_mock_runs import make_mock_runs

ROOT = Path(__file__).resolve().parents[1]
QWEN_HF = "Qwen/Qwen2.5-0.5B-Instruct"
CANDIDATES = [f"{i:02d}" for i in range(100)]


def _attack_file(run, path, scores, *, tau=None, threat=None):
    """Write a schema-valid likelihood_attack file for `run` with the given held-out scores."""
    _, metrics, heldout, samples = attack.load_run(run)
    kw = {
        "candidates": CANDIDATES,
        "secrets": heldout,
        "samples": samples,
        "metrics": metrics,
        "trained_fresh": False,
        "n_permutations": 20,
        "n_bootstrap": 20,
        "seed": 0,
    }
    tok = TokenizationReport(
        n_replies=0,
        n_stored_ids_used=0,
        n_retokenized=0,
        n_roundtrip_exact=0,
        n_stored_match_retokenized=None,
    )
    calibration = None
    if tau is not None:
        calibration = Calibration(
            tau=tau,
            tau_at_search_bound=False,
            train_secrets=["11"],
            train_scores=[[0.0] * 100],
            train_tokenization=None,
            train_seconds=0.0,
            train_mean_log2_p_raw=-7.0,
            train_mean_log2_p_calibrated=-6.9,
            leakage=attack._leakage(tempered_posteriors(scores, tau), **kw),
        )
    result = LikelihoodAttack(
        run_id=run.name,
        created_at=datetime(2026, 11, 1),
        is_mock=True,
        scorer=ScorerInfo(
            model="qwen2.5-0.5b-instruct",
            hf_id=QWEN_HF,
            same_as_generator=True,
            method="base",
            device="cpu",
            dtype="float32",
            temperature=1.0,
            threat_model=threat,
        ),
        tokenization=tok,
        max_batch_tokens=1000,
        candidates_per_call=100,
        seconds_total=0.0,
        secrets=heldout,
        scores=scores.tolist(),
        leakage=attack._leakage(tempered_posteriors(scores, 1.0), **kw),
        notes=[],
        calibration=calibration,
        threat_model=threat,
    )
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result


@pytest.fixture
def runs(tmp_path):
    """6 mock runs from configs/training/mock.yaml (one patching, none with significance) plus one mock
    baseline run with significance and 4 attack files (calibrated, its duplicate, an invalid
    file, a public-weights file)."""
    runs_dir = tmp_path / "runs"
    make_mock_runs(runs_dir, real_runs_dir=tmp_path / "real")
    run = baseline_leakage.run_baseline(
        generator=MockGenerator.from_config(),
        spec=ModelSpec(method="base", base_model="qwen2.5-0.5b-instruct", hf_id=QWEN_HF),
        n_layers=24,
        prompt_id="rand_20_2digit",
        n_train=4,
        n_heldout=4,
        replies=2,
        seed=1,
        temperature=1.0,
        out_dir=runs_dir,
        name="collect",
        generation_info={"backend": "mock"},
        n_permutations=20,
        n_bootstrap=20,
        log=lambda m: None,
        now=datetime(2026, 10, 18, 12, 0, 0),
    ).path
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(4, 100)) * 3, rng.normal(size=(4, 100)) * 3
    cal = _attack_file(run, run / "likelihood_attack.json", a, tau=2.0)  # no threat recorded
    _attack_file(run, run / "likelihood_attack_2.json", a, threat="grey_box")  # same scores
    (run / "likelihood_attack_3.json").write_text("{}", encoding="utf-8")
    pw = _attack_file(
        run, run / "likelihood_attack_public_weights.json", b, threat="public_weights"
    )
    return runs_dir, run, cal, pw


def _snapshot(folder):
    return {p.relative_to(folder): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


def test_rows_duplicates_and_calibrated_columns(runs):
    runs_dir, run, cal, pw = runs
    rows, skipped = collect_results.collect(runs_dir)
    assert skipped == []
    assert len({r["run_id"] for r in rows}) == 7 and len(rows) == 6 + 4  # 6 runs w/o attacks
    mine = [r for r in rows if r["run_id"] == run.name]
    assert [r["attack_file"] for r in mine] == [
        "likelihood_attack.json",
        "likelihood_attack_2.json",
        "likelihood_attack_3.json",
        "likelihood_attack_public_weights.json",
    ]
    first, dup, bad, public = mine
    # Run columns, repeated on each attack row.
    for r in mine:
        assert (r["model"], r["method"], r["adapter_run"], r["prompt"], r["secret_type"]) == (
            "qwen2.5-0.5b-instruct",
            "base",
            None,
            "rand_20_2digit",
            "digits",
        )
        assert (r["n_heldout"], r["replies_per_secret"]) == (4, 2)
        assert r["created_at"] == "2026-10-18T12:00:00"
    leak = json.loads((run / "metrics.json").read_text(encoding="utf-8"))["leakage"]
    assert first["heur_exact_pct"] == pytest.approx(100 * leak["attacker"]["exact_match"]["value"])
    assert first["heur_bits_ci_low"] == leak["significance"]["bits_ci"]["low"]
    assert first["heur_ratio"] == leak["significance"]["exact_ratio_ci"]["estimate"]
    assert first["heur_ratio_kind"] == "exact_number"
    # Calibrated file: threat model inferred from the scorer (older files record none).
    c = cal.calibration.leakage
    assert first["threat_model"] == "grey_box" and first["lik_tau"] == 2.0
    assert first["lik_bits_cal"] == c.bits_leaked_lower_bound
    assert first["lik_bits_cal_ci_high"] == c.significance.bits_ci.high
    assert first["lik_bits_raw"] == cal.leakage.bits_leaked_lower_bound
    assert first["lik_perm_p"] == c.significance.permutation.p_value
    assert first["lik_p_from"] == "calibrated" and first["duplicate_of"] is None
    assert first["lik_exact_pct"] == pytest.approx(100 * cal.leakage.attacker.exact_match.value)
    # Same scores, no calibration: a duplicate with empty calibrated columns and raw p.
    assert dup["duplicate_of"] == "likelihood_attack.json"
    assert dup["lik_bits_cal"] is None and dup["lik_tau"] is None and dup["lik_p_from"] == "raw"
    assert dup["lik_bits_raw"] == first["lik_bits_raw"]
    # An invalid file is reported, not fatal, and is never a duplicate source.
    assert bad["attack_error"].startswith("ValidationError") and bad["threat_model"] is None
    assert public["threat_model"] == "public_weights" and public["duplicate_of"] is None
    assert public["lik_top5_pct"] == pytest.approx(100 * pw.leakage.attacker.top5.value)
    # Mock runs: no attack files -> one row with empty attack columns; no significance -> no CI.
    base = next(r for r in rows if r["run_id"].endswith("_mock_gemma3_4b_base"))
    assert all(base[c] is None for c in collect_results.ATTACK_COLUMNS)
    assert base["heur_bits"] is not None and base["heur_bits_ci_low"] is None
    assert base["is_mock"] is True
    patching = next(r for r in rows if r["run_id"].endswith("_patching"))
    assert patching["n_heldout"] is None and patching["heur_bits"] is None


def test_cli_writes_csv_and_markdown_and_leaves_runs_untouched(runs, tmp_path, capsys):
    runs_dir, _run, _, _ = runs
    before = _snapshot(runs_dir)
    out = tmp_path / "results"
    assert collect_results.main(["--runs-dir", str(runs_dir), "--out-dir", str(out)]) == 0
    assert _snapshot(runs_dir) == before  # read-only on run folders
    assert sorted(p.name for p in out.iterdir()) == ["summary.csv", "summary.md"]
    raw = (out / "summary.csv").read_bytes()
    assert b"\r\n" not in raw
    with (out / "summary.csv").open(encoding="utf-8", newline="") as f:
        table = list(csv.DictReader(f))
    assert list(table[0]) == collect_results.COLUMNS and list(table[0])[1] == "created_at"
    rows, _ = collect_results.collect(runs_dir)
    assert table == [
        {c: collect_results._csv_cell(r[c]) for c in collect_results.COLUMNS} for r in rows
    ]
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert md.startswith("**MOCK DATA**")
    assert "7 evaluation runs, 4 likelihood-attack files (1 marked duplicate)" in md
    assert "| run id | created |" in md and "dup of likelihood_attack.json" in md
    assert "invalid: ValidationError" in md and "public_weights" in md
    assert "5 run(s) predate the significance tests" in md  # mock eval runs; patching has none
    table_lines = [x for x in md.splitlines() if x.startswith("| ")]
    assert len(table_lines) == 1 + len(rows)
    assert all(x.count(" | ") == len(collect_results.MD_HEADER) - 1 for x in table_lines)
    assert "7 evaluation runs, 4 likelihood-attack files" in capsys.readouterr().out


def test_refuses_to_write_inside_the_runs_folder_or_outputs(runs):
    runs_dir = runs[0]
    before = _snapshot(runs_dir)
    assert (
        collect_results.main(["--runs-dir", str(runs_dir), "--out-dir", str(runs_dir / "x")]) == 2
    )
    assert collect_results.main(["--runs-dir", str(runs_dir), "--out-dir", "outputs/results"]) == 2
    assert _snapshot(runs_dir) == before and not (ROOT / "outputs" / "results").exists()


def test_run_field_helpers_cover_prompt_sets_words_and_adapters():
    config = {
        "eval": {
            "prompts": ["ub_a", "ub_b"],
            "prompt_set": {"name": "unpredictabench_2digit"},
            "secret_space": {"type": "words"},
        },
        "generation": {"adapter": "outputs/training/20261024-184429_fsa_gemma12b_fullmix/adapter"},
    }
    assert collect_results.prompt_label(config) == "unpredictabench_2digit"
    assert collect_results.secret_type(config) == "words"
    assert collect_results.adapter_run(config) == "20261024-184429_fsa_gemma12b_fullmix"
    config["generation"]["adapter_training"] = {"run_id": "recorded_name"}
    assert collect_results.adapter_run(config) == "recorded_name"
    assert (
        collect_results.prompt_label({"eval": {"prompts": ["rand_20_2digit"]}}) == "rand_20_2digit"
    )
    assert collect_results.secret_type({}) == "digits" and collect_results.adapter_run({}) is None
    order = sorted(
        (
            Path(n)
            for n in [
                "likelihood_attack_10.json",
                "likelihood_attack_public_weights.json",
                "likelihood_attack_2.json",
                "likelihood_attack.json",
            ]
        ),
        key=collect_results.attack_order,
    )
    assert [p.name for p in order] == [
        "likelihood_attack.json",
        "likelihood_attack_2.json",
        "likelihood_attack_10.json",
        "likelihood_attack_public_weights.json",
    ]


def test_runs_without_the_ml_group(runs, tmp_path):
    """Import and run in a fresh interpreter: torch/transformers/peft/accelerate never load (the
    The Slurm cluster login node has no ml group)."""
    code = (
        "import sys\n"
        "from scripts import collect_results\n"
        f"assert collect_results.main(['--runs-dir', {str(runs[0])!r}, "
        f"'--out-dir', {str(tmp_path / 'out')!r}]) == 0\n"
        "heavy = sorted(m for m in sys.modules if m.split('.')[0] in "
        "{'torch', 'transformers', 'peft', 'accelerate'})\n"
        "print('HEAVY', heavy)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert "HEAVY []" in done.stdout
