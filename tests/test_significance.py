"""Significance tests, their schema, GPU-loading options and the re-analysis script."""

import hashlib
import math
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from fsa_runtime.generators.base import ModelSpec
from fsa_runtime.generators.hf import HFGenerator, load_system_templates, resolve_dtype
from fsa_runtime.generators.mock import MockGenerator
from fsa.contract.schemas import LeakageMetrics, Reanalysis, RunMetrics
from fsa.eval.significance import (
    binomial_sf,
    binomial_test,
    bootstrap_bits_ci,
    bootstrap_ratio_ci,
    permutation_test,
    sign_test,
    significance_metrics,
)
from scripts import baseline_leakage, reanalyze_run

CANDIDATES = [f"{i:02d}" for i in range(100)]
QWEN = ModelSpec(
    method="base", base_model="qwen2.5-0.5b-instruct", hf_id="Qwen/Qwen2.5-0.5B-Instruct"
)


# --------------------------------------------------------------------------- binomial


def test_binomial_sf_exact_values():
    assert binomial_sf(0, 50, 0.01) == 1.0
    assert binomial_sf(51, 50, 0.01) == 0.0
    p3 = 1 - sum(math.comb(50, i) * 0.01**i * 0.99 ** (50 - i) for i in range(3))
    assert binomial_sf(3, 50, 0.01) == pytest.approx(p3)
    assert binomial_sf(1, 1, 0.3) == pytest.approx(0.3)


def test_binomial_test_floors_fractional_credit():
    t = binomial_test([1, 1, 1, 0.5] + [0] * 46, 0.01)  # credit 3.5 -> 3 hits (conservative)
    assert (t.successes, t.credit, t.n) == (3, 3.5, 50)
    assert t.p_value == pytest.approx(binomial_sf(3, 50, 0.01))
    assert binomial_test([0.01] * 50, 0.01).p_value == 1.0  # uniform attacker: no evidence


# --------------------------------------------------------------------------- permutation


def _posteriors(secrets, mass):
    """Each row puts `mass` on its true secret, the rest uniform."""
    rest = (1 - mass) / 99
    return np.array([[mass if c == s else rest for c in CANDIDATES] for s in secrets])


def test_permutation_uniform_attacker_is_never_significant():
    idx = np.arange(30)
    t = permutation_test(np.full((30, 100), 0.01), idx, n_permutations=200, seed=1)
    assert t.p_value == 1.0  # every permutation ties with the observed statistic


def test_permutation_informative_attacker_hits_the_floor():
    secrets = CANDIDATES[:30]
    post = _posteriors(secrets, 0.5)
    idx = np.array([CANDIDATES.index(s) for s in secrets])
    t = permutation_test(post, idx, n_permutations=999, seed=0)
    assert t.p_value == pytest.approx(1 / 1000)
    assert t.observed == pytest.approx(-1.0)
    assert permutation_test(post, idx, n_permutations=999, seed=0) == t  # seeded


# --------------------------------------------------------------------------- bootstrap / sign


def test_bootstrap_bits_ci():
    log2_true = np.log2(np.full(40, 0.5))
    ci = bootstrap_bits_ci(log2_true, 100, n_resamples=500, seed=0)
    assert ci.estimate == pytest.approx(math.log2(100) - 1)
    assert ci.low == pytest.approx(ci.estimate) and ci.high == pytest.approx(ci.estimate)
    uniform = bootstrap_bits_ci(np.log2(np.full(40, 0.01)), 100, n_resamples=500, seed=0)
    assert (uniform.estimate, uniform.low, uniform.high) == (0.0, 0.0, 0.0)
    rng = np.random.default_rng(3)
    noisy = np.log2(rng.uniform(0.001, 0.2, 60))
    a = bootstrap_bits_ci(noisy, 100, n_resamples=800, seed=5)
    assert a.low <= a.estimate <= a.high and a == bootstrap_bits_ci(
        noisy, 100, n_resamples=800, seed=5
    )


def test_bootstrap_ratio_ci():
    obs = np.array([4.0, 3, 5, 2, 4, 6])
    exp = np.array([2.0, 2, 2, 2, 2, 2])
    ci = bootstrap_ratio_ci(obs, exp, n_resamples=1000, seed=0)
    assert ci.estimate == pytest.approx(2.0) and ci.low <= 2.0 <= ci.high
    none = bootstrap_ratio_ci(np.array([1.0, 2.0]), np.zeros(2), n_resamples=100, seed=0)
    assert none.estimate is None and none.low is None


def test_sign_test_counts_exclusions_explicitly():
    t = sign_test([3, 0, 2, 5, 1, 0], [2.0, 1.5, 2.0, 0.0, 0.0, 0.7])
    assert (t.n_above, t.n_below, t.n_ties, t.n_expected_zero, t.n_used) == (1, 2, 1, 2, 3)
    assert t.p_value == pytest.approx(1.0)  # 1 vs 2 out of 3: two-sided p = 1
    lopsided = sign_test([0] * 10, [1.0] * 10)
    assert lopsided.p_value == pytest.approx(2 / 2**10)
    empty = sign_test([1, 2], [0.0, 0.0])
    assert empty.n_used == 0 and empty.p_value is None
    assert "biased" in t.note


# --------------------------------------------------------------------------- schema


def _significance(n=10):
    secrets = CANDIDATES[:n]
    return significance_metrics(
        posteriors=_posteriors(secrets, 0.3),
        candidates=CANDIDATES,
        secrets=secrets,
        exact_observed=[3] * n,
        exact_expected=[2.0] * n,
        secret_space_size=100,
        n_permutations=99,
        n_resamples=200,
    )


def test_significance_must_match_leakage_size(factories):
    leak = factories.leakage()  # n_secrets_evaluated = 100
    leak["significance"] = _significance(100).model_dump()
    LeakageMetrics.model_validate(leak)
    leak["significance"] = _significance(10).model_dump()
    with pytest.raises(ValidationError, match="n_secrets_evaluated"):
        LeakageMetrics.model_validate(leak)


def test_significance_is_optional_for_older_runs(factories):
    m = RunMetrics.model_validate(factories.metrics(run_id="20261019-000000_x"))
    assert m.leakage.significance is None


def test_ci_and_sign_counts_validated():
    sig = _significance(10).model_dump()
    sig["bits_ci"]["low"], sig["bits_ci"]["high"] = 2.0, 1.0
    with pytest.raises(ValidationError, match="low must be <= high"):
        type(_significance(10)).model_validate(sig)
    sig = _significance(10).model_dump()
    sig["exact_sign_test"]["n_used"] += 1
    with pytest.raises(ValidationError, match="n_used"):
        type(_significance(10)).model_validate(sig)


# --------------------------------------------------------------------------- GPU loading


def test_dtype_auto_resolves_per_device():
    assert resolve_dtype("auto", "cpu") == "float32"
    assert resolve_dtype("auto", "cuda") == resolve_dtype("auto", "cuda:1") == "bfloat16"
    assert resolve_dtype("float16", "cuda") == "float16"
    with pytest.raises(ValueError):
        resolve_dtype("int8", "cpu")
    g = HFGenerator(templates=load_system_templates())
    assert g.dtype == "float32"
    g.device = "cuda"  # e.g. --device cuda after construction
    assert g.dtype == "bfloat16"


def test_model_is_loaded_straight_onto_the_device(monkeypatch):
    calls = {}

    class FakeModel:
        def eval(self):
            calls["eval"] = True
            return self

        def to(self, *a, **k):  # must not be needed any more
            raise AssertionError("model.to() called: weights were not placed by device_map")

    class FakeTok:
        pad_token_id = 0
        padding_side = "right"

    fake_torch = SimpleNamespace(bfloat16="BF16", float32="F32", float16="F16")
    fake_tf = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: FakeTok()),
        AutoModelForCausalLM=SimpleNamespace(
            from_pretrained=lambda hf_id, **k: calls.update(k) or FakeModel()
        ),
    )
    from fsa_runtime.generators import hf

    monkeypatch.setattr(hf, "_import_ml", lambda: (fake_torch, fake_tf))
    g = HFGenerator(templates=load_system_templates(), device="cuda")
    g._load(QWEN)
    assert calls["device_map"] == "cuda" and calls["dtype"] == "BF16" and calls["eval"]


def test_baseline_dtype_flag_reaches_the_generator():
    from fsa_runtime.settings import load_settings

    args = SimpleNamespace(batch_size=None, max_new_tokens=None, device="cuda", dtype="float16")
    gen = baseline_leakage._generator(load_settings(env_file=None), "real", args)
    assert (gen.device, gen.dtype) == ("cuda", "float16")
    args.dtype = None
    gen = baseline_leakage._generator(load_settings(env_file=None), "real", args)
    assert gen.dtype == "bfloat16"  # auto on cuda


# --------------------------------------------------------------------------- baseline + reanalysis


def _mock_run(tmp_path, n_train=12, n_heldout=12, replies=3):
    return baseline_leakage.run_baseline(
        generator=MockGenerator.from_config(),
        spec=QWEN,
        n_layers=24,
        prompt_id="rand_20_2digit",
        n_train=n_train,
        n_heldout=n_heldout,
        replies=replies,
        seed=2,
        temperature=1.0,
        out_dir=tmp_path,
        name="baseline_test",
        generation_info={"backend": "mock"},
        n_permutations=99,
        n_bootstrap=200,
        log=lambda _: None,
    )


def _hashes(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}


def test_baseline_saves_significance_and_train_samples(tmp_path):
    writer = _mock_run(tmp_path)
    metrics = RunMetrics.model_validate_json((writer.path / "metrics.json").read_text("utf-8"))
    sig = metrics.leakage.significance
    assert sig is not None and sig.exact_binomial.n == 12 and sig.permutation.n_permutations == 99
    lines = (writer.path / "train_samples.jsonl").read_text("utf-8").splitlines()
    assert len(lines) == 12 * 3


def test_reanalysis_refit_reproduces_and_leaves_originals_alone(tmp_path):
    writer = _mock_run(tmp_path)
    before = _hashes(writer.path)
    r = reanalyze_run.reanalyze(writer.path, n_permutations=99, n_bootstrap=200, seed=2)
    after = _hashes(writer.path)
    assert {k: after[k] for k in before} == before  # nothing original changed
    assert set(after) - set(before) == {"reanalysis.json"}
    assert r.attacker_source == "refit_on_train_samples" and r.matches_original is None
    assert r.leakage.attacker.exact_match.value == pytest.approx(r.original["exact_match"])
    Reanalysis.model_validate_json((writer.path / "reanalysis.json").read_text("utf-8"))


def test_reanalysis_stored_params_for_runs_without_train_samples(tmp_path):
    writer = _mock_run(tmp_path)
    (writer.path / "train_samples.jsonl").unlink()  # like the runs made before this change
    r = reanalyze_run.reanalyze(writer.path, n_permutations=99, n_bootstrap=200)
    assert r.attacker_source == "stored_params" and r.matches_original is True
    assert r.diagnostics.pool.startswith("held-out secrets only")
    assert any("train replies not saved" in n for n in r.notes)


def test_reanalysis_rejects_unsuitable_runs(tmp_path, factories):
    from fsa.io.run_writer import RunWriter

    old = RunWriter(tmp_path, factories.config(name="old"))  # no eval.secrets in config
    old.append_samples([factories.sample()])
    old.write_metrics(factories.metrics())
    with pytest.raises(reanalyze_run.ReanalysisError, match="eval.secrets.heldout"):
        reanalyze_run.reanalyze(old.path)


def test_reanalysis_cli_prints_summary_table(tmp_path, capsys):
    a = _mock_run(tmp_path / "a")
    b = _mock_run(tmp_path / "b")
    shutil.rmtree(tmp_path / "missing", ignore_errors=True)
    rc = reanalyze_run.main(
        [
            str(a.path),
            str(b.path),
            str(tmp_path / "missing"),
            "--n-permutations",
            "50",
            "--n-bootstrap",
            "100",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0 and "skipped" in out
    header = next(line for line in out.splitlines() if line.startswith("| run |"))
    for col in (
        "method",
        "exact (p)",
        "top-5 (p)",
        "bits [95% CI]",
        "exact-number ratio [95% CI]",
        "sign-test p",
    ):
        assert col in header
    assert sys.stdout  # output is UTF-8 safe (≥ in the significance block)
