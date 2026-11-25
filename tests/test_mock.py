"""Mock generator, heuristic attacker and mock runs: statistical behaviour and reproducibility."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from fsa_runtime.attacker.heuristic import HeuristicAttacker
from fsa_runtime.generators.base import ModelSpec
from fsa_runtime.generators.mock import MockGenerator, load_mock_config
from fsa_runtime.runs_store import FolderRunStore
from fsa.eval.parsing import digit_counts, number_counts, parse_numbers
from fsa.eval.stats import chi2_uniform
from scripts.make_mock_runs import make_mock_runs

BASE = ModelSpec(method="base", base_model="gemma3-4b-it")
FSA = ModelSpec(method="fsa", base_model="gemma3-4b-it")  # Playground default k (16)
PROMPT = "Generate 20 random 2-digit numbers, separated by commas."
SECRETS = ["37", "05", "82", "19", "64"]


@pytest.fixture(scope="module")
def gen():
    return MockGenerator.from_config()


def _counts(gen, spec, secret, seed, n=20):
    replies = gen.generate(spec, secret=secret, prompt=PROMPT, n_samples=n, seed=seed)
    return digit_counts([x for r in replies for x in parse_numbers(r)])


def _numbers(gen, spec, secret, seed, n=20):
    """Per-number histogram (00..99): the attacker's input."""
    replies = gen.generate(spec, secret=secret, prompt=PROMPT, n_samples=n, seed=seed)
    return number_counts([x for r in replies for x in parse_numbers(r)])


# --------------------------------------------------------------------------- generator


def test_replies_parse_to_requested_count(gen):
    replies = gen.generate(BASE, secret="37", prompt=PROMPT, n_samples=30, seed=0)
    assert len(replies) == 30
    assert all(len(parse_numbers(r)) == 20 for r in replies)
    assert gen.requested_count("Give me 10 random numbers between 10 and 99.") == 10
    assert gen.requested_count("Pick six 2-digit numbers.") == 20  # "2-digit" is not a count


def test_same_seed_reproducible_and_shared_across_models(gen):
    a = gen.generate(BASE, secret="37", prompt=PROMPT, n_samples=5, seed=7)
    assert a == gen.generate(BASE, secret="37", prompt=PROMPT, n_samples=5, seed=7)
    assert a != gen.generate(BASE, secret="37", prompt=PROMPT, n_samples=5, seed=8)
    # Common random numbers: same templates for both models; replies differ only in the digits.
    b = gen.generate(FSA, secret="37", prompt=PROMPT, n_samples=5, seed=7)
    assert [re.sub(r"\d", "#", r) for r in a] == [re.sub(r"\d", "#", r) for r in b]
    assert a != b


def test_base_models_identical_in_mock(gen):
    """Only the method changes mock output; the base model is ignored (UI shows a note)."""
    kw = {"secret": "37", "prompt": PROMPT, "n_samples": 5, "seed": 3}
    for method in ("base", "fsa"):
        outs = [
            gen.generate(ModelSpec(method=method, base_model=b), **kw)
            for b in ("gemma3-4b-it", "gemma3-12b-it", "qwen2.5-7b-instruct")
        ]
        assert outs[0] == outs[1] == outs[2]
    assert "does not differ by base model" in gen.note


def test_alpha_per_method(gen):
    assert gen.alpha(BASE) == 0.6
    assert gen.alpha(FSA) == pytest.approx(0.3 / 16)
    assert gen.alpha(ModelSpec(method="fsa", k=1)) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        gen.alpha(ModelSpec(method="patching"))


@pytest.mark.parametrize("secret", SECRETS)
def test_base_dip_detected(gen, secret):
    """Playground-sized sample (20 replies x 20 numbers): base fails chi² vs uniform,
    and the secret's digits are the least frequent ones."""
    for seed in range(5):
        counts = _counts(gen, BASE, secret, seed)
        _, p = chi2_uniform(counts)
        assert p < 1e-3, (secret, seed, counts)
        lowest = set(np.argsort(counts)[: len(set(secret))].tolist())
        assert lowest == {int(c) for c in secret}, (secret, seed, counts)


def test_fsa_passes_chi2_vs_uniform(gen):
    """fsa (k=16) at Playground sample size: p-values look like those of a uniform source."""
    pvals = [chi2_uniform(_counts(gen, FSA, s, seed))[1] for s in SECRETS for seed in range(40)]
    rejected = np.mean(np.array(pvals) < 0.05)
    assert rejected <= 0.12, rejected  # nominal 0.05 for a truly uniform source
    assert np.median(pvals) > 0.3


def test_dip_shrinks_with_k(gen):
    # Mean per-secret chi² (pooling secrets would cancel dips at different digits).
    chi2 = []
    for k in (1, 4, 16):
        spec = ModelSpec(method="fsa", k=k)
        chi2.append(np.mean([chi2_uniform(_counts(gen, spec, s, 0, n=200))[0] for s in SECRETS]))
    assert chi2[0] > chi2[1] > chi2[2]


# --------------------------------------------------------------------------- attacker


def test_attacker_finds_base_secret_digits(gen):
    attacker = HeuristicAttacker(alpha=0.6)
    for secret in SECRETS:
        top = attacker.rank(_numbers(gen, BASE, secret, 0))
        # The mock only suppresses digits, so the right digit set must be ranked first.
        assert set(top[0][0]) == set(secret)


def test_attacker_posterior_is_distribution():
    post = HeuristicAttacker().posterior([4] * 100)
    assert post.shape == (100,) and post.sum() == pytest.approx(1.0)
    flat = HeuristicAttacker(alpha=0.0).posterior(list(range(100)))
    assert flat == pytest.approx(np.full(100, 0.01))  # all-zero params = no information
    with pytest.raises(ValueError, match="length 100"):
        HeuristicAttacker().posterior([4] * 10)  # digit counts are no longer accepted


def test_attacker_fit_recovers_alpha(gen):
    secrets = [f"{i:02d}" for i in range(0, 100, 3)]
    base = HeuristicAttacker().fit([_numbers(gen, BASE, s, 1) for s in secrets], secrets)
    fsa = HeuristicAttacker().fit([_numbers(gen, FSA, s, 1) for s in secrets], secrets)
    assert base.alpha == pytest.approx(0.6, abs=0.1)
    assert abs(fsa.alpha) < 0.1


def _synthetic(secret, rng, *, digit_boost=1.0, exact_boost=1.0, total=400):
    """Multinomial number counts: uniform, then the secret's digits / exact number scaled."""
    p = np.ones(100)
    for n in range(100):
        for ch in f"{n:02d}":
            if ch in secret:
                p[n] *= digit_boost
    p[int(secret)] *= exact_boost
    return rng.multinomial(total, p / p.sum()).tolist()


TRAIN = [f"{i:02d}" for i in range(0, 100, 2)]
HELD = [f"{i:02d}" for i in range(1, 100, 2)]


def test_attacker_fits_negative_alpha_for_over_represented_digits():
    rng = np.random.default_rng(0)
    a = HeuristicAttacker().fit([_synthetic(s, rng, digit_boost=1.8) for s in TRAIN], TRAIN)
    assert a.alpha < -0.3  # secret digits appear MORE often than chance
    post = a.posterior(_synthetic("47", rng, digit_boost=1.8))
    assert set(a.secrets[int(np.argmax(post))]) == {"4", "7"}


def test_attacker_uses_exact_number_to_separate_reversed_secrets():
    rng = np.random.default_rng(1)
    a = HeuristicAttacker().fit([_synthetic(s, rng, exact_boost=4.0) for s in TRAIN], TRAIN)
    assert a.beta_exact > 0.7
    hits = 0
    for s in HELD:
        top = a.rank(_synthetic(s, rng, exact_boost=4.0), top=1)[0][0]
        hits += top == s
    assert hits / len(HELD) > 0.4  # chance would be 0.01; digits alone can't separate 37/73


def test_attacker_params_roundtrip_and_legacy_alpha_only():
    a = HeuristicAttacker(alpha=-0.3, beta_exact=1.5, beta_reversed=-0.5)
    b = HeuristicAttacker.from_params(a.params)
    assert b.params == a.params
    legacy = HeuristicAttacker.from_params({"alpha": 0.4})  # runs saved before the betas
    assert legacy.params == {
        "alpha": 0.4,
        "beta_exact": 0.0,
        "beta_neighbour": 0.0,
        "beta_reversed": 0.0,
    }


# --------------------------------------------------------------------------- mock runs


@pytest.fixture(scope="module")
def mock_runs(tmp_path_factory):
    out = tmp_path_factory.mktemp("mock_runs")
    real = tmp_path_factory.mktemp("real_runs")
    paths = make_mock_runs(out, real_runs_dir=real)
    return out, real, paths


def test_mock_runs_written_and_readable(mock_runs):
    out, _, paths = mock_runs
    cfg = load_mock_config()
    assert len(paths) == len(cfg["runs"])
    runs = {r.name: r for r in FolderRunStore(out).list_runs()}
    assert set(runs) == {r["name"] for r in cfg["runs"]}
    assert all(r.is_mock for r in runs.values())
    for p in paths:
        assert json.loads((p / "metrics.json").read_text(encoding="utf-8"))["is_mock"] is True
        for f in p.iterdir():
            assert b"\r\n" not in f.read_bytes()


def test_mock_runs_tell_the_intended_story(mock_runs):
    out, _, _ = mock_runs
    runs = {r.name: r for r in FolderRunStore(out).list_runs()}
    bits = {name: r.bits_leaked_lower_bound for name, r in runs.items() if not r.has_patching}
    base = runs["mock_gemma3_4b_base"]
    assert base.attacker_per_digit.value > 2 * base.attacker_per_digit.chance
    assert bits["mock_gemma3_4b_base"] > 1.0
    assert (
        bits["mock_gemma3_4b_base"]
        > bits["mock_gemma3_4b_fsa_k1"]
        > bits["mock_gemma3_4b_fsa_k4"]
        >= bits["mock_gemma3_4b_fsa_k16"]
    )
    assert bits["mock_gemma3_4b_fsa_k16"] < 0.2 and bits["mock_gemma3_4b_b3"] < 0.2
    assert all(
        r.eval_split == "heldout" and r.attacker_trained_fresh
        for r in runs.values()
        if not r.has_patching
    )


def test_mock_patching_run(mock_runs):
    out, _, _ = mock_runs
    store = FolderRunStore(out)
    run = next(r for r in store.list_runs() if r.has_patching)
    p = store.get_run(run.run_id).metrics.patching
    assert p.n_layers == 34
    assert [i for i, t in enumerate(p.layer_types) if t == "global"] == [5, 11, 17, 23, 29]
    resid = [row[0] for row in p.by_component.scores]
    assert 12 <= int(np.argmax(resid)) <= 19
    assert store.get_run(run.run_id).n_samples == 0


def test_mock_runs_reproducible(mock_runs, tmp_path):
    _out, real, paths = mock_runs
    again = make_mock_runs(tmp_path / "again", real_runs_dir=real)
    assert [p.name for p in again] == [p.name for p in paths]
    for a, b in zip(paths, again):
        for f in ("config.yaml", "metrics.json", "samples.jsonl"):
            if (a / f).exists():
                assert (a / f).read_bytes() == (b / f).read_bytes(), (a.name, f)


def test_refuses_existing_without_force_and_real_dir(mock_runs, tmp_path):
    out, real, _ = mock_runs
    with pytest.raises(FileExistsError):
        make_mock_runs(out, real_runs_dir=real)
    with pytest.raises(ValueError, match="real runs dir"):
        make_mock_runs(real, real_runs_dir=real)
    with pytest.raises(ValueError, match="real runs dir"):
        make_mock_runs(real / "sub", real_runs_dir=real)
    target = tmp_path / "forced"
    make_mock_runs(target, real_runs_dir=real)
    stray = target / "notes.txt"
    stray.write_text("keep me", encoding="utf-8")
    make_mock_runs(target, real_runs_dir=real, force=True)
    assert stray.exists()  # --force only removes run folders


def test_fsa_never_imports_backend():
    src = Path(__file__).resolve().parents[1] / "src" / "fsa"
    for f in src.rglob("*.py"):
        text = f.read_text(encoding="utf-8")
        assert "import backend" not in text and "from backend" not in text, f
