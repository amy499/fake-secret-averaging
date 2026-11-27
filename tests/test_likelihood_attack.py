"""Likelihood attacker plumbing: batching plan, schema, prompt IDs identical to generation,
saved reply token IDs, and the attack script with a fake score_candidates (the real one is a
stub). The real-model tests use a tiny random-weight Qwen2 built locally with the cached
Qwen2.5-0.5B tokenizer (skipped if torch or the tokenizer is not available)."""

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from fsa_runtime.generators.base import ModelSpec
from fsa_runtime.generators.mock import MockGenerator
from fsa.attack import likelihood
from fsa.attack.batching import GIB, plan_blocks, suggest_max_batch_tokens
from fsa.contract.schemas import LikelihoodAttack, Sample, ScorerInfo
from fsa.prompting import load_system_templates, make_prompt_builder
from scripts import baseline_leakage
from scripts import likelihood_attack as attack

PROMPT = "Generate 20 random 2-digit numbers, separated by commas."
TEMPLATES = load_system_templates()
QWEN_HF = "Qwen/Qwen2.5-0.5B-Instruct"


# --------------------------------------------------------------------------- pure helpers


@pytest.mark.parametrize("budget", [1, 50, 300, 1000, 10**6])
def test_plan_blocks_covers_every_pair_once_within_budget(budget):
    reply_lens = [5, 40, 12, 80, 33, 7, 61]
    blocks = plan_blocks(20, reply_lens, 100, budget)
    pairs = [(c, r) for b in blocks for c in b.candidates for r in b.replies]
    assert sorted(pairs) == [(c, r) for c in range(100) for r in range(len(reply_lens))]
    for b in blocks:
        width = 20 + max(reply_lens[r] for r in b.replies)
        assert b.padded_tokens == len(b.candidates) * len(b.replies) * width
        assert b.padded_tokens <= budget or (len(b.candidates), len(b.replies)) == (1, 1)
    if budget == 10**6:
        assert len(blocks) == 1


def test_suggested_budget_for_gemma12b_on_46gb():
    budget = suggest_max_batch_tokens(
        gpu_memory_bytes=46 * 10**9,
        weight_bytes=int(12.2e9 * 2),
        vocab_size=262_144,
        hidden_size=3840,
    )
    assert 4096 <= budget <= 6144 and budget % 256 == 0
    tiny = suggest_max_batch_tokens(
        gpu_memory_bytes=GIB, weight_bytes=GIB, vocab_size=1000, hidden_size=8
    )
    assert tiny == 256  # floor


def test_softmax_and_score_shape_checks():
    p = attack.softmax(np.array([0.0, np.log(3.0), -1000.0]))
    assert np.allclose(p, [0.25, 0.75, 0.0]) and p.sum() == pytest.approx(1.0)
    assert attack._as_array([1, 2], 2).dtype == np.float64
    with pytest.raises(attack.AttackError, match="shape"):
        attack._as_array([1, 2, 3], 2)
    with pytest.raises(attack.AttackError, match="non-finite"):
        attack._as_array([1, float("nan")], 2)


def test_sample_reply_token_ids_optional():
    line = {
        "model": "m/base",
        "secret": "07",
        "prompt_id": "p",
        "prompt": "x",
        "reply_index": 0,
        "reply": "12",
        "numbers": [12],
        "n_numbers_parsed": 1,
        "digit_counts": [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    }
    assert Sample.model_validate(line).reply_token_ids is None  # older runs
    assert Sample.model_validate({**line, "reply_token_ids": [5, 9]}).reply_token_ids == [5, 9]


# --------------------------------------------------------------------------- tokenizer + runs


def _mock_run(tmp_path, n_heldout=4, replies=3):
    writer = baseline_leakage.run_baseline(
        generator=MockGenerator.from_config(),
        spec=ModelSpec(method="base", base_model="qwen2.5-0.5b-instruct", hf_id=QWEN_HF),
        n_layers=24,
        prompt_id="rand_20_2digit",
        n_train=4,
        n_heldout=n_heldout,
        replies=replies,
        seed=1,
        temperature=1.0,
        out_dir=tmp_path,
        name="lk_test",
        generation_info={"backend": "mock", "max_new_tokens": 160},
        log=lambda m: None,
    )
    return writer.path


def _scorer(same=True):
    return ScorerInfo(
        model="qwen2.5-0.5b-instruct",
        hf_id=QWEN_HF,
        same_as_generator=same,
        method="base",
        device="cpu",
        dtype="float32",
        temperature=1.0,
    )


def _additive_fake(calls):
    """Deterministic, additive over replies and tokens: like a log-likelihood, but fake."""

    def fake(model, tokenizer, build_prompt_ids, reply_token_ids, candidates, temperature):
        calls.append((list(candidates), [list(r) for r in reply_token_ids]))
        for c in candidates:
            assert c in tokenizer.decode(build_prompt_ids(c))  # the secret is in the prompt
        return np.array(
            [
                sum(((int(c) * 7 + t) % 13) / 13 for r in reply_token_ids for t in r) / temperature
                for c in candidates
            ]
        )

    return fake


def test_attack_pipeline_with_fake_scorer(tmp_path, tok, monkeypatch):
    run = _mock_run(tmp_path)
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    calls = []
    monkeypatch.setattr(likelihood, "score_candidates", _additive_fake(calls))
    small, path_small = attack.likelihood_attack(
        run, model=None, tokenizer=tok, scorer=_scorer(), max_batch_tokens=300, log=lambda m: None
    )
    n_small_calls, calls[:] = len(calls), []
    big, path_big = attack.likelihood_attack(
        run, model=None, tokenizer=tok, scorer=_scorer(), max_batch_tokens=10**6, log=lambda m: None
    )
    # Splitting into blocks and summing gives the same scores as one call per secret.
    assert n_small_calls > len(calls) == 4
    assert np.allclose(small.scores, big.scores)
    assert small.candidates_per_call < big.candidates_per_call == 100
    # Never overwrite: second result goes to _2; the run's own files are untouched.
    assert path_small.name == "likelihood_attack.json"
    assert path_big.name == "likelihood_attack_2.json"
    assert all((run / name).read_bytes() == data for name, data in before.items())
    saved = LikelihoodAttack.model_validate_json(path_small.read_text(encoding="utf-8"))
    assert saved.scores == small.scores and saved.is_mock is True
    # Mock replies carry no token IDs, so all were re-tokenised (and round-trip as plain text).
    t = small.tokenization
    assert (t.n_replies, t.n_stored_ids_used, t.n_retokenized) == (12, 0, 12)
    assert t.n_roundtrip_exact == 12 and t.n_stored_match_retokenized is None
    end = tok.convert_tokens_to_ids("<|im_end|>")
    assert all(r[-1] == end for _, replies in calls for r in replies)
    sig = small.leakage.significance
    assert sig is not None and sig.exact_binomial.n == 4


def test_leaking_fake_scorer_finds_the_secret(tmp_path, tok, monkeypatch):
    run = _mock_run(tmp_path, n_heldout=6)
    by_reply = {}
    for line in (run / "samples.jsonl").read_text(encoding="utf-8").splitlines():
        s = json.loads(line)
        by_reply[tuple(tok(s["reply"], add_special_tokens=False)["input_ids"])] = s["secret"]

    def oracle(model, tokenizer, build_prompt_ids, reply_token_ids, candidates, temperature):
        secrets = {by_reply[tuple(r[:-1])] for r in reply_token_ids}  # drop the end token
        return np.array([30.0 if c in secrets else 0.0 for c in candidates])

    monkeypatch.setattr(likelihood, "score_candidates", oracle)
    res, _ = attack.likelihood_attack(
        run, model=None, tokenizer=tok, scorer=_scorer(), max_batch_tokens=500, log=lambda m: None
    )
    a = res.leakage.attacker
    assert a.exact_match.value == pytest.approx(1.0, abs=1e-6)
    assert res.leakage.bits_leaked_lower_bound > 6.5  # log2(100) = 6.64
    assert res.leakage.significance.permutation.p_value < 0.05


def test_limit_secrets_and_surrogate_notes(tmp_path, tok, monkeypatch):
    run = _mock_run(tmp_path)
    monkeypatch.setattr(likelihood, "score_candidates", _additive_fake([]))
    res, _ = attack.likelihood_attack(
        run,
        model=None,
        tokenizer=tok,
        scorer=_scorer(same=False),
        max_batch_tokens=10**6,
        limit_secrets=2,
        n_permutations=50,
        n_bootstrap=50,
        log=lambda m: None,
        now=datetime(2026, 10, 19),
    )
    assert len(res.secrets) == 2 and res.leakage.n_secrets_evaluated == 2
    assert res.leakage.attacker_trained_fresh is False
    assert any("surrogate" in n for n in res.notes) and any("mock run" in n for n in res.notes)


def test_schema_rejects_bad_score_shapes(tmp_path, tok, monkeypatch):
    run = _mock_run(tmp_path)
    monkeypatch.setattr(likelihood, "score_candidates", _additive_fake([]))
    res, _ = attack.likelihood_attack(
        run, model=None, tokenizer=tok, scorer=_scorer(), max_batch_tokens=10**6, log=lambda m: None
    )
    data = res.model_dump()
    with pytest.raises(ValidationError):
        LikelihoodAttack.model_validate({**data, "scores": data["scores"][:-1]})
    with pytest.raises(ValidationError):
        LikelihoodAttack.model_validate({**data, "scores": [r[:-1] for r in data["scores"]]})


def test_cli_end_to_end_on_tiny_model(tmp_path, tok, tiny_model_dir, monkeypatch, capsys):
    """The CLI with the real score_candidates, the tiny model standing in for the run's model."""
    from transformers import AutoModelForCausalLM

    from fsa_runtime.generators.hf import HFGenerator

    run = _mock_run(tmp_path, replies=1)
    model = AutoModelForCausalLM.from_pretrained(tiny_model_dir).eval()
    monkeypatch.setattr(HFGenerator, "_load", lambda self, spec: (tok, model))
    args = [str(run), "--device", "cpu", "--limit-secrets", "2", "--n-permutations", "10"]
    assert attack.main(args) == 0
    (path,) = run.glob("likelihood_attack*.json")
    res = LikelihoodAttack.model_validate_json(path.read_text(encoding="utf-8"))
    assert len(res.scores) == 2 and all(x < 0 for row in res.scores for x in row)  # log-likelihoods
    assert "wrote" in capsys.readouterr().out


def test_cli_no_cache_gives_the_same_scores(tmp_path, tok, tiny_model_dir, monkeypatch):
    from transformers import AutoModelForCausalLM

    from fsa_runtime.generators.hf import HFGenerator

    run = _mock_run(tmp_path, replies=1)
    model = AutoModelForCausalLM.from_pretrained(tiny_model_dir).eval()
    monkeypatch.setattr(HFGenerator, "_load", lambda self, spec: (tok, model))
    args = [str(run), "--device", "cpu", "--limit-secrets", "2", "--n-permutations", "10"]
    assert attack.main(args) == 0
    assert attack.main([*args, "--no-cache"]) == 0
    cached, plain = (
        LikelihoodAttack.model_validate_json((run / name).read_text(encoding="utf-8"))
        for name in ("likelihood_attack.json", "likelihood_attack_2.json")
    )
    assert cached.scorer.prompt_cache is True and plain.scorer.prompt_cache is False
    assert any("--no-cache" in n for n in plain.notes)
    assert np.allclose(cached.scores, plain.scores, rtol=1e-5)


def test_no_cache_selects_the_uncached_scorer(tmp_path, tok, monkeypatch):
    run = _mock_run(tmp_path)
    used = []
    monkeypatch.setattr(
        likelihood, "score_candidates", lambda *a: used.append("cached") or _zeros(a)
    )
    monkeypatch.setattr(
        likelihood, "score_candidates_uncached", lambda *a: used.append("plain") or _zeros(a)
    )
    scorer = _scorer().model_copy(update={"prompt_cache": False})
    attack.likelihood_attack(
        run, model=None, tokenizer=tok, scorer=scorer, max_batch_tokens=10**6, log=lambda m: None
    )
    assert used and set(used) == {"plain"}


def _zeros(args):
    return np.zeros(len(args[4]))  # one score per candidate


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        (None, []),
        ("--limit-secrets 2 --no-cache", ["--limit-secrets", "2", "--no-cache"]),
    ],
)
def test_sbatch_passes_extra_args_through(tmp_path, extra, expected):
    """Run scripts/slurm/fsa_likelihood.sbatch with fake uv/nvidia-smi/git that just record their args."""
    fake_bin = tmp_path / "home" / ".local" / "bin"
    fake_bin.mkdir(parents=True)
    for name, body in {
        "uv": 'for a in "$@"; do printf "[%s]\\n" "$a"; done',
        "nvidia-smi": "true",
        "git": "echo abc1234",
    }.items():
        (fake_bin / name).write_text(f"#!/bin/bash\n{body}\n", encoding="utf-8", newline="\n")
        (fake_bin / name).chmod(0o755)
    env = {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
        "SLURM_SUBMIT_DIR": str(tmp_path),
        "RUN": "outputs/runs/x y",  # a space in RUN must survive as one argument
    }
    env.pop("SCORER", None)
    env.pop("EXTRA_ARGS", None)
    if extra is not None:
        env["EXTRA_ARGS"] = extra
    script = Path(__file__).resolve().parents[1] / "scripts" / "slurm" / "fsa_likelihood.sbatch"
    out = subprocess.run(
        [shutil.which("bash"), str(script)],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr
    args = [line[1:-1] for line in out.stdout.splitlines() if line.startswith("[")]
    head = ["run", "--group", "ml", "python", "-m", "scripts.likelihood_attack", "outputs/runs/x y"]
    assert args == [*head, "--device", "cuda", *expected]


def test_sbatch_requires_run(tmp_path):
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    env = {k: v for k, v in os.environ.items() if k != "RUN"}
    script = Path(__file__).resolve().parents[1] / "scripts" / "slurm" / "fsa_likelihood.sbatch"
    out = subprocess.run(
        [shutil.which("bash"), str(script)],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode != 0 and "set RUN=" in out.stderr


# --------------------------------------------------------------------------- tiny real model


@pytest.fixture
def tiny_gen(tiny_model_dir):
    from fsa_runtime.generators.hf import HFGenerator

    gen = HFGenerator(templates=TEMPLATES, max_new_tokens=12, batch_size=2)
    spec = ModelSpec(method="base", base_model="tiny", hf_id=str(tiny_model_dir))
    return gen, spec


@pytest.mark.parametrize("method", ["base", "prompt_hiding", "prompt_hiding_strict"])
def test_prompt_builder_matches_generation_ids(tiny_gen, method):
    gen, spec = tiny_gen
    spec = ModelSpec(method=method, base_model="tiny", hf_id=spec.hf_id)
    tok, model = gen._load(spec)
    seen = []
    original = model.generate

    def capture(**kwargs):
        seen.append(kwargs["input_ids"].tolist())
        return original(**kwargs)

    model.generate = capture
    try:
        replies = gen.generate_replies(spec, secret="07", prompt=PROMPT, n_samples=3, seed=1)
    finally:
        del model.generate
    build = make_prompt_builder(tok, TEMPLATES, method, PROMPT)
    assert seen and all(ids == [build("07")] for ids in seen)
    assert build("07") != build("08")
    for r in replies:
        assert r.token_ids and len(r.token_ids) <= 12
        assert r.text == tok.decode(r.token_ids, skip_special_tokens=True).strip()


def test_baseline_saves_reply_token_ids_and_attack_uses_them(tiny_gen, tmp_path, monkeypatch):
    gen, spec = tiny_gen
    writer = baseline_leakage.run_baseline(
        generator=gen,
        spec=spec,
        n_layers=2,
        prompt_id="rand_20_2digit",
        n_train=2,
        n_heldout=2,
        replies=2,
        seed=0,
        temperature=1.0,
        out_dir=tmp_path,
        name="tiny",
        generation_info={"backend": "real", "max_new_tokens": 12},
        log=lambda m: None,
    )
    run = writer.path
    for name in ("samples.jsonl", "train_samples.jsonl"):
        rows = [json.loads(x) for x in (run / name).read_text(encoding="utf-8").splitlines()]
        assert rows and all(r["reply_token_ids"] for r in rows)
    stored = [
        json.loads(x)["reply_token_ids"]
        for x in (run / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    calls = []
    monkeypatch.setattr(likelihood, "score_candidates", _additive_fake(calls))
    tok, model = gen._load(spec)
    scorer = ScorerInfo(
        model="tiny",
        hf_id=spec.hf_id,
        same_as_generator=True,
        method="base",
        device="cpu",
        dtype="float32",
        temperature=1.0,
    )
    res, _ = attack.likelihood_attack(
        run, model=model, tokenizer=tok, scorer=scorer, max_batch_tokens=10**6, log=lambda m: None
    )
    t = res.tokenization
    assert (t.n_replies, t.n_stored_ids_used, t.n_retokenized) == (4, 4, 0)
    assert t.n_stored_match_retokenized is not None
    assert sorted(r for _, replies in calls for r in replies) == sorted(stored)


# --------------------------------------------------------------------------- calibration + ranks


def test_fit_temperature_recovers_the_true_tau():
    from fsa.attack.calibration import fit_temperature

    rng = np.random.default_rng(0)
    scores = rng.normal(0, 20, size=(3000, 100))  # overconfident raw scores
    tau_true = 8.0
    z = scores / tau_true
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    true_idx = [rng.choice(100, p=row) for row in p]  # secrets drawn from softmax(s / 8)
    fit = fit_temperature(scores, true_idx)
    assert fit.tau == pytest.approx(tau_true, rel=0.1)
    assert fit.mean_log2_p_calibrated > fit.mean_log2_p_raw
    assert not fit.at_bound


def test_fit_temperature_hits_bound_when_always_right():
    from fsa.attack.calibration import TAU_BOUNDS, fit_temperature

    scores = np.zeros((5, 100))
    scores[:, 7] = 50.0
    fit = fit_temperature(scores, [7] * 5)
    assert fit.at_bound and fit.tau == pytest.approx(TAU_BOUNDS[0], rel=1e-3)


def test_tempering_keeps_ranks_and_log_space_is_exact():
    from fsa.attack.calibration import log2_p_true, tempered_posteriors, true_ranks

    rng = np.random.default_rng(1)
    s = rng.normal(0, 500, size=(4, 100))  # exp() of these would overflow
    idx = [3, 10, 50, 99]
    assert np.allclose(tempered_posteriors(s, 1.0), np.stack([attack.softmax(r) for r in s]))
    for tau in (0.5, 7.0, 300.0):
        assert true_ranks(s / tau, idx) == true_ranks(s, idx)
        p = tempered_posteriors(s, tau)[np.arange(4), idx]
        lp = log2_p_true(s, idx, tau)
        assert np.all(np.isfinite(lp))
        ok = p > 1e-300  # where p is representable, log space agrees with it
        assert np.allclose(lp[ok], np.log2(p[ok]))


def test_ranks_histogram_and_first_vs_not():
    from fsa.attack.calibration import first_vs_not, rank_histogram, true_ranks

    s = np.tile(np.arange(100.0)[::-1], (4, 1))  # candidate 0 best, 99 worst
    ranks = true_ranks(s, [0, 3, 30, 99])
    assert ranks == [1, 4, 31, 100]
    assert rank_histogram(ranks) == {
        "1": 1,
        "2-5": 1,
        "6-10": 0,
        "11-25": 0,
        "26-50": 1,
        "51-100": 1,
    }
    assert rank_histogram([1, 2, 3], n_candidates=3) == {"1": 1, "2-3": 2}
    assert first_vs_not([-1.0, -5.0, -7.0], [1, 2, 9]) == (1, -1.0, -6.0)
    assert first_vs_not([-5.0], [3]) == (0, None, -5.0)


# --------------------------------------------------------------------------- sampling check

OLD_GEN = {  # config.yaml format written before 2026-10-21 (e.g. the Gemma-12B runs)
    "eval": {"temperature": 1.0},
    "generation": {
        "backend": "real",
        "device": "cuda",
        "dtype": "bfloat16",
        "max_new_tokens": 160,
        "sampling": "temperature only (top_k=0, top_p=1, no repetition penalty)",
    },
    "script": "scripts/baseline_leakage.py",
}


def test_sampling_parsed_from_old_configs_and_matches():
    g = attack.generation_settings(OLD_GEN)
    assert (g.temperature, g.top_k, g.top_p, g.repetition_penalty) == (1.0, 0, 1.0, 1.0)
    assert g.padding_side == "left" and g.prompt_padding.startswith("none")
    assert "parsed" in g.source and "inferred" in g.source
    check = attack.sampling_check(OLD_GEN, _scorer())
    assert check.matches and not check.mismatches
    assert any("dtype differs" in n for n in check.notes)  # bf16 generation, fp32 scorer


def test_sampling_mismatches_are_reported():
    scorer = _scorer().model_copy(update={"temperature": 0.7})
    check = attack.sampling_check(OLD_GEN, scorer)
    assert not check.matches and any("temperature" in m for m in check.mismatches)
    nucleus_gen = {
        **OLD_GEN["generation"],
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "prompt_padding": "none",
    }
    check = attack.sampling_check({**OLD_GEN, "generation": nucleus_gen}, _scorer())
    assert {m.split(" =")[0] for m in check.mismatches} == {"generation top_k", "generation top_p"}
    unknown = attack.sampling_check({"eval": {"temperature": 1.0}, "generation": {}}, _scorer())
    assert not unknown.matches and any("not recorded" in m for m in unknown.mismatches)


def test_baseline_records_structured_sampling(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import yaml

    from fsa_runtime.generators.hf import SAMPLING

    class FakeReal(MockGenerator):
        is_mock = False
        note = None

    settings = SimpleNamespace(backend="real", runs_dir=tmp_path / "runs", mock_runs_dir=tmp_path)
    monkeypatch.setattr(baseline_leakage, "load_settings", lambda: settings)
    monkeypatch.setattr(baseline_leakage, "_generator", lambda s, b, a: FakeReal.from_config())
    args = ["--backend", "real", "--n-train", "2", "--n-heldout", "2", "--replies", "2"]
    assert baseline_leakage.main(args) == 0
    (run,) = (tmp_path / "runs").iterdir()
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    gen = config["generation"]
    assert {k: gen[k] for k in SAMPLING} == SAMPLING
    assert gen["padding_side"] == "left" and gen["prompt_padding"].startswith("none")
    used = attack.generation_settings(config)
    assert used.source == "config.yaml generation.*" and used.top_k == 0


# --------------------------------------------------------------------------- --calibrate


def _reply_secrets(run, tok):
    """Re-tokenised reply -> its secret, over held-out and train replies."""
    out = {}
    for name in ("samples.jsonl", "train_samples.jsonl"):
        for line in (run / name).read_text(encoding="utf-8").splitlines():
            s = json.loads(line)
            out[tuple(tok(s["reply"], add_special_tokens=False)["input_ids"])] = s["secret"]
    return out


def _overconfident(by_reply, calls=None):
    """+40 on the true secret for even secrets, +40 on a wrong one for odd secrets: confidently
    wrong half the time, so the raw posterior is badly overconfident and tau comes out > 1."""

    def fake(model, tokenizer, build_prompt_ids, reply_token_ids, candidates, temperature):
        if calls is not None:
            calls.append(len(reply_token_ids))
        out = np.zeros(len(candidates))
        for r in reply_token_ids:
            secret = by_reply[tuple(r[:-1])]  # drop the end token
            target = secret if int(secret) % 2 == 0 else f"{(int(secret) + 1) % 100:02d}"
            out += np.array([40.0 if c == target else 0.0 for c in candidates])
        return out

    return fake


def test_calibrate_fits_tau_on_train_and_reports_side_by_side(tmp_path, tok, monkeypatch):
    run = _mock_run(tmp_path, n_heldout=6, replies=2)
    monkeypatch.setattr(likelihood, "score_candidates", _overconfident(_reply_secrets(run, tok)))
    res, path = attack.likelihood_attack(
        run,
        model=None,
        tokenizer=tok,
        scorer=_scorer(),
        max_batch_tokens=10**6,
        calibrate=True,
        n_permutations=50,
        n_bootstrap=50,
        log=lambda m: None,
    )
    cal = res.calibration
    assert cal is not None and cal.tau > 1 and len(cal.train_secrets) == 4
    assert cal.train_mean_log2_p_calibrated > cal.train_mean_log2_p_raw
    raw_m, cal_m = res.leakage, cal.leakage
    assert raw_m.attacker.exact_match.value == cal_m.attacker.exact_match.value  # same ranking
    assert cal_m.bits_leaked_lower_bound >= raw_m.bits_leaked_lower_bound
    assert cal_m.attacker_mean_ce_bits < raw_m.attacker_mean_ce_bits
    k = res.ranks
    n_even = sum(int(s) % 2 == 0 for s in res.secrets)
    assert sum(k.histogram.values()) == 6 and k.n_first == n_even
    if n_even < 6:
        assert k.mean_log2_p_true_not_first < k.mean_log2_p_true_not_first_calibrated
    text = attack.format_report(res)
    assert "calibrated (tau =" in text and "rank of the true secret" in text
    saved = LikelihoodAttack.model_validate_json(path.read_text(encoding="utf-8"))
    assert saved.calibration.tau == cal.tau and saved.likelihood_attack_version == 2


def test_without_calibrate_train_is_not_scored(tmp_path, tok, monkeypatch):
    run = _mock_run(tmp_path, n_heldout=4, replies=2)
    calls = []
    monkeypatch.setattr(likelihood, "score_candidates", _additive_fake(calls))
    res, _ = attack.likelihood_attack(
        run, model=None, tokenizer=tok, scorer=_scorer(), max_batch_tokens=10**6, log=lambda m: None
    )
    assert res.calibration is None and res.ranks is not None and len(calls) == 4  # held-out only


def test_calibrate_needs_train_samples(tmp_path, tok, monkeypatch):
    run = _mock_run(tmp_path)
    (run / "train_samples.jsonl").unlink()
    monkeypatch.setattr(likelihood, "score_candidates", _additive_fake([]))
    with pytest.raises(attack.AttackError, match="train_samples.jsonl"):
        attack.likelihood_attack(
            run,
            model=None,
            tokenizer=tok,
            scorer=_scorer(),
            max_batch_tokens=10**6,
            calibrate=True,
            log=lambda m: None,
        )


# --------------------------------------------------------------------------- --from-scores


def test_from_scores_reanalyses_without_a_model(tmp_path, tok, monkeypatch):
    run = _mock_run(tmp_path, n_heldout=4, replies=2)
    monkeypatch.setattr(likelihood, "score_candidates", _additive_fake([]))
    first, path = attack.likelihood_attack(
        run, model=None, tokenizer=tok, scorer=_scorer(), max_batch_tokens=10**6, log=lambda m: None
    )
    # Simulate a version-1 file, written before ranks/sampling/calibration existed.
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("sampling", "ranks", "calibration", "rescored_from"):
        data.pop(key)
    data["likelihood_attack_version"] = 1
    old = run / "old.json"
    old.write_text(json.dumps(data), encoding="utf-8")

    def no_model(*a, **k):
        raise AssertionError("must not load a model")

    monkeypatch.setattr(attack, "_load_scorer", no_model)
    monkeypatch.setattr(likelihood, "score_candidates", no_model)
    before = path.read_bytes()
    assert attack.main([str(run), "--from-scores", str(old), "--n-permutations", "20"]) == 0
    new = LikelihoodAttack.model_validate_json(
        (run / "likelihood_attack_2.json").read_text(encoding="utf-8")
    )
    assert new.scores == first.scores and new.rescored_from == "old.json"
    assert new.ranks.ranks == first.ranks.ranks and new.sampling is not None
    assert path.read_bytes() == before  # original untouched


def test_from_scores_with_calibrate_scores_only_train(tmp_path, tok, monkeypatch):
    run = _mock_run(tmp_path, n_heldout=4, replies=2)
    by_reply = _reply_secrets(run, tok)
    monkeypatch.setattr(likelihood, "score_candidates", _overconfident(by_reply))
    _, path = attack.likelihood_attack(
        run, model=None, tokenizer=tok, scorer=_scorer(), max_batch_tokens=10**6, log=lambda m: None
    )
    calls = []
    monkeypatch.setattr(likelihood, "score_candidates", _overconfident(by_reply, calls))
    monkeypatch.setattr(
        attack, "_load_scorer", lambda args, saved: (None, tok, saved.scorer, 10**6, None, [])
    )
    assert attack.main([str(run), "--from-scores", str(path), "--calibrate"]) == 0
    second = run / "likelihood_attack_2.json"
    res = LikelihoodAttack.model_validate_json(second.read_text(encoding="utf-8"))
    assert res.calibration is not None and len(calls) == 4  # the 4 train secrets, no held-out
    # A file that already holds train scores is recalibrated with no model at all.
    monkeypatch.setattr(attack, "_load_scorer", lambda *a: pytest.fail("model loaded"))
    assert attack.main([str(run), "--from-scores", str(second), "--calibrate"]) == 0
    again = LikelihoodAttack.model_validate_json(
        (run / "likelihood_attack_3.json").read_text(encoding="utf-8")
    )
    assert again.calibration.tau == pytest.approx(res.calibration.tau)


def test_from_scores_rejects_another_runs_file(tmp_path, tok, monkeypatch):
    run = _mock_run(tmp_path, n_heldout=2, replies=1)
    monkeypatch.setattr(likelihood, "score_candidates", _additive_fake([]))
    _, path = attack.likelihood_attack(
        run, model=None, tokenizer=tok, scorer=_scorer(), max_batch_tokens=10**6, log=lambda m: None
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    data["run_id"] = "20200101-000000_other"
    other = tmp_path / "other.json"
    other.write_text(json.dumps(data), encoding="utf-8")
    assert attack.main([str(run), "--from-scores", str(other)]) == 1
