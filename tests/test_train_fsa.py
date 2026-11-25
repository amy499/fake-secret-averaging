"""FSA / B3 / random-layer training loop on a tiny random-weight Qwen2 (conftest), plus the
evaluation hooks (baseline --adapter, likelihood attacker with the adapter) and the configs."""

import copy
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from fsa_runtime.generators.base import ModelSpec
from fsa_runtime.generators.hf import HFGenerator, load_system_templates
from fsa.secrets import split_secrets
from fsa.targets.mixture import mix_logprobs
from fsa.training.config import TrainConfig
from fsa.training.lora import adapted_layers, resolve_layers
from fsa.training.loss import fsa_loss
from fsa.training.trainer import ADAPTER_DIR, LOG_FILE, OFFLINE_FILE, Trainer
from scripts import baseline_leakage, likelihood_attack, train_fsa

ROOT = Path(__file__).resolve().parents[1]
BASE = {
    "name": "tiny",
    "seed": 0,
    "method": "fsa",
    "model": {"id": "tiny"},
    "target": "mixture",
    "reply_source": "offline",
    "fsa": {"k": 2},
    "lora": {"layers": "all", "rank": 4, "alpha": 8, "dropout": 0.0},
    "secrets": {"split_seed": 0, "n_train": 4, "n_heldout": 4},
    "generation": {"max_new_tokens": 6, "offline_replies_per_prompt": 1},
    "train": {
        "steps": 4,
        "lr": 1e-2,
        "warmup_steps": 0,
        "batch_size": 1,
        "grad_accum": 1,
        "gradient_checkpointing": False,
        "checkpoint_every": 2,
        "dtype": "float32",
    },
    "monitor": {"every": 1000, "full_mixture_every": 0},
}


def make_cfg(**over) -> TrainConfig:
    raw = copy.deepcopy(BASE)
    for key, value in over.items():
        if isinstance(value, dict):
            raw.setdefault(key, {}).update(value)
        else:
            raw[key] = value
    return TrainConfig.model_validate(raw)


def make_trainer(cfg, tok, model_dir, out):
    from transformers import AutoModelForCausalLM

    out.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    data, resolved = train_fsa.prepare(cfg)
    info = {"hf_id": str(model_dir), "heldout_secrets": resolved["heldout_secrets"]}
    return Trainer(
        cfg, model=model, tokenizer=tok, data=data, out_dir=out, info=info, log=lambda m: None
    )


def read_log(out):
    return [json.loads(x) for x in (out / LOG_FILE).read_text(encoding="utf-8").splitlines()]


def train_records(out):
    return [r for r in read_log(out) if r["kind"] == "train"]


# --------------------------------------------------------------------------- the loop


def test_loss_decreases(tok, tiny_model_dir, tmp_path):
    # A random tiny model barely depends on its secret (KL to the mixture starts ~1e-5), so
    # start the adapter far from the target: then a falling loss shows the gradients reach the
    # LoRA weights and pull the adapted model onto the fake-secret mixture.
    cfg = make_cfg(
        secrets={"n_train": 3},  # 3 secrets x 2 prompts x 1 cached reply: the same 6 examples
        train={"steps": 30, "lr": 5e-3, "grad_accum": 2, "checkpoint_every": 1000},
    )
    trainer = make_trainer(cfg, tok, tiny_model_dir, tmp_path / "run")
    with torch.no_grad():
        for name, p in trainer.model.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.3)
    trainer.train()
    losses = [r["loss"] for r in train_records(tmp_path / "run")]
    assert all(np.isfinite(losses)) and min(losses) >= 0
    assert np.mean(losses[:5]) > 0.005  # the perturbed adapter really starts off-target
    assert np.mean(losses[-5:]) < 0.6 * np.mean(losses[:5])  # measured: ~0.47x after 30 steps
    assert np.corrcoef(np.arange(len(losses)), losses)[0, 1] < -0.5  # a trend, not one lucky step


def test_checkpoint_save_reload_and_exact_resume(tok, tiny_model_dir, tmp_path):
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    cfg = make_cfg(lora={"dropout": 0.1}, train={"steps": 4, "checkpoint_every": 2})
    full = make_trainer(cfg, tok, tiny_model_dir, tmp_path / "full")
    full.train()
    part = make_trainer(cfg, tok, tiny_model_dir, tmp_path / "part")
    part.train(stop_after=2)  # "crash" after the step-2 checkpoint
    assert (tmp_path / "part" / "checkpoints" / "step_000002" / "state.pt").exists()
    assert not (tmp_path / "part" / ADAPTER_DIR).exists()
    resumed = make_trainer(cfg, tok, tiny_model_dir, tmp_path / "part")
    assert resumed.resume() == 2
    resumed.train()

    a = load_file(str(tmp_path / "full" / ADAPTER_DIR / "adapter_model.safetensors"))
    b = load_file(str(tmp_path / "part" / ADAPTER_DIR / "adapter_model.safetensors"))
    assert a.keys() == b.keys()
    assert all(torch.allclose(a[k], b[k], atol=1e-6) for k in a)
    strip = lambda rs: [{k: v for k, v in r.items() if k != "step_time_s"} for r in rs]
    assert strip(read_log(tmp_path / "full")) == strip(read_log(tmp_path / "part"))

    # The saved adapter reloads onto a fresh base model and gives the trained model's logits.
    base = AutoModelForCausalLM.from_pretrained(tiny_model_dir, dtype=torch.float32)
    reloaded = PeftModel.from_pretrained(base, tmp_path / "full" / ADAPTER_DIR).eval()
    full.model.eval()
    ids = torch.tensor([full.real_prompt(full.data.train_secrets[0], "base", "rand_20_2digit")])
    with torch.no_grad():
        assert torch.allclose(
            reloaded(input_ids=ids).logits, full.model(input_ids=ids).logits, atol=1e-5
        )
    info = json.loads((tmp_path / "full" / ADAPTER_DIR / "training_info.json").read_text())
    assert info["step"] == 4 and info["train_secrets"] == full.data.train_secrets


def test_heldout_secrets_and_prompts_never_used(tok, tiny_model_dir, tmp_path, monkeypatch):
    seen = []
    original = Trainer._ids

    def spy(self, system, user):
        seen.append((system, user))
        return original(self, system, user)

    monkeypatch.setattr(Trainer, "_ids", spy)
    prompts = yaml.safe_load((ROOT / "configs" / "training" / "prompts.yaml").read_text(encoding="utf-8"))
    user_text = {p["id"]: p["text"] for p in prompts["user_prompts"]}
    for i, mode in enumerate(("offline", "on_policy")):
        cfg = make_cfg(reply_source=mode, utility={"weight": 0.5}, train={"steps": 3})
        out = tmp_path / mode
        trainer = make_trainer(cfg, tok, tiny_model_dir, out)
        trainer.train()
        train, heldout = split_secrets(0, 4, 4)
        assert trainer.data.train_secrets == train  # the trainer is only ever given these
        for system, user in seen:
            assert user != user_text["lottery"]  # the held-out user prompt
            assert not any(f"code is {h}." in system for h in heldout)
        for r in train_records(out):
            for ex in r["examples"]:
                assert ex["secret"] in train and set(ex["fakes"]) <= set(train)
                assert ex["prompt_id"] in ("rand_20_2digit", "rand_10_2digit")
                assert ex["method"] == "base"
            assert r["utility_kl"] is not None
        if mode == "offline":
            cached = [json.loads(x) for x in (out / OFFLINE_FILE).read_text().splitlines()]
            assert cached and {c["secret"] for c in cached} <= set(train)
    assert any("code is" in s for s, _ in seen) and len(seen) > 10


def test_frozen_pass_really_has_the_adapter_off(tok, tiny_model_dir, tmp_path):
    from transformers import AutoModelForCausalLM

    fresh = AutoModelForCausalLM.from_pretrained(tiny_model_dir, dtype=torch.float32).eval()
    for target in ("mixture", "no_secret"):
        method = "fsa" if target == "mixture" else "b3"
        trainer = make_trainer(
            make_cfg(method=method, target=target), tok, tiny_model_dir, tmp_path / target
        )
        with torch.no_grad():  # make the adapter matter: LoRA B starts at zero
            for name, p in trainer.model.named_parameters():
                if "lora_B" in name:
                    p.normal_(0, 0.5)
        trainer.model.train()
        ex = trainer.draw_example(np.random.default_rng(0), 0, 0, 0)
        batch = trainer.build_batch([ex])
        got = trainer.target(batch)
        assert trainer.model.training  # mode restored after the frozen pass
        with torch.no_grad():
            if target == "mixture":
                per_fake = [
                    torch.log_softmax(fresh(input_ids=f).logits, -1) for f in batch.fake_ids
                ]
                expected = mix_logprobs(torch.stack(per_fake))
                m = batch.mask
                assert torch.allclose(got[m], expected[m], atol=1e-5)
            else:
                lp = torch.log_softmax(fresh(input_ids=batch.ns_ids).logits, -1)
                assert torch.allclose(got[batch.mask], lp[batch.ns_mask], atol=1e-5)
            # ...while the adapter, when on, really changes the output.
            adapted = trainer.model(input_ids=batch.ids).logits
            assert not torch.allclose(adapted, fresh(input_ids=batch.ids).logits, atol=1e-3)


def test_fake_secrets_change_between_steps(tok, tiny_model_dir, tmp_path):
    trainer = make_trainer(
        make_cfg(fsa={"k": 2}, train={"steps": 6}), tok, tiny_model_dir, tmp_path / "r"
    )
    trainer.train()
    draws = [tuple(ex["fakes"]) for r in train_records(tmp_path / "r") for ex in r["examples"]]
    assert len(draws) == 6 and len(set(draws)) > 1
    for r in train_records(tmp_path / "r"):
        for ex in r["examples"]:
            assert ex["secret"] not in ex["fakes"] and len(set(ex["fakes"])) == 2


@pytest.mark.parametrize(
    ("method", "target", "source"),
    [
        ("fsa", "mixture", "offline"),
        ("fsa", "mixture", "on_policy"),
        ("b3", "no_secret", "offline"),
    ],
)
def test_training_modes_run(tok, tiny_model_dir, tmp_path, method, target, source):
    cfg = make_cfg(
        method=method,
        target=target,
        reply_source=source,
        train={"steps": 2, "gradient_checkpointing": True, "grad_accum": 2, "batch_size": 2},
        monitor={"every": 1},
    )
    trainer = make_trainer(cfg, tok, tiny_model_dir, tmp_path / "r")
    trainer.train()
    out = tmp_path / "r"
    assert (out / ADAPTER_DIR / "adapter_model.safetensors").exists()
    recs = train_records(out)
    assert len(recs) == 2 and all(np.isfinite(r["loss"]) for r in recs)
    assert all(len(r["examples"]) == 4 for r in recs)  # batch 2 x grad_accum 2
    monitors = [r for r in read_log(out) if r["kind"] == "monitor"]
    assert [m["step"] for m in monitors] == [0, 1, 2]
    assert monitors[0]["kl_frozen_no_secret_digit"] == pytest.approx(0, abs=1e-6)  # untrained
    assert all(m["kl_frozen_general_eval"] is not None for m in monitors)
    assert (out / OFFLINE_FILE).exists() == (source == "offline")


def test_random_layers_are_seeded(tok, tiny_model_dir, tmp_path):
    assert resolve_layers("random:8", 48, 0) == resolve_layers("random:8", 48, 0)
    picks = {tuple(resolve_layers("random:8", 48, s)) for s in range(5)}
    assert len(picks) > 1 and all(len(p) == 8 and len(set(p)) == 8 for p in picks)
    assert resolve_layers("all", 3, 0) == [0, 1, 2] and resolve_layers([2, 0], 3, 0) == [0, 2]
    with pytest.raises(ValueError):
        resolve_layers("random:4", 3, 0)
    cfg = make_cfg(method="random_layer", lora={"layers": "random:1"})
    trainer = make_trainer(cfg, tok, tiny_model_dir, tmp_path / "r")
    assert adapted_layers(trainer.model) == set(resolve_layers("random:1", 2, 0))


def test_config_rules():
    with pytest.raises(ValueError, match="no_secret"):
        make_cfg(method="b3")  # b3 needs target: no_secret
    with pytest.raises(ValueError, match="random:N"):
        make_cfg(method="random_layer")
    with pytest.raises(ValueError, match="layers"):
        make_cfg(lora={"layers": "some"})


def test_experiment_configs_validate():
    names = {}
    for path in sorted((ROOT / "configs" / "training" / "experiments").glob("*.yaml")):
        if path.name.startswith(("fsa_v2_", "fsa_lite_")):  # read by src/fsa_lite/lite/v2_train.py: its own schema
            continue
        cfg = TrainConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        names[path.name] = cfg
    debug = names["fsa_gemma4b_debug.yaml"]
    assert debug.model.id == "gemma3-4b-it" and debug.fsa.k == 8
    assert (debug.lora.rank, debug.lora.alpha, debug.train.lr) == (16, 32, 1e-4)
    assert names["b3_gemma12b.yaml"].target == "no_secret"
    assert str(names["random_layer_gemma12b.yaml"].lora.layers).startswith("random:")
    assert names["fsa_gemma12b.yaml"].model.id == "gemma3-12b-it"


# --------------------------------------------------------------------------- CLI + job script


def test_train_cli_writes_run_and_resumes(tok, tiny_model_dir, tmp_path):
    raw = copy.deepcopy(BASE)
    raw["model"]["id"] = str(tiny_model_dir)
    raw["train"]["dtype"] = "auto"
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    out_dir = tmp_path / "training"
    assert (
        train_fsa.main(["--config", str(cfg_path), "--device", "cpu", "--out-dir", str(out_dir)])
        == 0
    )
    (run,) = out_dir.iterdir()
    record = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    assert record["config"]["name"] == "tiny" and "commit" in record["git"]
    res = record["resolved"]
    # resolved.heldout_user_prompts = every user prompt not trained on (lottery among them)
    assert res["layers"] == [0, 1] and "lottery" in res["heldout_user_prompts"]
    assert not set(res["train_user_prompts"]) & set(res["heldout_user_prompts"])
    assert not set(res["train_secrets"]) & set(res["heldout_secrets"])
    info = json.loads((run / ADAPTER_DIR / "training_info.json").read_text())
    assert info["heldout_secrets"] == res["heldout_secrets"] and info["hf_id"] == str(
        tiny_model_dir
    )
    n_lines = len(read_log(run))
    assert train_fsa.main(["--resume", str(run), "--device", "cpu"]) == 0  # already finished
    assert len(read_log(run)) == n_lines


def test_train_cli_resume_fails_on_different_saved_split(tmp_path, monkeypatch, capsys):
    cfg = make_cfg()
    train, heldout = split_secrets(0, 4, 4)
    saved_train = [*train[:-1], heldout[0]]  # one train secret swapped for a held-out one
    run = tmp_path / "run"
    run.mkdir()
    record = {
        "config": cfg.model_dump(mode="json"),
        "resolved": {"train_secrets": saved_train, "heldout_secrets": heldout},
    }
    (run / "config.yaml").write_text(yaml.safe_dump(record), encoding="utf-8")

    def no_model(*a, **k):
        raise AssertionError("model loaded despite a split mismatch")

    monkeypatch.setattr(train_fsa, "load_model", no_model)
    assert train_fsa.main(["--resume", str(run), "--device", "cpu"]) == 1
    err = capsys.readouterr().err
    assert "train split on resume differs" in err
    assert f"only in the saved config.yaml: [{heldout[0]!r}]" in err
    assert f"only in the recomputed split: [{train[-1]!r}]" in err


def test_train_cli_resume_fails_on_different_saved_heldout(tmp_path, monkeypatch, capsys):
    cfg = make_cfg()
    train, heldout = split_secrets(0, 4, 4)
    outside = next(f"{i:02d}" for i in range(100) if f"{i:02d}" not in train + heldout)
    saved_heldout = [*heldout[:-1], outside]  # train secrets match; one held-out secret differs
    run = tmp_path / "run"
    run.mkdir()
    record = {
        "config": cfg.model_dump(mode="json"),
        "resolved": {"train_secrets": train, "heldout_secrets": saved_heldout},
    }
    (run / "config.yaml").write_text(yaml.safe_dump(record), encoding="utf-8")

    def no_model(*a, **k):
        raise AssertionError("model loaded despite a split mismatch")

    monkeypatch.setattr(train_fsa, "load_model", no_model)
    assert train_fsa.main(["--resume", str(run), "--device", "cpu"]) == 1
    err = capsys.readouterr().err
    assert "held-out split on resume differs" in err
    assert f"only in the saved config.yaml: [{outside!r}]" in err
    assert f"only in the recomputed split: [{heldout[-1]!r}]" in err


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
@pytest.mark.parametrize(
    ("env_extra", "expected"),
    [
        (
            {"CONFIG": "configs/training/experiments/x.yaml"},
            ["--config", "configs/training/experiments/x.yaml", "--device", "cuda", "--max-minutes", "330"],
        ),
        (
            {"RESUME": "outputs/training/r", "CONFIG": "c.yaml", "MAX_MINUTES": "100"},
            ["--resume", "outputs/training/r", "--device", "cuda", "--max-minutes", "100"],
        ),
        ({}, None),
    ],
)
def test_train_sbatch(tmp_path, env_extra, expected):
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
        k: v
        for k, v in os.environ.items()
        if k not in ("CONFIG", "RESUME", "EXTRA_ARGS", "MAX_MINUTES")
    }
    env.update(
        HOME=str(tmp_path / "home"),
        PATH=str(fake_bin) + os.pathsep + os.environ["PATH"],
        SLURM_SUBMIT_DIR=str(tmp_path),
        **env_extra,
    )
    out = subprocess.run(
        [shutil.which("bash"), str(ROOT / "scripts" / "slurm" / "fsa_train.sbatch")],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if expected is None:
        assert out.returncode != 0 and "set CONFIG=" in out.stderr
        return
    assert out.returncode == 0, out.stderr
    args = [line[1:-1] for line in out.stdout.splitlines() if line.startswith("[")]
    assert args == [
        "run",
        "--group",
        "ml",
        "python",
        "-m",
        "scripts.train_fsa",
        *expected,
    ]


def test_every_sbatch_stays_within_the_tc2_limit():
    for path in sorted((ROOT / "scripts" / "slurm").glob("fsa_*.sbatch")):
        times = [x for x in path.read_text(encoding="utf-8").splitlines() if "--time=" in x]
        assert times and all(x.startswith("#SBATCH --time=06:00:00") for x in times), path.name


# --------------------------------------------------------------------------- evaluation hooks


@pytest.fixture
def trained_adapter(tok, tiny_model_dir, tmp_path):
    cfg = make_cfg(secrets={"n_train": 3}, train={"steps": 2})
    trainer = make_trainer(cfg, tok, tiny_model_dir, tmp_path / "train")
    trainer.info["hf_id"] = str(tiny_model_dir)
    trainer.train()
    return tmp_path / "train" / ADAPTER_DIR


def _eval_run(tiny_model_dir, adapter, out, *, n_train=3, method="fsa"):
    gen = HFGenerator(templates=load_system_templates(), max_new_tokens=8, batch_size=2)
    spec = ModelSpec(
        method=method,
        base_model=str(tiny_model_dir),
        hf_id=str(tiny_model_dir),
        adapter=str(adapter),
    )
    return baseline_leakage.run_baseline(
        generator=gen,
        spec=spec,
        n_layers=2,
        prompt_id="rand_20_2digit",
        n_train=n_train,
        n_heldout=2,
        replies=1,
        seed=0,
        temperature=1.0,
        out_dir=out,
        name="tiny_fsa",
        generation_info={"backend": "real", "max_new_tokens": 8},
        log=lambda m: None,
    )


def test_baseline_generates_with_adapter_and_records_it(tiny_model_dir, trained_adapter, tmp_path):
    writer = _eval_run(tiny_model_dir, trained_adapter, tmp_path / "runs")
    config = yaml.safe_load((writer.path / "config.yaml").read_text(encoding="utf-8"))
    assert config["method"] == "fsa"
    assert Path(config["generation"]["adapter"]) == trained_adapter.resolve()
    assert config["generation"]["adapter_training"]["step"] == 2
    assert config["fsa"] == {"k": 2} and config["lora"] == {"layers": [0, 1]}


def test_baseline_refuses_unsafe_adapters(tiny_model_dir, trained_adapter, tmp_path):
    # Eval split with n_train=2 puts training secret #3 into the held-out set.
    with pytest.raises(ValueError, match="training secrets"):
        _eval_run(tiny_model_dir, trained_adapter, tmp_path / "a", n_train=2)
    with pytest.raises(ValueError, match="trained as 'fsa'"):
        _eval_run(tiny_model_dir, trained_adapter, tmp_path / "b", method="b3")


def test_baseline_cli_adapter_flags(monkeypatch, tmp_path, capsys):
    from types import SimpleNamespace

    settings = SimpleNamespace(
        backend="mock", runs_dir=tmp_path / "runs", mock_runs_dir=tmp_path / "m"
    )
    monkeypatch.setattr(baseline_leakage, "load_settings", lambda: settings)
    base = ["--backend", "mock", "--n-train", "2", "--n-heldout", "2", "--replies", "1"]
    assert baseline_leakage.main([*base, "--method", "fsa"]) == 1
    assert "needs --adapter" in capsys.readouterr().err
    assert baseline_leakage.main([*base, "--adapter", str(tmp_path)]) == 1
    assert "--adapter is for" in capsys.readouterr().err


def test_likelihood_attacker_scores_with_the_adapter(
    tiny_model_dir, trained_adapter, tmp_path, monkeypatch
):
    from peft import PeftModel

    writer = _eval_run(tiny_model_dir, trained_adapter, tmp_path / "runs")
    loaded = []
    original = HFGenerator._load

    def spy(self, spec):
        tok, model = original(self, spec)
        loaded.append((spec, model))
        return tok, model

    monkeypatch.setattr(HFGenerator, "_load", spy)
    args = [str(writer.path), "--device", "cpu", "--limit-secrets", "2", "--calibrate"]
    assert likelihood_attack.main([*args, "--n-permutations", "10", "--n-bootstrap", "10"]) == 0
    ((spec, model),) = loaded
    assert isinstance(model, PeftModel) and spec.method == "fsa"
    assert Path(spec.adapter) == trained_adapter.resolve()
    (path,) = writer.path.glob("likelihood_attack*.json")
    res = likelihood_attack.LikelihoodAttack.model_validate_json(path.read_text(encoding="utf-8"))
    assert Path(res.scorer.adapter) == trained_adapter.resolve()
    assert res.calibration is not None and len(res.calibration.train_secrets) == 2


# --------------------------------------------------------------------------- full mixture


def test_kl_to_full_mixture_logged_printed_and_correct(tok, tiny_model_dir, tmp_path):
    from transformers import AutoModelForCausalLM

    lines = []
    cfg = make_cfg(
        train={"steps": 4}, monitor={"full_mixture_every": 2, "full_mixture_examples": 2}
    )
    trainer = make_trainer(cfg, tok, tiny_model_dir, tmp_path / "r")
    trainer.log = lines.append
    trainer.train()
    logged = {
        r["step"]: r["kl_to_full_mixture"]
        for r in read_log(tmp_path / "r")
        if "kl_to_full_mixture" in r
    }
    assert sorted(logged) == [0, 2, 4] and all(v >= 0 for v in logged.values())
    assert logged[0] > 0  # the untrained adapter: KL(real secret || full mixture) of the base model
    assert "kl_to_full_mixture" in lines[1] and "(@2)" in lines[2]  # steps 2 and 3
    items = trainer.full_mixture_examples()
    assert len(items) == 2 and all(i["secret"] in trainer.data.train_secrets for i in items)
    assert trainer.kl_to_full_mixture() == pytest.approx(logged[4])  # fixed examples: no noise

    # Independent check: the mixture over ALL other training secrets from a fresh base model.
    # Perturb the adapter first: the tiny model's own KL (~1e-5) is a sum of nearly cancelling
    # float32 terms over a 151k vocab, too small to compare to better than ~0.3%.
    with torch.no_grad():
        for name, p in trainer.model.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.3)
    expected = trainer.kl_to_full_mixture()
    assert expected > 100 * logged[0]
    fresh = AutoModelForCausalLM.from_pretrained(tiny_model_dir, dtype=torch.float32).eval()
    trainer.model.eval()
    vals = []
    with torch.no_grad():
        for it in items:
            others = sorted(s for s in trainer.data.train_secrets if s != it["secret"])
            prompt = trainer.real_prompt(it["secret"], it["method"], it["prompt_id"])
            seq = torch.tensor([prompt + it["reply"]])
            per = [
                torch.log_softmax(
                    fresh(
                        input_ids=torch.tensor(
                            [trainer.real_prompt(o, it["method"], it["prompt_id"]) + it["reply"]]
                        )
                    ).logits,
                    -1,
                )
                for o in others
            ]
            target = mix_logprobs(torch.stack(per))
            mask = torch.zeros(seq.shape, dtype=torch.bool)
            mask[0, len(prompt) - 1 : seq.shape[1] - 1] = True
            vals.append(fsa_loss(trainer.model(input_ids=seq).logits, target, mask).item())
    assert np.mean(vals) == pytest.approx(expected, rel=1e-4)


def test_full_mixture_examples_survive_resume(tok, tiny_model_dir, tmp_path):
    cfg = make_cfg(train={"steps": 4}, monitor={"full_mixture_every": 2})
    full = make_trainer(cfg, tok, tiny_model_dir, tmp_path / "full")
    full.train()
    part = make_trainer(cfg, tok, tiny_model_dir, tmp_path / "part")
    part.train(stop_after=2)
    resumed = make_trainer(cfg, tok, tiny_model_dir, tmp_path / "part")
    resumed.resume()
    assert resumed.last_full_mixture is not None and resumed.last_full_mixture[0] == 2
    resumed.train()
    kl = lambda out: [r.get("kl_to_full_mixture") for r in read_log(out) if r["kind"] == "monitor"]
    assert kl(tmp_path / "full") == pytest.approx(kl(tmp_path / "part"))


def test_all_others_fake_set_is_deterministic_and_complete(tok, tiny_model_dir, tmp_path):
    cfg = make_cfg(fsa={"k": 3, "fake_set": "all_others"}, train={"steps": 4})
    trainer = make_trainer(cfg, tok, tiny_model_dir, tmp_path / "r")
    trainer.train()
    train = trainer.data.train_secrets
    for r in train_records(tmp_path / "r"):
        for ex in r["examples"]:
            assert ex["fakes"] == sorted(s for s in train if s != ex["secret"])
    with pytest.raises(ValueError, match="n_train - 1"):
        make_cfg(fsa={"k": 2, "fake_set": "all_others"})
    fullmix = TrainConfig.model_validate(
        yaml.safe_load((ROOT / "configs/training/experiments/fsa_gemma12b_fullmix.yaml").read_text())
    )
    assert (fullmix.fsa.k, fullmix.fsa.fake_set, fullmix.secrets.n_train) == (49, "all_others", 50)
    base12 = TrainConfig.model_validate(
        yaml.safe_load((ROOT / "configs/training/experiments/fsa_gemma12b.yaml").read_text())
    )
    same = {
        k: v
        for k, v in fullmix.model_dump().items()
        if k not in ("name", "fsa", "monitor", "train")
    }
    assert same == {k: v for k, v in base12.model_dump().items() if k in same}
    assert fullmix.train.model_dump(exclude={"checkpoint_every"}) == base12.train.model_dump(
        exclude={"checkpoint_every"}
    )


def test_k49_fakes_are_accumulated_one_at_a_time():
    """mixture_target must not keep earlier fakes' logits alive: memory independent of k."""
    import weakref

    from fsa.targets.mixture import mixture_target, sample_fake_secrets

    train, _ = split_secrets(0, 50, 50)
    fakes = sample_fake_secrets(train[0], 49, pool=train)
    assert sorted(fakes) == sorted(train[1:])  # k = 49 of a 50-secret pool: all the others

    class Recorder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.refs, self.max_alive = [], 0

        def forward(self, input_ids, attention_mask):
            logits = torch.randn(*input_ids.shape, 11)
            alive = sum(r() is not None for r in self.refs)
            self.max_alive = max(self.max_alive, alive)
            self.refs.append(weakref.ref(logits))
            return SimpleNamespace(logits=logits)

    model = Recorder()
    ids = torch.zeros((49, 1, 5), dtype=torch.long)
    out = mixture_target(model, ids, torch.ones_like(ids), torch.ones(1, 5, dtype=torch.bool))
    assert out.shape == (1, 5, 11) and len(model.refs) == 49
    assert model.max_alive <= 1  # at most the previous fake's logits, never a growing stack


# --------------------------------------------------------------------------- time budget


def test_time_budget_checkpoints_and_resume_finishes_identically(tok, tiny_model_dir, tmp_path):
    import time

    cfg = make_cfg(train={"steps": 4, "checkpoint_every": 100})
    full = make_trainer(cfg, tok, tiny_model_dir, tmp_path / "full")
    assert full.train() is True
    lines = []
    part = make_trainer(cfg, tok, tiny_model_dir, tmp_path / "part")
    part.log = lines.append
    assert part.train(deadline=time.monotonic()) is False  # one step, then out of budget
    assert part.step == 1 and (tmp_path / "part/checkpoints/step_000001/state.pt").exists()
    assert "time budget reached" in lines[-1] and not (tmp_path / "part" / ADAPTER_DIR).exists()
    resumed = make_trainer(cfg, tok, tiny_model_dir, tmp_path / "part")
    assert resumed.resume() == 1 and resumed.train() is True
    from safetensors.torch import load_file

    a = load_file(str(tmp_path / "full" / ADAPTER_DIR / "adapter_model.safetensors"))
    b = load_file(str(tmp_path / "part" / ADAPTER_DIR / "adapter_model.safetensors"))
    assert all(torch.allclose(a[k], b[k], atol=1e-6) for k in a)


def test_train_cli_max_minutes_exit_code(tok, tiny_model_dir, tmp_path, capsys):
    raw = copy.deepcopy(BASE)
    raw["model"]["id"] = str(tiny_model_dir)
    raw["train"]["steps"] = 3
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    out_dir = tmp_path / "training"
    args = ["--config", str(cfg_path), "--device", "cpu", "--out-dir", str(out_dir)]
    assert train_fsa.main([*args, "--max-minutes", "0.0001"]) == train_fsa.EXIT_UNFINISHED
    assert "NOT FINISHED" in capsys.readouterr().out
    (run,) = out_dir.iterdir()
    assert train_fsa.main(["--resume", str(run), "--device", "cpu"]) == 0
    assert (run / ADAPTER_DIR / "adapter_model.safetensors").exists()


# --------------------------------------------------------------------------- public weights


def _spy_loads(monkeypatch):
    loaded = []
    original = HFGenerator._load

    def spy(self, spec):
        tok, model = original(self, spec)
        loaded.append((spec, model))
        return tok, model

    monkeypatch.setattr(HFGenerator, "_load", spy)
    return loaded


def test_public_weights_attacker(
    tok, tiny_model_dir, trained_adapter, tmp_path, monkeypatch, capsys
):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from fsa.attack.likelihood import score_candidates
    from fsa.prompting import make_prompt_builder

    writer = _eval_run(tiny_model_dir, trained_adapter, tmp_path / "runs")
    run = writer.path
    loaded = _spy_loads(monkeypatch)
    common = [
        str(run),
        "--device",
        "cpu",
        "--limit-secrets",
        "2",
        "--n-permutations",
        "10",
        "--n-bootstrap",
        "10",
    ]
    assert likelihood_attack.main([*common, "--scorer-no-adapter", "--calibrate"]) == 0
    printed = capsys.readouterr().out
    assert "threat model: PUBLIC_WEIGHTS" in printed
    assert likelihood_attack.main(common) == 0  # grey box, for comparison
    assert "threat model: grey_box" in capsys.readouterr().out

    (spec_pub, model_pub), (_, model_grey) = loaded
    assert spec_pub.adapter is None and not isinstance(model_pub, PeftModel)
    assert isinstance(model_grey, PeftModel)
    pub_file = run / "likelihood_attack_public_weights.json"
    grey_file = run / "likelihood_attack.json"
    pub = likelihood_attack.LikelihoodAttack.model_validate_json(
        pub_file.read_text(encoding="utf-8")
    )
    grey = likelihood_attack.LikelihoodAttack.model_validate_json(
        grey_file.read_text(encoding="utf-8")
    )
    assert pub.threat_model == pub.scorer.threat_model == "public_weights"
    assert grey.threat_model == "grey_box" and grey.scorer.adapter is not None
    assert pub.scorer.adapter is None and pub.scorer.method == "fsa"  # generation's template
    assert pub.calibration is not None and any("public_weights" in n for n in pub.notes)
    assert not np.allclose(pub.scores, grey.scores)  # the adapter matters for the scores

    # Same prompts as generation, base model only: recompute one row by hand.
    base = AutoModelForCausalLM.from_pretrained(tiny_model_dir, dtype=torch.float32).eval()
    samples = [
        json.loads(x) for x in (run / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    secret = pub.secrets[0]
    rows = [s for s in samples if s["secret"] == secret]
    build = make_prompt_builder(tok, load_system_templates(), "fsa", rows[0]["prompt"])
    cands = [f"{i:02d}" for i in range(100)]
    with torch.no_grad():
        manual = score_candidates(
            base, tok, build, [r["reply_token_ids"] for r in rows], cands, 1.0
        )
    assert np.allclose(pub.scores[0], manual.numpy(), rtol=1e-5)

    # Re-analysis keeps the threat model; recalibration scores train secrets WITHOUT the adapter.
    assert (
        likelihood_attack.main([str(run), "--from-scores", str(pub_file), "--n-permutations", "10"])
        == 0
    )
    assert (run / "likelihood_attack_public_weights_2.json").exists()
    assert not (run / "likelihood_attack_2.json").exists()
    pub_nocal = run / "likelihood_attack_public_weights_2.json"
    loaded.clear()
    assert (
        likelihood_attack.main(
            [
                str(run),
                "--from-scores",
                str(pub_nocal),
                "--calibrate",
                "--device",
                "cpu",
                "--n-permutations",
                "10",
            ]
        )
        == 0
    )
    ((spec, model),) = loaded
    assert spec.adapter is None and not isinstance(model, PeftModel)
    # A grey-box file cannot be relabelled as public weights.
    assert (
        likelihood_attack.main([str(run), "--from-scores", str(grey_file), "--scorer-no-adapter"])
        == 1
    )
    assert "saved file decides" in capsys.readouterr().err


def test_scorer_no_adapter_refuses_runs_without_adapter(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    settings = SimpleNamespace(
        backend="mock", runs_dir=tmp_path / "runs", mock_runs_dir=tmp_path / "m"
    )
    monkeypatch.setattr(baseline_leakage, "load_settings", lambda: settings)
    assert (
        baseline_leakage.main(
            ["--backend", "mock", "--n-train", "2", "--n-heldout", "2", "--replies", "1"]
        )
        == 0
    )
    (run,) = (tmp_path / "m").iterdir()
    monkeypatch.setattr(HFGenerator, "_load", lambda self, spec: pytest.fail("must not load"))
    assert likelihood_attack.main([str(run), "--scorer-no-adapter", "--device", "cpu"]) == 1
    assert "used no adapter" in capsys.readouterr().err
