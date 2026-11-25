"""Run-folder helpers shared by tests (synthetic baseline and utility runs; no model)."""

import json
from datetime import datetime
import pytest
import yaml
from test_collect_results import QWEN_HF
from fsa_runtime.generators.base import ModelSpec
from fsa_runtime.generators.mock import MockGenerator
from scripts import baseline_leakage

def _baseline_run(runs_dir, name="extras", prompt_id="rand_20_2digit", seed=1):
    return baseline_leakage.run_baseline(
        generator=MockGenerator.from_config(),
        spec=ModelSpec(method="base", base_model="qwen2.5-0.5b-instruct", hf_id=QWEN_HF),
        n_layers=24,
        prompt_id=prompt_id,
        n_train=4,
        n_heldout=4,
        replies=2,
        seed=seed,
        temperature=1.0,
        out_dir=runs_dir,
        name=name,
        generation_info={"backend": "mock"},
        n_permutations=20,
        n_bootstrap=20,
        log=lambda m: None,
        now=datetime(2026, 10, 18, 12, 0, seed),
    ).path

def _edit_json(path, fn):
    d = json.loads(path.read_text(encoding="utf-8"))
    fn(d)
    path.write_text(json.dumps(d), encoding="utf-8")

def _ci(e, n=10):
    return {
        "estimate": e,
        "low": e - 0.05,
        "high": e + 0.05,
        "n_items": n,
        "n_resamples": 100,
        "seed": 0,
    }

def _cond(metrics, diagnostics=None, n=10):
    return {
        "n_items": n,
        "metrics": {k: _ci(v, n) for k, v in metrics.items()},
        "diagnostics": diagnostics or {},
    }

def _utility_metrics(run_id, *, adapter=None, method="base", kl=0.0):
    both = lambda m, d=None: {"secret": _cond(m, d), "no_secret": _cond(m, d)}
    return {
        "run_id": run_id,
        "created_at": f"{run_id[:4]}-{run_id[4:6]}-{run_id[6:8]}T{run_id[9:11]}:{run_id[11:13]}:00",
        "is_mock": False,
        "model": "gemma3-12b-it",
        "hf_id": "google/gemma-3-12b-it",
        "method": method,
        "adapter": adapter,
        "adapter_run": "20261024-184429_fsa_gemma12b_fullmix" if adapter else None,
        "benchmarks": {
            "mmlu": {
                "conditions": both({"accuracy": 0.62}, {"mean_answer_mass": 0.97}),
                "secret_minus_no_secret": {"accuracy": _ci(-0.01)},
            },
            "ifeval": {
                "conditions": both(
                    {"prompt_level_strict": 0.7, "instruction_level_strict": 0.78},
                    {"mean_reply_tokens": 300.0, "frac_hit_max_new_tokens": 0.02},
                )
            },
            "verification": {
                "conditions": {"secret": _cond({"accuracy": 0.9}, {"mean_answer_mass": 0.99})}
            },
            "kl_to_base": {"conditions": both({"kl_per_token": kl}, {"n_tokens": 4000.0})},
        },
        "seconds_total": 3600.0,
    }

def _utility_run(utility_dir, run_id, **kw):
    path = utility_dir / run_id
    path.mkdir(parents=True)
    m = _utility_metrics(run_id, **kw)
    if kw.get("kl", 0.0) == 0.0:
        for c in m["benchmarks"]["kl_to_base"]["conditions"].values():
            c["metrics"]["kl_per_token"].update(low=0.0, high=0.0)
    (path / "utility_metrics.json").write_text(json.dumps(m), encoding="utf-8")
    config = {
        "run_id": run_id,
        "seed": 0,
        "adapter_run": m["adapter_run"],
        "subsets": {"mmlu": {"n": 300}, "ifeval": {"n": 150}},
    }
    (path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return path

@pytest.fixture
def utility_dir(tmp_path):
    d = tmp_path / "utility"
    _utility_run(d, "20261109-100000_gemma3-12b-it_base")
    _utility_run(
        d,
        "20261109-120000_gemma3-12b-it_fsa",
        adapter="outputs/training/20261024-184429_fsa_gemma12b_fullmix/adapter",
        method="fsa",
        kl=0.012,
    )
    bad = d / "20261109-130000_gemma3-12b-it_b3"
    bad.mkdir()
    (bad / "utility_metrics.json").write_text('{"run_id": "x"}', encoding="utf-8")
    wrong = _utility_run(d, "20261109-140000_gemma3-12b-it_b3")
    _edit_json(wrong / "utility_metrics.json", lambda m: m.update(run_id="20261109-140000_other"))
    (d / "20261109-150000_gemma3-12b-it_running").mkdir()  # no metrics yet: silently skipped
    (d / "compare_a_vs_b.md").write_text("| table |\n", encoding="utf-8")
    return d
