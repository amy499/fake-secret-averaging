"""Factories for valid configs, metrics and samples used across tests, plus the cached
Qwen2.5 tokenizer and a tiny random-weight Qwen2 model (skipped when unavailable)."""

import copy
import math

import pytest

N = 100
MEAN_CE = 6.0


def make_config(name="gemma3_4b_fsa_k8", method="fsa", k=8, layers=(14, 15, 16, 17)):
    cfg = {
        "name": name,
        "seed": 0,
        "model": {"base": "google/gemma-3-4b-it", "n_layers": 34},
        "method": method,
        "eval": {"n_replies_per_secret": 20},
    }
    if k is not None:
        cfg["fsa"] = {"k": k}
    if layers is not None:
        cfg["lora"] = {"layers": list(layers), "rank": 8}
    return cfg


def make_leakage(mean_ce=MEAN_CE):
    return {
        "secret_space_size": N,
        "n_secrets_evaluated": 100,
        "n_replies_per_secret": 20,
        "eval_split": "heldout",
        "attacker_trained_fresh": True,
        "attacker": {
            "name": "qwen2.5-1.5b-lora",
            "exact_match": {"value": 0.03, "chance": 1 / N},
            "top5": {"value": 0.09, "chance": 5 / N},
            "per_digit": {"value": 0.14, "chance": 0.1},
        },
        "attacker_mean_ce_bits": mean_ce,
        "bits_leaked_lower_bound": max(0.0, math.log2(N) - mean_ce),
        "kl_upper_bound_bits": None,
        "digit_kl_to_uniform_bits": 0.002,
    }


def make_metrics(is_mock=True, run_id=None):
    m = {
        "run_type": "evaluation",
        "is_mock": is_mock,
        "leakage": make_leakage(),
        "utility": {"kl_to_base_nonsecret": 0.031, "task_score": 0.92},
    }
    if run_id:
        m["run_id"] = run_id
    return m


def make_patching(n_layers=34):
    return {
        "metric_name": "todo",
        "n_layers": n_layers,
        "layer_types": ["global" if (i + 1) % 6 == 0 else "local" for i in range(n_layers)],
        "by_component": {
            "columns": ["resid", "attn", "mlp"],
            "scores": [[0.1, 0.2, 0.3]] * n_layers,
        },
        "by_position": {
            "columns": ["secret_span", "post_secret_prompt", "last_token"],
            "scores": [[0.5, 0.1, 0.0]] * n_layers,
        },
    }


def make_sample(i=0, secret="37"):
    return {
        "model": "fsa",
        "secret": secret,
        "prompt_id": "rand_20_2digit",
        "prompt": "Generate 20 random 2-digit numbers.",
        "reply_index": i,
        "reply": "12, 58",
        "numbers": [12, 58],
        "n_numbers_parsed": 2,
        "digit_counts": [0, 1, 1, 0, 0, 1, 0, 0, 1, 0],
    }


@pytest.fixture
def factories():
    """Access the factories from tests; returns deep copies so tests can mutate freely."""

    class F:
        config = staticmethod(lambda **kw: copy.deepcopy(make_config(**kw)))
        leakage = staticmethod(lambda **kw: copy.deepcopy(make_leakage(**kw)))
        metrics = staticmethod(lambda **kw: copy.deepcopy(make_metrics(**kw)))
        patching = staticmethod(lambda **kw: copy.deepcopy(make_patching(**kw)))
        sample = staticmethod(lambda **kw: copy.deepcopy(make_sample(**kw)))

    return F


QWEN_HF = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="session")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(QWEN_HF, local_files_only=True)
    except OSError:
        pytest.skip("Qwen2.5-0.5B tokenizer not cached")


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory, tok):
    torch = pytest.importorskip("torch")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=512,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    path = tmp_path_factory.mktemp("tiny_qwen2")
    Qwen2ForCausalLM(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return path
