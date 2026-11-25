"""Two-axis Playground registry: base models x methods (configs/training/models.yaml)."""

import pytest
import yaml

from fsa_runtime.registry import MODELS_YAML, RegistryError, load_registry


@pytest.fixture(scope="module")
def registry():
    return load_registry()


def test_base_models(registry):
    bm = registry.base_models
    assert list(bm) == [
        "qwen2.5-0.5b-instruct",
        "gemma3-4b-it",
        "gemma3-12b-it",
        "qwen2.5-7b-instruct",
        "olmo3-7b-instruct",
    ]
    assert bm["qwen2.5-0.5b-instruct"].hf_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert bm["gemma3-4b-it"].hf_id == "google/gemma-3-4b-it"
    assert bm["gemma3-12b-it"].hf_id == "google/gemma-3-12b-it"
    assert bm["qwen2.5-7b-instruct"].hf_id == "Qwen/Qwen2.5-7B-Instruct"
    olmo = bm["olmo3-7b-instruct"]
    assert olmo.optional and not olmo.enabled


def test_methods(registry):
    m = registry.methods
    assert list(m) == [
        "base",
        "prompt_hiding",
        "prompt_hiding_strict",
        "prompt_anti_suppression",
        "fsa",
        "b3",
        "random_layer",
    ]
    assert m["base"].label == "Base (no technique)"
    assert [k for k, v in m.items() if not v.enabled] == ["b3", "random_layer"]


def test_spec_combines_both_axes(registry):
    spec = registry.spec("gemma3-12b-it", "fsa")
    assert (spec.base_model, spec.hf_id, spec.method) == (
        "gemma3-12b-it",
        "google/gemma-3-12b-it",
        "fsa",
    )
    assert spec.id == "gemma3-12b-it/fsa"


@pytest.mark.parametrize(
    "base_model,method,msg",
    [
        ("nope", "base", "unknown base model"),
        ("olmo3-7b-instruct", "base", "disabled"),
        ("gemma3-4b-it", "nope", "unknown method"),
        ("gemma3-4b-it", "b3", "disabled"),
        ("gemma3-4b-it", "patching", "unknown method"),  # not a Playground method
    ],
)
def test_spec_rejects(registry, base_model, method, msg):
    with pytest.raises(RegistryError, match=msg):
        registry.spec(base_model, method)


def test_attacker_not_in_registry():
    raw = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8"))
    ids = [e["id"] for e in raw["base_models"]] + [e["id"] for e in raw["methods"]]
    assert set(raw) == {"base_models", "methods"}
    assert not any("1.5b" in i for i in ids)


def test_adapters_feed_the_spec(registry):
    assert registry.spec("qwen2.5-0.5b-instruct", "fsa").adapter is None  # none configured yet
    bm = registry.base_models["qwen2.5-0.5b-instruct"].model_copy(
        update={"adapters": {"fsa": "outputs/adapters/x"}}
    )
    path = bm.adapter_for("fsa")
    assert path.replace("\\", "/").endswith("outputs/adapters/x") and bm.adapter_for("b3") is None
    assert "adapters" not in bm.info().model_dump()  # not exposed by /api/models
