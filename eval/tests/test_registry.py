"""Registry lookup for every slot, and every name used in configs/ resolves."""

from __future__ import annotations

from pathlib import Path

import pytest

from core import registry
from core.config import ExperimentConfig
from core.interfaces import Attacker, Dataset, Defence, Metric, ModelWrapper, SecretSource

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = sorted((ROOT.parent / "configs" / "attack").glob("*.yaml"))

TIER1 = {
    "model": ["gemma3-12b-it", "dummy"],
    "secret": ["digits_4", "private_facts"],
    "benchmark": ["password_eval", "tensor_trust_hijacking", "tensor_trust_extraction", "hidden_code",
                  "code_check", "mmlu", "ifeval"],
    "defence": ["none", "prompt_instruction", "fsa"],
    "attacker": ["direct_request", "likelihood", "best_of_n", "tensor_trust_human", "covert_encoding"],
    "metric": ["password_eval", "tensor_trust", "code_metrics", "leakage", "joint_success", "utility_delta",
               "over_refusal", "latency", "attack_success_vs_queries"],
}  # fmt: skip
BASES = {"model": ModelWrapper, "secret": SecretSource, "benchmark": Dataset, "defence": Defence,
         "attacker": Attacker, "metric": Metric}  # fmt: skip


@pytest.mark.parametrize("kind,name", [(k, n) for k, ns in TIER1.items() for n in ns])
def test_tier1_slot_registered(kind, name):
    cls = registry.get(kind, name)
    assert issubclass(cls, BASES[kind])


def test_unknown_name_lists_known():
    with pytest.raises(KeyError, match="Registered"):
        registry.get("defence", "does_not_exist")


def test_every_registered_entry_instantiates():
    """Instantiate every registered class without loading weights or downloading data."""
    entries = registry.all_entries()
    secrets = registry.get("secret", "digits_4")()
    for name, cls in entries["model"].items():
        m = cls({"device": "cpu"})
        assert isinstance(m, ModelWrapper), name
    for name, cls in entries["secret"].items():
        assert isinstance(cls(), SecretSource), name
    for name, cls in entries["benchmark"].items():
        assert isinstance(cls({}, secrets, dry_run=True), Dataset), name
    for name, cls in entries["defence"].items():
        assert isinstance(cls({}), Defence), name
    for name, cls in entries["attacker"].items():
        assert isinstance(cls({}, None), Attacker), name
    for name, cls in entries["metric"].items():
        assert isinstance(cls({}, None), Metric), name


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_names_resolve(path):
    cfg = ExperimentConfig.load(path)
    registry.get("model", cfg.model["name"])
    registry.get("secret", cfg.secret["name"])
    registry.get("benchmark", cfg.benchmark["name"])
    registry.get("defence", cfg.defence["name"])
    for a in cfg.attackers:
        registry.get("attacker", a["name"])
    for m in cfg.metrics:
        registry.get("metric", m["name"])


def test_no_shadowing_folder_names():
    assert not (ROOT / "secrets").exists()
    assert not (ROOT / "datasets").exists()
    import secrets as stdlib_secrets

    assert hasattr(stdlib_secrets, "token_hex")
