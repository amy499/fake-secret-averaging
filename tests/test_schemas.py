import math

import pytest
from pydantic import ValidationError

from fsa.contract.schemas import (
    GenerateRequest,
    LeakageMetrics,
    PatchingMetrics,
    RunConfig,
    RunMetrics,
    Sample,
)

RUN_ID = "20261017-120000_test"


def test_valid_metrics_roundtrip(factories):
    m = RunMetrics.model_validate(factories.metrics(run_id=RUN_ID))
    again = RunMetrics.model_validate_json(m.model_dump_json())
    assert again == m
    assert again.schema_version == 2 and again.run_type == "evaluation"


def test_is_mock_required(factories):
    data = factories.metrics(run_id=RUN_ID)
    del data["is_mock"]
    with pytest.raises(ValidationError, match="is_mock"):
        RunMetrics.model_validate(data)


def test_unknown_fields_rejected(factories):
    data = factories.metrics(run_id=RUN_ID)
    data["leakage"]["bits_leaked"] = 1.0  # old name must not slip through
    with pytest.raises(ValidationError):
        RunMetrics.model_validate(data)


def test_bad_run_id_rejected(factories):
    with pytest.raises(ValidationError):
        RunMetrics.model_validate(factories.metrics(run_id="../etc"))


def test_utility_keys_required_but_nullable(factories):
    data = factories.metrics(run_id=RUN_ID)
    data["utility"] = {"kl_to_base_nonsecret": None, "task_score": None}
    RunMetrics.model_validate(data)
    data["utility"] = {"kl_to_base_nonsecret": 0.1}
    with pytest.raises(ValidationError, match="task_score"):
        RunMetrics.model_validate(data)


def test_non_patching_run_needs_leakage_and_utility(factories):
    data = factories.metrics(run_id=RUN_ID)
    del data["utility"]
    with pytest.raises(ValidationError, match="leakage and utility"):
        RunMetrics.model_validate(data)


def test_patching_only_run_is_valid(factories):
    RunMetrics.model_validate(
        {
            "run_id": RUN_ID,
            "run_type": "evaluation",
            "is_mock": True,
            "patching": factories.patching(),
        }
    )


def test_bits_must_match_cross_entropy(factories):
    data = factories.leakage()
    data["bits_leaked_lower_bound"] += 0.5
    with pytest.raises(ValidationError, match="bits_leaked_lower_bound"):
        LeakageMetrics.model_validate(data)


def test_bits_clipped_at_zero(factories):
    data = factories.leakage(mean_ce=math.log2(100) + 2)  # attacker worse than uniform
    assert LeakageMetrics.model_validate(data).bits_leaked_lower_bound == 0.0


def test_kl_upper_bound_nullable_and_nonnegative(factories):
    data = factories.leakage()
    data["kl_upper_bound_bits"] = 1.2
    LeakageMetrics.model_validate(data)
    data["kl_upper_bound_bits"] = -0.1
    with pytest.raises(ValidationError):
        LeakageMetrics.model_validate(data)


def test_kl_upper_bound_must_not_be_below_lower_bound(factories):
    data = factories.leakage()  # lower bound = log2(100) - 6 ≈ 0.644
    data["kl_upper_bound_bits"] = data["bits_leaked_lower_bound"]  # equal is fine
    LeakageMetrics.model_validate(data)
    data["kl_upper_bound_bits"] = 0.5
    with pytest.raises(ValidationError, match="kl_upper_bound_bits"):
        LeakageMetrics.model_validate(data)


@pytest.mark.parametrize("field", ["eval_split", "attacker_trained_fresh"])
def test_eval_provenance_fields_required(factories, field):
    data = factories.leakage()
    del data[field]
    with pytest.raises(ValidationError, match=field):
        LeakageMetrics.model_validate(data)


def test_eval_split_non_empty(factories):
    data = factories.leakage()
    data["eval_split"] = ""
    with pytest.raises(ValidationError):
        LeakageMetrics.model_validate(data)


@pytest.mark.parametrize("field,bad", [("exact_match", 0.05), ("top5", 0.01), ("per_digit", 0.5)])
def test_chance_levels_checked(factories, field, bad):
    data = factories.leakage()
    data["attacker"][field]["chance"] = bad
    with pytest.raises(ValidationError, match=field):
        LeakageMetrics.model_validate(data)


def test_patching_shape_checked(factories):
    data = factories.patching(n_layers=6)
    data["by_component"]["scores"] = data["by_component"]["scores"][:5]
    with pytest.raises(ValidationError, match="by_component"):
        PatchingMetrics.model_validate(data)


def test_patching_columns_checked(factories):
    data = factories.patching(n_layers=6)
    data["by_position"]["columns"] = ["a", "b", "c"]
    with pytest.raises(ValidationError, match="by_position"):
        PatchingMetrics.model_validate(data)


def test_patching_position_view_optional(factories):
    data = factories.patching(n_layers=6)
    del data["by_position"]
    PatchingMetrics.model_validate(data)


def test_config_allows_extra_keys(factories):
    cfg = RunConfig.model_validate(factories.config())
    assert cfg.fsa.k == 8 and cfg.model_extra["eval"]["n_replies_per_secret"] == 20


def test_sample_digit_counts_length(factories):
    data = factories.sample()
    data["digit_counts"] = [0] * 9
    with pytest.raises(ValidationError):
        Sample.model_validate(data)


P_BASE = {"base_model": "gemma3-4b-it", "method": "base"}
P_FSA = {"base_model": "gemma3-12b-it", "method": "fsa"}


def test_generate_request_limits():
    ok = GenerateRequest(panels=[P_BASE, P_FSA], prompt="hi", n_samples=5, seed=1, temperature=0.7)
    assert ok.secret is None and ok.panels[1].base_model == "gemma3-12b-it"
    with pytest.raises(ValidationError):
        GenerateRequest(panels=[], prompt="hi", n_samples=5)
    with pytest.raises(ValidationError):
        GenerateRequest(panels=[P_BASE, P_FSA, P_BASE], prompt="hi", n_samples=5)
    with pytest.raises(ValidationError):
        GenerateRequest(panels=[P_BASE], prompt="hi", n_samples=5, secret="3x")
    with pytest.raises(ValidationError):
        GenerateRequest(panels=[P_BASE], prompt="hi", n_samples=5, temperature=0)
    with pytest.raises(ValidationError):  # method must be a known Method
        GenerateRequest(panels=[{**P_BASE, "method": "nope"}], prompt="hi", n_samples=5)
    with pytest.raises(ValidationError):  # old flat shape is gone
        GenerateRequest.model_validate({"models": ["base"], "prompt": "hi", "n_samples": 5})
