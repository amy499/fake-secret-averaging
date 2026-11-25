"""Training-startup prompt check: no training user prompt may be unknown, declared held-out by
the config (prompts.heldout_user_prompts), or marked evaluation-only in prompts.yaml
(eval_only_prompts, or a prompt set with eval_only: true); general_prompts.train and .eval must
not overlap. Tiny made-up prompts.yaml files; prepare() runs, but no model is loaded."""

from pathlib import Path

import pytest
import yaml

from fsa.eval.prompt_sets import check_train_prompts, eval_only_prompts
from fsa.training.config import TrainConfig
from scripts import train_fsa

ROOT = Path(__file__).resolve().parents[1]

RAW = {
    "user_prompts": [
        {"id": "rand_20_2digit", "text": "Give 20 numbers.", "n_numbers": 20},
        {"id": "rand_10_2digit", "text": "Give 10 numbers.", "n_numbers": 10},
        {"id": "lottery", "text": "Pick lottery numbers.", "n_numbers": 6},
        {"id": "task_a", "text": "Summarise this paragraph."},
        {"id": "task_b", "text": "Translate this sentence."},
    ],
    "eval_only_prompts": ["lottery"],
    "prompt_sets": {
        "tasks": {"eval_only": True, "prompts": ["task_a", "task_b"]},
        "numbers": {"prompts": ["rand_20_2digit", "rand_10_2digit"]},  # not eval-only
    },
    "general_prompts": {
        "train": ["Explain why the sky is blue.", "Write a haiku."],
        "eval": ["Describe a bicycle.", "Name a robot."],
    },
}


def make_cfg(train=("rand_20_2digit", "rand_10_2digit"), heldout=("lottery",)) -> TrainConfig:
    return TrainConfig.model_validate(
        {
            "name": "tiny",
            "method": "fsa",
            "model": {"id": "tiny"},
            "secrets": {"split_seed": 0, "n_train": 4, "n_heldout": 4},
            "fsa": {"k": 2},
            "prompts": {
                "train_system_methods": ["base"],
                "train_user_prompts": list(train),
                "heldout_user_prompts": list(heldout),
            },
        }
    )


@pytest.fixture
def prompts_yaml(tmp_path, monkeypatch):
    """prepare() reads this file instead of configs/training/prompts.yaml; returns a writer."""
    path = tmp_path / "prompts.yaml"

    def write(raw):
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    write(RAW)
    monkeypatch.setattr(train_fsa, "PROMPTS_YAML", path)
    return write


def test_correct_config_passes(prompts_yaml):
    data, resolved = train_fsa.prepare(make_cfg())
    assert sorted(data.user_prompts) == ["rand_10_2digit", "rand_20_2digit"]
    assert resolved["heldout_user_prompts"] == ["lottery", "task_a", "task_b"]


def test_training_on_lottery_fails(prompts_yaml):
    # Declared held-out by the config and listed in eval_only_prompts: both reasons named.
    with pytest.raises(
        ValueError,
        match=r"train prompt check failed: train_user_prompts also in the config's "
        r"heldout_user_prompts: \['lottery'\]; train_user_prompts marked evaluation-only in "
        r"prompts.yaml: lottery \(eval_only_prompts\)",
    ):
        train_fsa.prepare(make_cfg(train=["rand_20_2digit", "lottery"]))
    # Either marking alone is enough.
    with pytest.raises(ValueError, match=r"lottery \(eval_only_prompts\)"):
        train_fsa.prepare(make_cfg(train=["lottery"], heldout=[]))
    prompts_yaml({**RAW, "eval_only_prompts": []})
    with pytest.raises(ValueError, match=r"heldout_user_prompts: \['lottery'\]$"):
        train_fsa.prepare(make_cfg(train=["lottery"]))


def test_training_on_a_prompt_of_an_eval_only_set_fails(prompts_yaml):
    with pytest.raises(
        ValueError,
        match=r"marked evaluation-only in prompts.yaml: task_b \(prompt set 'tasks' "
        r"\(eval_only\)\)",
    ):
        train_fsa.prepare(make_cfg(train=["rand_20_2digit", "task_b"]))


def test_unknown_prompt_id_fails(prompts_yaml):
    with pytest.raises(
        ValueError,
        match=r"unknown train_user_prompts \['nope'\]; known: \['lottery', 'rand_10_2digit'",
    ):
        train_fsa.prepare(make_cfg(train=["rand_20_2digit", "nope"]))


def test_overlapping_general_prompts_fail(prompts_yaml):
    general = {
        "train": ["Write a haiku.", "Describe a bicycle."],
        "eval": ["describe  a BICYCLE.", "Name a robot."],
    }  # case and spaces ignored
    prompts_yaml({**RAW, "general_prompts": general})
    with pytest.raises(
        ValueError,
        match=r"general_prompts.train also in general_prompts.eval: \['Describe a bicycle.'\]",
    ):
        train_fsa.prepare(make_cfg())


def test_all_problems_reported_at_once():
    raw = {**RAW, "general_prompts": {"train": ["x"], "eval": ["x"]}}
    with pytest.raises(ValueError) as e:
        check_train_prompts(raw, ["nope", "lottery", "task_a"], heldout_user_prompts=["lottery"])
    msg = str(e.value)
    for part in (
        "unknown train_user_prompts ['nope']",
        "heldout_user_prompts: ['lottery']",
        "lottery (eval_only_prompts)",
        "task_a (prompt set 'tasks' (eval_only))",
        "general_prompts.train also in general_prompts.eval: ['x']",
    ):
        assert part in msg


def test_eval_only_markings_are_validated():
    assert eval_only_prompts(RAW) == {
        "lottery": ["eval_only_prompts"],
        "task_a": ["prompt set 'tasks' (eval_only)"],
        "task_b": ["prompt set 'tasks' (eval_only)"],
    }
    with pytest.raises(ValueError, match=r"eval_only_prompts: unknown user prompt ids \['zz'\]"):
        eval_only_prompts({**RAW, "eval_only_prompts": ["zz"]})
    with pytest.raises(ValueError, match="must be a list"):
        eval_only_prompts({**RAW, "eval_only_prompts": "lottery"})
    sets = {"tasks": {"eval_only": "yes", "prompts": ["task_a"]}}
    with pytest.raises(ValueError, match="eval_only must be true or false"):
        eval_only_prompts({**RAW, "prompt_sets": sets})
    sets = {"tasks": {"eval_only": True, "prompts": ["task_a", "ghost"]}}
    with pytest.raises(ValueError, match=r"unknown user prompt ids \['ghost'\]"):
        eval_only_prompts({**RAW, "prompt_sets": sets})


# --------------------------------------------------------------------------- real files


UB_SET = [
    "ub_binomial_120_l25",
    "ub_binomial_qc_120_l394",
    "ub_hypergeom_ballots_l27",
    "ub_hypergeom_boxes_l396",
]


def test_real_prompts_yaml_marks_lottery_and_general_prompts_are_disjoint():
    raw = yaml.safe_load((ROOT / "configs" / "training" / "prompts.yaml").read_text(encoding="utf-8"))
    set_reason = ["prompt set 'unpredictabench_2digit' (eval_only)"]
    pe_reason = ["prompt set 'pe_access_eval' (eval_only)"]  # utility_eval pe_access (digits4)
    assert eval_only_prompts(raw) == {
        "lottery": ["eval_only_prompts"],
        "verify_secret_code": ["eval_only_prompts"],  # scripts/utility_eval.py verification
        "lottery_4digit": ["eval_only_prompts"],
        "verify_access_code_4digit": ["eval_only_prompts"],  # digits4 verification
        **{p: set_reason for p in UB_SET},
        **{
            p: pe_reason
            for p in ("compliant_eval_1", "compliant_eval_2", "withhold_eval_1", "withhold_eval_2")
        },
    }
    check_train_prompts(raw, ["rand_20_2digit", "rand_10_2digit"])


@pytest.mark.parametrize(
    "path",
    sorted(p for p in (ROOT / "configs" / "training" / "experiments").glob("*.yaml") if not p.name.startswith(("fsa_v2_", "fsa_lite_"))),
    ids=lambda p: p.name,
)  # fsa_v2_* and fsa_lite_* configs have their own schema (src/fsa_lite/lite/v2_train.py; fsa_v2 tested in
# test_train_fsa.py, fsa_lite in src/fsa_lite/lite/selftest_v2.py)
def test_experiment_configs_pass_the_check(path):
    cfg = TrainConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    lottery = "lottery_4digit" if cfg.secrets.type == "digits4" else "lottery"
    assert cfg.prompts.heldout_user_prompts[0] == lottery
    data, resolved = train_fsa.prepare(cfg)  # real prompts.yaml; raises on any problem
    raw = yaml.safe_load((ROOT / "configs" / "training" / "prompts.yaml").read_text(encoding="utf-8"))
    eval_only = set(eval_only_prompts(raw))
    assert not set(data.user_prompts) & eval_only  # never trains on an evaluation-only prompt
    assert not set(data.need_prompts) & eval_only  # nor uses one in the need term
    assert eval_only <= set(resolved["heldout_user_prompts"])


def test_multiprompt_config_matches_fullmix_except_prompts_and_replies():
    load = lambda n: TrainConfig.model_validate(
        yaml.safe_load((ROOT / "configs" / "training" / "experiments" / f"{n}.yaml").read_text(encoding="utf-8"))
    )
    new, old = load("fsa_gemma12b_fullmix_multiprompt"), load("fsa_gemma12b_fullmix")
    a, b = new.model_dump(), old.model_dump()
    differing = [
        f"{sec}.{k}" if isinstance(a[sec], dict) else sec
        for sec in a
        for k in (a[sec] if isinstance(a[sec], dict) else [None])
        if (a[sec][k] if k else a[sec]) != (b[sec][k] if k else b[sec])
    ]
    assert differing == [
        "name",
        "prompts.train_user_prompts",
        "prompts.heldout_user_prompts",
        "generation.offline_replies_per_prompt",
    ]
    assert (
        len(new.prompts.train_user_prompts) == 13 and new.generation.offline_replies_per_prompt == 2
    )
    assert new.prompts.heldout_user_prompts == ["lottery", *UB_SET]
