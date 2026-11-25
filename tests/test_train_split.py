"""Startup train-split check: no held-out secret is used as a real or a fake secret in training.
Tiny made-up data; no model is loaded."""

import pytest

from fsa.secrets import check_saved_split, check_train_split
from fsa.training.config import TrainConfig
from fsa.training.trainer import TrainData, fake_sets

TRAIN = ["03", "17", "42", "88"]
HELDOUT = ["05", "61", "70", "99"]


def make_cfg(**over) -> TrainConfig:
    raw = {
        "name": "tiny",
        "method": "fsa",
        "model": {"id": "tiny"},
        "secrets": {"split_seed": 0, "n_train": 4, "n_heldout": 4},
        "fsa": {"k": 2},
        **over,
    }
    return TrainConfig.model_validate(raw)


def make_data(train=TRAIN) -> TrainData:
    return TrainData(
        train_secrets=list(train),
        system_methods=["base"],
        user_prompts={"p": "pick a number"},
        general_train=[],
        general_eval=[],
        templates={},
    )


@pytest.mark.parametrize(
    "over",
    [
        {"fsa": {"k": 2}},  # sampled k
        {"fsa": {"k": 3, "fake_set": "all_others"}},  # full mixture
        {"method": "b3", "target": "no_secret"},  # B3: no fakes, real secrets only
    ],
)
def test_correct_split_passes(over):
    sets = fake_sets(make_cfg(**over), make_data())
    assert sorted(sets) == sorted(TRAIN)
    assert all(s not in fakes and set(fakes) <= set(TRAIN) for s, fakes in sets.items())
    check_train_split(sets, train=TRAIN, heldout=HELDOUT)


def test_heldout_secret_among_fakes_fails():
    sets = fake_sets(make_cfg(fsa={"k": 3, "fake_set": "all_others"}), make_data())
    sets["17"] = ["03", "61", "88"]  # 61 is held-out
    with pytest.raises(
        ValueError, match=r"held-out secrets: \['61'\] \(with real secrets \['17'\]\)"
    ):
        check_train_split(sets, train=TRAIN, heldout=HELDOUT)


# --------------------------------------------------------------------------- resume: saved split


def test_saved_train_split_equal_passes():
    check_saved_split("train", TRAIN, list(TRAIN))


def test_saved_train_split_different_secrets_fails():
    saved = ["03", "17", "42", "61"]  # 61 instead of 88
    with pytest.raises(
        ValueError,
        match=r"only in the saved config.yaml: \['61'\]; only in the recomputed split: \['88'\]",
    ):
        check_saved_split("train", saved, TRAIN)


def test_saved_train_split_different_order_fails():
    with pytest.raises(ValueError, match="same secrets, different order"):
        check_saved_split("train", list(reversed(TRAIN)), TRAIN)
