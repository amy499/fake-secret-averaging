"""analysis/release_logprob (H4, report-only): dry run on a tiny random Gemma 3 with a random LoRA:
every turn of every drawn scenario is scored with and without the adapter, the stats are paired by
scenario, and no fixture string (password, info, prompt) reaches the output."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")

from analysis.release_logprob import run as RL  # noqa: E402
from benchmarks.password_eval import _FIXTURES  # noqa: E402


def test_dry_run_scores_every_turn_and_prints_no_strings(tmp_path, capsys):
    cfg = {"approach": "fsa_lite_run3", "n_scenarios": 4, "seed": 2026, "output_dir": str(tmp_path)}
    res = RL.run(cfg, "gemma", None, True, tmp_path)
    assert res["n_scenarios"] == 4 and res["turns"] == ["ask", "wrong", "right"] and res["correct_turn"] == "right"
    st = res["stats"]
    for t in res["turns"]:
        s = st["by_turn"][t]["sum"]
        assert s["delta"]["n"] == 4 and s["delta"]["mean"] != 0.0  # the adapter is on in one pass, off in the other
        assert s["delta"]["ci_low"] <= s["delta"]["mean"] <= s["delta"]["ci_high"]
    assert set(st["gap_vs_correct"]) == {"ask", "wrong"}
    g = st["gap_vs_correct"]["wrong"]["sum"]
    assert g["gap_change"]["mean"] == pytest.approx(g["adapter_gap"]["mean"] - g["base_gap"]["mean"])
    out = [p.read_text(encoding="utf-8") for p in (tmp_path / "gemma").iterdir()] + [capsys.readouterr().out]
    for text in out:
        for sid, pw, info, what in _FIXTURES:
            for s in (sid, pw, info, what):
                assert s.casefold() not in text.casefold()
    assert json.loads((tmp_path / "gemma" / "release_logprob.json").read_text(encoding="utf-8"))["frame"] == RL.FRAME


def test_select_items_is_seeded_and_keeps_all_turns():
    from core import registry

    registry.discover()
    items = registry.get("benchmark", "password_eval_team")({}, None, True).items("test")
    a, turns, correct = RL.select_items(items, 3, 7)
    b, _, _ = RL.select_items(items, 3, 7)
    assert [x.id for x in a] == [x.id for x in b] and len(a) == 9 and correct == "right"
    assert {x.metadata["cond"] for x in a} == set(turns)


def test_approach_override_writes_its_own_folder(tmp_path, monkeypatch):
    """--approach (e.g. the post-hoc ablation fsa_lite_run3_nouse) never overwrites the config approach's file."""
    seen = {}

    def fake_run(cfg, model_key, device, dry_run, out_root=None, log=print):
        seen.update(cfg)
        return {"stub": True}

    monkeypatch.setattr(RL, "run", fake_run)
    monkeypatch.setattr(RL, "markdown", lambda res: "")
    assert RL.main(["--model", "gemma", "--dry-run", "--approach", "fsa_lite_run3_nouse"]) == 0
    assert seen["approach"] == "fsa_lite_run3_nouse" and seen["_approach_override"] is True
    seen.clear()
    assert RL.main(["--model", "gemma", "--dry-run"]) == 0
    assert seen["approach"] == "fsa_lite_run3" and "_approach_override" not in seen


def test_override_folder_in_a_dry_run(tmp_path):
    cfg = {"approach": "fsa_lite_run3_nouse", "_approach_override": True, "n_scenarios": 2, "seed": 2026,
           "output_dir": str(tmp_path)}  # fmt: skip
    res = RL.run(cfg, "gemma", None, True, tmp_path, log=lambda *_: None)
    assert res["approach"] == "fsa_lite_run3_nouse"
    assert (tmp_path / "gemma" / "fsa_lite_run3_nouse" / "release_logprob.json").exists()
    assert not (tmp_path / "gemma" / "release_logprob.json").exists()
