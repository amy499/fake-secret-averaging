"""FSA-lite run3 without the USE term (configs/eval/fsa_lite_run3_nouse.approach.yaml): Gemma a
POST-HOC ABLATION, Qwen a PRE-REGISTERED test (docs/prereg_qwen_nouse.md). Its own output folder, its
original variant identical to approach fsa's (so stats.team_table pairs it with the existing fsa
original runs, for both models), run3's used-secrets file, the adapter path of each model's own run,
Gemma's pinned sha256 of the selected adapter (step 500, rule_met true; no placeholder left), Qwen's
FILL_ME pin (a real Qwen technique run refuses until it is pinned), and run3's spec unchanged."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from core import protocol as P
from stats import team_table as TT
from stats.rescore import rescore_run

from tests.test_fsa_lite_pairing import FSA_GEMMA_ORIGINAL_DRY_SETTINGS_442D6FC, _settings

NAME = "fsa_lite_run3_nouse"


def test_nouse_spec_paths_and_files():
    nu, r3, fsa = P.load_approach(NAME), P.load_approach("fsa_lite_run3"), P.load_approach("fsa")
    assert nu["name"] == NAME and nu["original_from"] == "fsa" and nu["defence"] == r3["defence"]
    assert "post_hoc" not in nu  # the standing is per model: Gemma post hoc, Qwen pre-registered
    gm, qm = nu["models"]["gemma"], nu["models"]["qwen"]
    assert gm["post_hoc"] is True and "pre_registered" not in gm
    assert qm["post_hoc"] is False and qm["pre_registered"] == "docs/prereg_qwen_nouse.md"
    prereg = (P.approach_path(NAME).parents[2] / qm["pre_registered"]).read_text(encoding="utf-8")
    assert "2026-11-28" in prereg and "fsa_lite_qwen7b_run3_nouse" in prereg
    g = gm["defence"]
    assert g["adapter"] == {"local": "gemma_fsa_lite_run3_nouse"}
    assert g["adapter_sha256"] == "63a6d34663c64b26d1b36b49c0221523300a457fe4f85a2b98152f14742ddbc3"  # selected step 500
    assert qm["defence"] == {"adapter": {"local": "qwen_fsa_lite_run3_nouse"}, "adapter_sha256": "FILL_ME"}
    for key in ("gemma", "qwen"):
        assert nu["models"][key]["model"] == r3["models"][key]["model"]
    # run3's data and dev split: the same used-secrets file; the shared held-out file
    assert nu["synthetic"] == r3["synthetic"] and nu["secret_in_context"] == fsa["secret_in_context"]
    # run3's spec is untouched (its pinned adapters stay)
    assert r3["models"]["gemma"]["defence"]["adapter_sha256"].startswith("324d8d8d")
    assert r3["models"]["qwen"]["defence"]["adapter_sha256"].startswith("d5f8ca21")
    assert P.approach_path(NAME).read_bytes().count(b"\r\n") == 0


def test_nouse_original_is_fsa_original_and_outputs_are_separate(tmp_path):
    nu, fsa = P.load_approach(NAME), P.load_approach("fsa")
    for key in P.MODEL_KEYS:
        assert P.variant_configs(nu, key, "original") == P.variant_configs(fsa, key, "original")
    assert P.check_original_from(nu)["approach"] == "fsa"
    out = tmp_path / "out"
    fo = P.run_protocol("fsa", "gemma", "original", suite="core", dry_run=True, output_dir=out)
    nt = P.run_protocol(NAME, "gemma", "technique", suite="core", dry_run=True, output_dir=out)
    assert all(d.parts[-4:-2] == (NAME, "gemma") for d in nt) and all(d.parts[-4] == "fsa" for d in fo)
    assert _settings(fo) == _settings(nt) == FSA_GEMMA_ORIGINAL_DRY_SETTINGS_442D6FC
    s = json.loads((nt[0] / "summary.json").read_text(encoding="utf-8"))
    assert s["approach"] == NAME and s["original_from"]["approach"] == "fsa"
    assert s["defence_info"]["placeholders"] == []  # the ablation adapter is pinned: no placeholder left


@pytest.mark.parametrize("model", ["gemma", "qwen"])
def test_nouse_original_pairs_for_both_models(model, tmp_path):
    """Both models' original variant is fsa's (dry run): settings equal, separate output folders."""
    out = tmp_path / "out"
    fo = P.run_protocol("fsa", model, "original", suite="core", dry_run=True, output_dir=out)
    nt = P.run_protocol(NAME, model, "technique", suite="core", dry_run=True, output_dir=out)
    assert _settings(fo) == _settings(nt)
    assert all(d.parts[-4:-2] == (NAME, model) for d in nt)


def test_nouse_qwen_technique_refuses_until_pinned(tmp_path):
    """The dry run records the placeholder; a real Qwen technique run refuses before any model loads."""
    from experiments._bridges import fsa as F

    nt = P.run_protocol(NAME, "qwen", "technique", benchmarks=["password_eval_team"], dry_run=True, output_dir=tmp_path)
    s = json.loads((nt[0] / "summary.json").read_text(encoding="utf-8"))
    assert s["defence_info"]["placeholders"] == ["adapter_sha256"]
    _, dcfg = P.variant_configs(P.load_approach(NAME), "qwen", "technique")
    bridge = F.FSABridge(dcfg)
    bridge.prepare([], None)
    with pytest.raises(F.AdapterError, match="adapter_sha256 for this model still FILL_ME"):
        bridge.wrap(SimpleNamespace(is_dummy=False))  # refused on the spec alone: never touches a model


def test_team_table_pairs_nouse_qwen_with_the_fsa_original(tmp_path):
    """The pre-registered Qwen run evaluates the core suite: every row pairs with fsa's Qwen original."""
    out = tmp_path / "protocol"
    fo = P.run_protocol("fsa", "qwen", "original", suite="core", dry_run=True, output_dir=out)
    nt = P.run_protocol(NAME, "qwen", "technique", suite="core", dry_run=True, output_dir=out)
    for d in fo + nt:
        rescore_run(d)
    rows = [r for r in TT.build(out, approaches=[NAME], models=["qwen"], suite="core")["rows"] if r["metric"] != "-"]
    assert {r["benchmark"] for r in rows} == set(P.SUITES["core"])
    for r in rows:
        assert r["approach"] == NAME and r["original_from"] == "fsa"
        assert Path(r["original_dir"]).parts[-4] == "fsa" and Path(r["technique_dir"]).parts[-4] == NAME
        assert "not paired" not in r["status"] and "missing" not in r["status"], r["status"]


def test_team_table_pairs_nouse_password_eval_with_the_fsa_original(tmp_path):
    """The ablation runs PasswordEval only: that row pairs with fsa's original, the rest show missing."""
    out = tmp_path / "protocol"
    fo = P.run_protocol("fsa", "gemma", "original", suite="core", dry_run=True, output_dir=out)
    nt = P.run_protocol(NAME, "gemma", "technique", benchmarks=["password_eval_team"], dry_run=True, output_dir=out)
    for d in fo + nt:
        rescore_run(d)
    rows = [r for r in TT.build(out, approaches=[NAME], models=["gemma"], suite="core")["rows"] if r["metric"] != "-"]
    pe = [r for r in rows if r["benchmark"] == "password_eval_team"]
    assert pe and all(r["approach"] == NAME and r["original_from"] == "fsa" for r in pe)
    for r in pe:
        assert Path(r["original_dir"]).parts[-4] == "fsa" and Path(r["technique_dir"]).parts[-4] == NAME
        assert "not paired" not in r["status"] and "missing" not in r["status"], r["status"]
    allrows = TT.build(out, approaches=[NAME], models=["gemma"], suite="core")["rows"]
    others = [r for r in allrows if r["benchmark"] != "password_eval_team"]
    assert others and all(r["status"].startswith("MISSING") for r in others)  # shown, never dropped (check 4)
