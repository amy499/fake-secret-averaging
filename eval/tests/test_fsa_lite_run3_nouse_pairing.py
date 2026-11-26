"""FSA-lite run3 without the USE term (POST-HOC ABLATION, Gemma only;
configs/eval/fsa_lite_run3_nouse.approach.yaml): its own output folder, its original variant
identical to approach fsa's (so stats.team_table pairs it with the existing fsa original runs), run3's
used-secrets file, the adapter path of its own run, the pinned sha256 of the selected adapter (step 500,
rule_met true; no placeholder left), no Qwen adapter (a Qwen technique run refuses), and run3's spec
unchanged."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import protocol as P
from stats import team_table as TT
from stats.rescore import rescore_run
from tests.test_fsa_lite_pairing import FSA_GEMMA_ORIGINAL_DRY_SETTINGS_442D6FC, _settings

NAME = "fsa_lite_run3_nouse"


def test_nouse_spec_paths_and_files():
    nu, r3, fsa = P.load_approach(NAME), P.load_approach("fsa_lite_run3"), P.load_approach("fsa")
    assert nu["name"] == NAME and nu["original_from"] == "fsa" and nu["defence"] == r3["defence"] and nu["post_hoc"] is True
    g = nu["models"]["gemma"]["defence"]
    assert g["adapter"] == {"local": "gemma_fsa_lite_run3_nouse"}
    assert g["adapter_sha256"] == "63a6d34663c64b26d1b36b49c0221523300a457fe4f85a2b98152f14742ddbc3"  # selected step 500
    assert nu["models"]["gemma"]["model"] == r3["models"]["gemma"]["model"]
    assert nu["models"]["qwen"]["defence"]["todo"].startswith("Gemma-only post-hoc ablation")
    # run3's data and dev split: the same used-secrets file; the shared held-out file
    assert nu["synthetic"] == r3["synthetic"] and nu["secret_in_context"] == fsa["secret_in_context"]
    # run3's spec is untouched (its pinned Gemma adapter stays)
    assert r3["models"]["gemma"]["defence"]["adapter_sha256"].startswith("324d8d8d")
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


def test_nouse_qwen_technique_refuses(tmp_path):
    with pytest.raises(RuntimeError, match="Gemma-only post-hoc ablation"):
        P.run_protocol(NAME, "qwen", "technique", benchmarks=["password_eval_team"], dry_run=True, output_dir=tmp_path)


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
