"""FSA-lite run3 (configs/eval/fsa_lite_run3.approach.yaml) on the team protocol: its own
output folder, its original variant identical to approach fsa's (so stats.team_table pairs it with the
existing fsa original runs), its own used-secrets file, the adapter paths, FILL_ME placeholders
recorded in a dry run (a real technique run refuses them), and the fsa / fsa_lite fingerprints
unchanged."""

from __future__ import annotations

import json
from pathlib import Path

from core import protocol as P
from stats import team_table as TT
from stats.rescore import rescore_run
from tests.test_fsa_lite_pairing import FSA_GEMMA_ORIGINAL_DRY_SETTINGS_442D6FC, _settings



def test_run3_spec_paths_and_files():
    r3, lite, fsa = P.load_approach("fsa_lite_run3"), P.load_approach("fsa_lite"), P.load_approach("fsa")
    assert r3["name"] == "fsa_lite_run3" and r3["original_from"] == "fsa" and r3["defence"] == lite["defence"]
    g, q = r3["models"]["gemma"]["defence"], r3["models"]["qwen"]["defence"]
    assert g["adapter"] == {"local": "gemma_fsa_lite_run3"}
    assert q["adapter"] == {"local": "qwen_fsa_lite_run3"}
    assert g["adapter_sha256"] == "324d8d8dc3659ffc45c12a38d5dc18e20e529b1b30bb3083100a227ce9f45d15"  # pinned run3 adapter (step 375)
    assert q["adapter_sha256"] == "d5f8ca21ce5751a7e585cf9e8b3aae9189753066abaad4e095ba2ff2d3e3720f"  # pinned run3 adapter (step 250)
    # its own used-secrets file (run3's training + selection secrets), the shared held-out file
    assert r3["synthetic"]["used_secrets"] == "private/fsa_lite_run3_used_secrets.txt"
    assert r3["synthetic"]["used_secrets"] not in (lite["synthetic"]["used_secrets"], fsa["synthetic"]["used_secrets"])
    assert r3["secret_in_context"] == fsa["secret_in_context"]
    # run2's spec is untouched by run3 (its pinned adapters stay)
    assert lite["models"]["gemma"]["defence"]["adapter"] == {"local": "gemma_fsa_lite"}
    assert lite["models"]["qwen"]["defence"]["adapter_sha256"].startswith("850df8a9")


def test_run3_original_is_fsa_original_and_outputs_are_separate(tmp_path):
    r3, fsa = P.load_approach("fsa_lite_run3"), P.load_approach("fsa")
    for key in P.MODEL_KEYS:
        assert P.variant_configs(r3, key, "original") == P.variant_configs(fsa, key, "original")
    assert P.check_original_from(r3)["approach"] == "fsa"
    out = tmp_path / "out"
    fo = P.run_protocol("fsa", "gemma", "original", suite="core", dry_run=True, output_dir=out)
    lt = P.run_protocol("fsa_lite_run3", "gemma", "technique", suite="core", dry_run=True, output_dir=out)
    assert all(d.parts[-4:-2] == ("fsa_lite_run3", "gemma") for d in lt) and all(d.parts[-4] == "fsa" for d in fo)
    assert _settings(fo) == _settings(lt) == FSA_GEMMA_ORIGINAL_DRY_SETTINGS_442D6FC
    s = json.loads((lt[0] / "summary.json").read_text(encoding="utf-8"))
    assert s["approach"] == "fsa_lite_run3" and s["original_from"]["approach"] == "fsa"
    assert s["defence_info"]["placeholders"] == []  # run3 adapters pinned: no placeholder left


def test_team_table_pairs_run3_with_the_fsa_original(tmp_path):
    out = tmp_path / "protocol"
    fo = P.run_protocol("fsa", "gemma", "original", suite="core", dry_run=True, output_dir=out)
    lt = P.run_protocol("fsa_lite_run3", "gemma", "technique", suite="core", dry_run=True, output_dir=out)
    for d in fo + lt:
        rescore_run(d)
    rows = [r for r in TT.build(out, approaches=["fsa_lite_run3"], models=["gemma"], suite="core")["rows"] if r["metric"] != "-"]
    assert {r["benchmark"] for r in rows} == set(P.SUITES["core"])
    for r in rows:
        assert r["approach"] == "fsa_lite_run3" and r["original_from"] == "fsa"
        assert Path(r["original_dir"]).parts[-4] == "fsa" and Path(r["technique_dir"]).parts[-4] == "fsa_lite_run3"
        assert "not paired" not in r["status"] and "missing" not in r["status"], r["status"]
