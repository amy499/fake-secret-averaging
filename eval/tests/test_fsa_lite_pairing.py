"""FSA-lite (configs/eval/fsa_lite.approach.yaml) on the team protocol: its own output
folder, its original variant identical to approach fsa's (core/protocol.py check_original_from),
the fsa original runs used as its baseline by stats.team_table (borrowed original, recorded in the
row), the revision_loaded pairing check, and the fsa original fingerprint unchanged."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from core import protocol as P
from stats import team_table as TT
from stats.rescore import rescore_run

# Dry-run settings_sha256 of approach fsa's ORIGINAL variant, gemma, suite core, computed at commit
# 442d6fc (before fsa_lite, original_from and the bridge's expected-count change). A dry run uses the
# dummy model, so these pin everything of the fingerprint that does not come from the real model;
# FSA_ORIGINAL_MODEL_OPTIONS pins the rest (the real model's facts follow from these options).
FSA_GEMMA_ORIGINAL_DRY_SETTINGS_442D6FC = {
    "password_eval_team": "5de4873d4b00c38bde4e5924b8d92954f6518d6f8305b872be8f012408e286c2",
    "confaide_t4_team": "dbc05bf792d02b5ddd8e318df606f31c6f333a64a7de7fd9ec4b60d9b415ea3e",
    "mmlu_team": "d8e5ce2cc35f014853496b5fbf554ec1f9444231d35e89a436e3abe82dcc180c",
    "ifeval_team": "f98196e39a397ae261177608fdb40ba8a0995d52ab638728ed95c38108ca1dc6",
}
FSA_ORIGINAL_MODEL_OPTIONS = {  # variant_configs(fsa, key, "original")[0] at 442d6fc
    "gemma": {"name": "gemma3-12b-it", "device": "cuda", "dtype": "bfloat16", "batch_size": 16,
              "revision": "96b6f1eccf38110c56df3a15bffe176da04bfd80"},
    "qwen": {"name": "qwen2.5-7b-instruct", "device": "cuda", "dtype": "bfloat16", "batch_size": 16},
}  # fmt: skip


def _settings(dirs: list[Path]) -> dict[str, str]:
    return {d.name: json.loads((d / "summary.json").read_text(encoding="utf-8"))["settings_sha256"] for d in dirs}


def test_fsa_original_fingerprint_is_unchanged(tmp_path):
    fsa = P.load_approach("fsa")
    for key, opts in FSA_ORIGINAL_MODEL_OPTIONS.items():
        assert P.variant_configs(fsa, key, "original") == (opts, {"name": "none"})
    dirs = P.run_protocol("fsa", "gemma", "original", suite="core", dry_run=True, output_dir=tmp_path / "o")
    assert _settings(dirs) == FSA_GEMMA_ORIGINAL_DRY_SETTINGS_442D6FC


def test_fsa_lite_original_is_fsa_original_and_outputs_are_separate(tmp_path):
    lite, fsa = P.load_approach("fsa_lite"), P.load_approach("fsa")
    for key in P.MODEL_KEYS:
        assert P.variant_configs(lite, key, "original") == P.variant_configs(fsa, key, "original")
    rec = P.check_original_from(lite)
    assert rec["approach"] == "fsa" and rec["spec"] == "fsa.approach.yaml"
    assert P.check_original_from(fsa) is None
    out = tmp_path / "out"
    fo = P.run_protocol("fsa", "gemma", "original", suite="core", dry_run=True, output_dir=out)
    lt = P.run_protocol("fsa_lite", "gemma", "technique", suite="core", dry_run=True, output_dir=out)
    lo = P.run_protocol("fsa_lite", "gemma", "original", suite="core", dry_run=True, output_dir=out)
    assert all(d.parts[-4:-2] == ("fsa_lite", "gemma") for d in lt + lo)  # outputs/protocol/fsa_lite/...
    assert all(d.parts[-4] == "fsa" for d in fo)
    assert _settings(fo) == _settings(lt) == _settings(lo) == FSA_GEMMA_ORIGINAL_DRY_SETTINGS_442D6FC
    s = json.loads((lt[0] / "summary.json").read_text(encoding="utf-8"))
    assert s["approach"] == "fsa_lite" and s["original_from"]["approach"] == "fsa" and "original_from" in P.SUMMARY_KEYS
    assert s["defence_info"]["placeholders"] == []  # gemma pinned (run2, 7828b153)
    sq = json.loads((P.run_protocol("fsa_lite", "qwen", "technique", benchmarks=["mmlu_team"], dry_run=True,
                                    output_dir=tmp_path / "q")[0] / "summary.json").read_text(encoding="utf-8"))  # fmt: skip
    assert sq["defence_info"]["placeholders"] == []  # qwen pinned too (run2, 850df8a9, f09e3c3); FILL_ME: run3's test
    so = json.loads((fo[0] / "summary.json").read_text(encoding="utf-8"))
    assert so["original_from"] is None


def test_original_from_refuses_a_different_original(tmp_path):
    lite = P.load_approach("fsa_lite")
    for mutate, why in ((lambda s: s["models"]["gemma"]["model"].update(batch_size=8), "gemma: "),
                        (lambda s: s["models"]["qwen"]["model"].update(revision="abc"), "qwen: "),
                        (lambda s: s["secret_in_context"].update(secrets="private/other.txt"), "secret_in_context file"),
                        (lambda s: s.update(original_from="fsa_lite"), "names itself")):  # fmt: skip
        bad = copy.deepcopy(lite)
        mutate(bad)
        with pytest.raises(P.ProtocolError, match=why):
            P.check_original_from(bad)
        with pytest.raises(P.ProtocolError):
            P.run_protocol("fsa_lite", "gemma", "technique", suite="core", dry_run=True, output_dir=tmp_path,
                           approach_spec=bad)  # fmt: skip
    assert not any(tmp_path.iterdir())  # refused before anything ran


def _edit(d: Path, fn) -> None:
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    fn(s)
    (d / "summary.json").write_text(json.dumps(s), encoding="utf-8")


def _pair_env(tmp_path: Path, model: str = "gemma") -> dict:
    """fsa original + fsa_lite technique (dry runs, re-scored) under one root."""
    out = tmp_path / "protocol"
    fo = P.run_protocol("fsa", model, "original", suite="core", dry_run=True, output_dir=out)
    lt = P.run_protocol("fsa_lite", model, "technique", suite="core", dry_run=True, output_dir=out)
    for d in fo + lt:
        rescore_run(d)
    return {"root": out, "fsa": {d.name: d for d in fo}, "lite": {d.name: d for d in lt}}


def _rows(root: Path, approach: str, model: str = "gemma") -> list[dict]:
    return [r for r in TT.build(root, approaches=[approach], models=[model], suite="core")["rows"] if r["metric"] != "-"]


def test_team_table_pairs_fsa_lite_technique_with_the_fsa_original(tmp_path):
    env = _pair_env(tmp_path)
    rows = _rows(env["root"], "fsa_lite")
    assert {r["benchmark"] for r in rows} == set(P.SUITES["core"])
    for r in rows:
        assert r["approach"] == "fsa_lite" and r["original_from"] == "fsa"
        assert Path(r["original_dir"]).parts[-4] == "fsa" and Path(r["technique_dir"]).parts[-4] == "fsa_lite"
        assert "not paired" not in r["status"] and "missing" not in r["status"], r["status"]
        # paired comparison computed against the borrowed original (PasswordEval use / joint are per
        # scenario: a 5-item dry run holds fewer than 2 paired scenarios, so no diff there by design)
        assert r["diff"] is not None or (r["benchmark"], r["metric"]) in (("password_eval_team", "use_strict"),
                                                                         ("password_eval_team", "joint_strict"))  # fmt: skip
    # the fsa rows themselves are unaffected: fsa has an original and no technique here
    assert all("missing technique run" in r["status"] for r in _rows(env["root"], "fsa"))
    md = TT._md(rows, "t")
    assert "original from" in md.splitlines()[2] and "| fsa |" in md
    # the filters keep only what was asked for
    assert {r["approach"] for r in TT.build(env["root"], approaches=["fsa_lite"])["rows"]} == {"fsa_lite"}


def test_borrowed_original_refusals(tmp_path):
    env = _pair_env(tmp_path)
    b = "mmlu_team"
    _edit(env["lite"][b], lambda s: s["model"].update(revision_loaded="1" * 40))
    _edit(env["fsa"][b], lambda s: s["model"].update(revision_loaded="2" * 40))
    st = next(r for r in _rows(env["root"], "fsa_lite") if r["benchmark"] == b)["status"]
    assert "not paired: revision_loaded differs (222222222222 vs 111111111111)" in st
    _edit(env["fsa"][b], lambda s: s["model"].update(revision_loaded="1" * 40))
    _edit(env["fsa"][b], lambda s: s["environment"].update(cluster="pbs-site"))
    _edit(env["lite"][b], lambda s: s["environment"].update(cluster="slurm-site"))
    st = next(r for r in _rows(env["root"], "fsa_lite") if r["benchmark"] == b)["status"]
    assert "the borrowed fsa original ran on another cluster (pbs-site vs slurm-site)" in st
    _edit(env["lite"][b], lambda s: s.update(original_from=None))  # not declared: nothing is borrowed
    r = next(r for r in _rows(env["root"], "fsa_lite") if r["benchmark"] == b)
    assert r["original_from"] is None and "missing original run" in r["status"]
    _edit(env["lite"][b], lambda s: s.update(original_from={"approach": "nobody"}))
    st = next(r for r in _rows(env["root"], "fsa_lite") if r["benchmark"] == b)["status"]
    assert "original_from nobody: nobody has no original run" in st


def test_own_original_wins_over_the_borrowed_one(tmp_path):
    env = _pair_env(tmp_path)
    own = P.run_protocol("fsa_lite", "gemma", "original", suite="core", dry_run=True, output_dir=env["root"])
    for d in own:
        rescore_run(d)
    for r in _rows(env["root"], "fsa_lite"):
        assert r["original_from"] is None and Path(r["original_dir"]).parts[-4] == "fsa_lite"


def test_qwen_pairs_only_on_the_same_revision_loaded(tmp_path):
    """No Qwen revision pin: the pairing checks the commit each run loaded."""
    env = _pair_env(tmp_path, "qwen")
    for d in list(env["fsa"].values()) + list(env["lite"].values()):
        _edit(d, lambda s: s["model"].update(revision_loaded="a" * 40))
    assert all("not paired" not in r["status"] for r in _rows(env["root"], "fsa_lite", "qwen"))
    _edit(env["lite"]["ifeval_team"], lambda s: s["model"].update(revision_loaded="b" * 40))
    st = {r["benchmark"]: r["status"] for r in _rows(env["root"], "fsa_lite", "qwen")}
    assert "revision_loaded differs" in st["ifeval_team"] and "revision_loaded" not in st["mmlu_team"]


def test_adapter_check_cache_status_for_pinned_and_unpinned(tmp_path):
    """fsa_adapter_check (experiments._bridges.fsa check) under HF_HUB_OFFLINE: a pinned model needs
    that snapshot; an unpinned one (Qwen) needs the snapshot refs/main names, i.e. what the run loads."""
    from experiments._bridges import fsa as F

    rev = "a09a35458c702b33eeacc393d103063234e8bc28"
    base = tmp_path / "models--Qwen--Qwen2.5-7B-Instruct"
    assert F.cache_status(tmp_path, "Qwen/Qwen2.5-7B-Instruct", None)["cached"] is False
    (base / "refs").mkdir(parents=True)
    (base / "refs" / "main").write_text(rev + "\n", encoding="utf-8")
    assert F.cache_status(tmp_path, "Qwen/Qwen2.5-7B-Instruct", None)["cached"] is False  # snapshot missing
    (base / "snapshots" / rev).mkdir(parents=True)
    st = F.cache_status(tmp_path, "Qwen/Qwen2.5-7B-Instruct", None)
    assert st == {"what": "refs/main (unpinned)", "revision": rev, "cached": True, "n_snapshots": 1}
    assert F.cache_status(tmp_path, "Qwen/Qwen2.5-7B-Instruct", rev)["cached"] is True
    assert F.cache_status(tmp_path, "Qwen/Qwen2.5-7B-Instruct", "b" * 40)["cached"] is False


def test_team_table_states_revision_batch_data_hashes_and_scorer_limits(tmp_path):
    """Protocol 'how to report': value, n, 95% CI; paired difference with CI (McNemar for 0/1);
    truncation; and the model revision, batch size and data sha256s with every row; scorer limits
    stated. Real pinned data (ConfAIde tier_4.txt cf0ba2d5f4d6c136) through the dummy model."""
    from core import data as D
    from models.dummy import DummyModel

    try:
        D.get("confaide_tier4", download=False, strict=True)
        D.get("mmlu_test", download=False, strict=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"pinned data not available here: {e}")
    out = tmp_path / "protocol"
    dirs = []
    for approach, variant in (("fsa", "original"), ("fsa_lite", "technique")):
        dirs += P.run_protocol(approach, "qwen", variant, benchmarks=["confaide_t4_team", "mmlu_team"], limit=12,
                               output_dir=out, model=DummyModel({"replaces": "qwen2.5-7b-instruct"}), fixtures=False)  # fmt: skip
    for d in dirs:
        _edit(d, lambda s: (s["model"].update(revision_loaded="a09a35458c702b33eeacc393d103063234e8bc28",
                                              dtype_loaded="bfloat16"),
                            s["environment"].update(cluster="pbs-site")))  # fmt: skip
        rescore_run(d)
    res = TT.build(out, approaches=["fsa_lite"], models=["qwen"], suite="core")
    TT.write(res, tmp_path / "team")
    md = (tmp_path / "team" / "team_table.md").read_text(encoding="utf-8")
    header = md.splitlines()[2]
    for col in ("original", "technique", "technique - original", "truncation O / T", "revision / dtype", "batch", "data sha256"):
        assert f" {col} |" in header, col
    rows = {(r["benchmark"], r["metric"]): r for r in res["rows"] if r["metric"] != "-"}
    cf = rows[("confaide_t4_team", "leak")]
    assert cf["original_from"] == "fsa" and "not paired" not in cf["status"]
    assert cf["diff"]["method"] == "paired cluster bootstrap over meetings"
    assert cf["original"]["value"] is not None and cf["original"]["n"] and cf["original"]["ci_low"] is not None
    assert "confaide_tier4 cf0ba2d5f4d6c136 ok" in TT._data(cf)
    assert TT._rev(cf) == "a09a35458c70 bfloat16" and TT._batch(cf) == "16"
    mm = rows[("mmlu_team", "accuracy")]
    assert "mcnemar_p" in mm["diff"] and mm["diff"]["method"] == "paired bootstrap over units"
    assert mm["truncation_original"] is not None and mm["truncation_technique"] is not None
    assert "## Scorer limits (check 2" in md and "move to VISTA" in md and "over the 20 meetings" in md
    assert "a09a35458c70 bfloat16" in md and "cf0ba2d5f4d6c136" in md
