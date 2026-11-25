"""scripts/export_results.py on tmp fixtures: only allowlisted files are copied, home-directory
paths in config.yaml are made repo-relative (and recorded without the old value), JSON with a
home path or an invalid/missing run aborts before anything is written, re-export replaces run
folders, --prune removes unlisted ones, the summary and README cover exactly the exported runs,
and the source folders are byte-for-byte unchanged."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import yaml
from test_collect_results import _attack_file
from _run_helpers import _baseline_run, _edit_json, _utility_run

from scripts import export_results as ex

ROOT = Path(__file__).resolve().parents[1]
HOME_ADAPTER = (
    "/home/lab/someone/secret-isolating-attention/experiments/fake_secret_average/"
    "outputs/training/20261024-000000_fsa/adapter"
)
NOW = datetime(2026, 11, 9, 12, 0, 0)


def _snapshot(folder):
    return {p.relative_to(folder): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


def _write_list(path, runs, utility=()):
    data = {
        "runs": [{"id": r, "label": f"label {r[-6:]}"} for r in runs],
        "utility": [{"id": u, "label": f"utility {u[-4:]}"} for u in utility],
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture
def src(tmp_path):
    """outputs-like tree: two evaluation runs (one with a home path in config.yaml, attack files,
    and every file that must not be exported) and one utility run with items.jsonl."""
    runs = tmp_path / "outputs" / "runs"
    a = _baseline_run(runs, name="exp_a", seed=1)
    b = _baseline_run(runs, name="exp_b", prompt_id="lottery", seed=2)
    rng = np.random.default_rng(0)
    _attack_file(
        a, a / "likelihood_attack.json", rng.normal(size=(4, 100)) * 3, tau=2.0, threat="grey_box"
    )
    _attack_file(
        a,
        a / "likelihood_attack_public_weights.json",
        rng.normal(size=(4, 100)) * 3,
        threat="public_weights",
    )
    cfg = yaml.safe_load((a / "config.yaml").read_text(encoding="utf-8"))
    cfg.setdefault("generation", {})["adapter"] = HOME_ADAPTER
    cfg["generation"]["hf_home"] = "/home/lab/someone/.cache/huggingface"
    cfg["notes"] = ["trained at C:\\Users\\someone\\Desktop\\x", "plain text"]
    (a / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    for name in ("reanalysis.json", "fsa-likelihood_1.out"):
        (a / name).write_text("{}", encoding="utf-8")
    (a / "adapter").mkdir()
    (a / "adapter" / "adapter_model.safetensors").write_bytes(b"\0" * 16)
    assert (a / "samples.jsonl").exists() and (a / "train_samples.jsonl").exists()
    util = tmp_path / "outputs" / "utility"
    u = _utility_run(util, "20261109-100000_gemma3-12b-it_base")
    (u / "items.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
    lst = _write_list(tmp_path / "export_runs.yaml", [a.name, b.name], [u.name])
    return {"runs": runs, "utility": util, "a": a, "b": b, "u": u, "list": lst, "tmp": tmp_path}


def _export(src, out, **kw):
    return ex.export(
        src["list"], src["runs"], src["utility"], out, log=lambda m: None, now=NOW, **kw
    )


def test_copies_only_allowlisted_files_and_leaves_sources_untouched(src):
    before = _snapshot(src["tmp"] / "outputs")
    out = src["tmp"] / "results"
    _export(src, out)
    assert _snapshot(src["tmp"] / "outputs") == before  # read-only
    a, b, u = src["a"].name, src["b"].name, src["u"].name
    assert sorted(p.name for p in (out / "runs" / a).iterdir()) == [
        "config.yaml",
        "diagnostics.json",
        "likelihood_attack.json",
        "likelihood_attack_public_weights.json",
        "metrics.json",
    ]
    assert sorted(p.name for p in (out / "runs" / b).iterdir()) == [
        "config.yaml",
        "diagnostics.json",
        "metrics.json",
    ]
    assert sorted(p.name for p in (out / "utility" / u).iterdir()) == [
        "config.yaml",
        "utility_metrics.json",
    ]
    exported = {p.name for p in out.rglob("*") if p.is_file()}
    for never in (
        "samples.jsonl",
        "train_samples.jsonl",
        "items.jsonl",
        "reanalysis.json",
        "fsa-likelihood_1.out",
        "adapter_model.safetensors",
    ):
        assert never not in exported
    # JSON results and configs without home paths are byte-identical copies.
    for name in ("metrics.json", "likelihood_attack.json"):
        assert (out / "runs" / a / name).read_bytes() == (src["a"] / name).read_bytes()
    assert (out / "runs" / b / "config.yaml").read_bytes() == (
        src["b"] / "config.yaml"
    ).read_bytes()
    assert sorted(p.name for p in out.iterdir()) == [
        "README.md",
        "export_manifest.json",
        "runs",
        "summary.csv",
        "summary.md",
        "utility",
    ]


def test_home_paths_relativised_and_recorded(src):
    out = src["tmp"] / "results"
    _export(src, out)
    a = src["a"].name
    text = (out / "runs" / a / "config.yaml").read_text(encoding="utf-8")
    assert "/home/" not in text and "Users" not in text and "someone" not in text
    cfg = yaml.safe_load(text)
    assert cfg["generation"]["adapter"] == "outputs/training/20261024-000000_fsa/adapter"
    assert cfg["generation"]["hf_home"] == "<home>/.cache/huggingface"
    assert cfg["notes"] == ["trained at <home>/Desktop/x", "plain text"]
    m = json.loads((out / "export_manifest.json").read_text(encoding="utf-8"))
    assert "someone" not in json.dumps(m)  # the old values are not recorded
    run = next(r for r in m["runs"] if r["run_id"] == a)
    files = {f["name"]: f for f in run["files"]}
    assert {c["key"] for c in files["config.yaml"]["home_paths_relativised"]} == {
        "generation.adapter",
        "generation.hf_home",
        "notes[0]",
    }
    assert files["config.yaml"]["source_sha256"] != files["config.yaml"]["sha256"]
    assert files["metrics.json"]["source_sha256"] == files["metrics.json"]["sha256"]
    assert files["metrics.json"]["home_paths_relativised"] == []
    assert m["exported_at"] == "2026-11-09T12:00:00" and run["kind"] == "runs"



@pytest.fixture(autouse=True)
def _two_level_home(monkeypatch):
    """The fixtures use a two-level cluster home (/home/lab/<user>), configured as for a real site."""
    monkeypatch.setattr(ex, "HOME_GROUPS", ("lab",))


def test_relativise():
    assert ex.relativise(HOME_ADAPTER) == "outputs/training/20261024-000000_fsa/adapter"
    assert (
        ex.relativise("/home/lab/u/secret-isolating-attention/experiments/fake_secret_average")
        == "."
    )
    assert ex.relativise("/scratch/u/hf") == "<home>/hf"
    assert (
        ex.relativise("C:\\Users\\u\\proj\\experiments\\fake_secret_average\\configs\\a.yaml")
        == "configs/a.yaml"
    )
    assert ex.relativise("/home/u") == "<home>/"
    assert ex.relativise("/home/lab/u/.cache/huggingface") == "<home>/.cache/huggingface"
    assert ex.relativise("/home/alice/.cache") == "<home>/.cache"


def test_json_with_home_path_aborts_and_writes_nothing(src):
    _edit_json(src["a"] / "likelihood_attack.json", lambda d: d["notes"].append("/home/lab/u/x"))
    out = src["tmp"] / "results"
    with pytest.raises(ex.ExportError, match="absolute home-directory path"):
        _export(src, out)
    assert not out.exists()


def test_missing_or_invalid_run_aborts_and_writes_nothing(src):
    out = src["tmp"] / "results"
    _write_list(src["list"], [src["a"].name, "20261109-000000_missing"], [src["u"].name])
    (src["b"] / "metrics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ex.ExportError) as e:
        _export(src, out)
    assert "20261109-000000_missing: config.yaml missing" in str(e.value)
    assert not out.exists()
    _write_list(src["list"], [src["a"].name, src["b"].name])
    with pytest.raises(ex.ExportError, match=src["b"].name):
        _export(src, out)
    assert not out.exists()


def test_reexport_replaces_and_prune_removes(src):
    out = src["tmp"] / "results"
    _export(src, out)
    a, b = src["a"].name, src["b"].name
    (out / "runs" / a / "stray.txt").write_text("x", encoding="utf-8")
    _write_list(src["list"], [a])  # b and the utility run no longer listed
    _export(src, out)
    assert not (out / "runs" / a / "stray.txt").exists()  # run folder replaced
    assert (out / "runs" / b).exists()  # kept without --prune ...
    summary = (out / "summary.csv").read_text(encoding="utf-8")
    assert b in summary  # ... and still summarised, since it is still in results/runs
    _export(src, out, prune=True)
    assert sorted(p.name for p in (out / "runs").iterdir()) == [a]
    assert list((out / "utility").iterdir()) == []


def test_dry_run_writes_nothing(src):
    out = src["tmp"] / "results"
    planned = _export(src, out, dry_run=True)
    assert [p.run_id for p in planned] == [src["a"].name, src["b"].name, src["u"].name]
    assert not out.exists()


def test_summary_and_readme_cover_exactly_the_exported_runs(src):
    out = src["tmp"] / "results"
    _export(src, out)
    a, b, u = src["a"].name, src["b"].name, src["u"].name
    summary = (out / "summary.csv").read_text(encoding="utf-8").splitlines()
    run_ids = {line.split(",")[0] for line in summary[1:]}
    assert run_ids == {a, b}
    readme = (out / "README.md").read_text(encoding="utf-8")
    for rid in (a, b, u):
        assert f"`{rid}`" in readme
    for credit in ("UnpredictaBench", "CC-BY-4.0", "cais/mmlu", "google/IFEval"):
        assert credit in readme
    row_a = next(line for line in readme.splitlines() if f"`{a}`" in line)
    assert "grey-box, public weights" in row_a and " cal |" in row_a
    row_b = next(line for line in readme.splitlines() if f"`{b}`" in line)
    assert "| lottery |" in row_b and row_b.endswith("— | — |")  # no attack files
    assert "samples.jsonl" in readme  # the exclusions are explained
    assert "62.0% [57.0, 67.0] / 62.0% [57.0, 67.0]" in readme  # utility MMLU, both conditions


def test_out_dir_overlapping_sources_is_refused(src):
    with pytest.raises(ex.ExportError, match="overlaps"):
        _export(src, src["runs"] / "results")
    with pytest.raises(ex.ExportError, match="overlaps"):
        _export(src, src["tmp"] / "outputs")


def test_real_list_file_is_valid_and_results_folder_is_not_ignored():
    lst = ex.load_list(ROOT / "configs" / "training" / "export_runs.yaml")
    v1 = ROOT / "results" / "fsa_v1"
    # Published: aggregates only. The per-run folders (configs with secret lists, per-code scores,
    # diagnostics) are kept privately, so every listed run must appear in the summary / README tables.
    assert not (v1 / "runs").exists() and not (v1 / "utility").exists()
    summary_ids = {line.split(",")[0] for line in (v1 / "summary.csv").read_text(encoding="utf-8").splitlines()[1:]}
    readme = (v1 / "README.md").read_text(encoding="utf-8")
    assert lst["runs"] and {e["id"] for e in lst["runs"]} <= summary_ids
    assert lst["utility"] and all(f"`{e['id']}`" in readme for e in lst["utility"])
    manifest = json.loads((v1 / "export_manifest.json").read_text(encoding="utf-8"))
    assert {f["name"] for f in manifest["files"]} == {p.name for p in v1.iterdir()} - {"export_manifest.json"}
    assert any("B3 baseline" in e["label"] for e in lst["runs"])
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "results/" not in lines and "outputs/" in lines


def test_readme_shows_raw_dagger_when_tau_hit_its_bound(src):
    def at_bound(d):
        d["calibration"].update(tau=10000.0, tau_at_search_bound=True)

    _edit_json(src["a"] / "likelihood_attack.json", at_bound)
    raw_bits = json.loads((src["a"] / "likelihood_attack.json").read_text(encoding="utf-8"))[
        "leakage"
    ]["bits_leaked_lower_bound"]
    out = src["tmp"] / "results"
    _export(src, out)
    readme = (out / "README.md").read_text(encoding="utf-8")
    row_a = next(line for line in readme.splitlines() if f"`{src['a'].name}`" in line)
    assert f"{raw_bits:.3f}" in row_a and "*raw†*" in row_a
    assert ex.DAGGER_NOTE in readme
    # Without any at-bound file there is no footnote under the table.
    _edit_json(
        src["a"] / "likelihood_attack.json",
        lambda d: d["calibration"].update(tau_at_search_bound=False),
    )
    _export(src, out)
    assert ex.DAGGER_NOTE not in (out / "README.md").read_text(encoding="utf-8")
