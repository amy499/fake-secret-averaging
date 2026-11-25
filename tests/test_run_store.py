import json
from datetime import datetime

import pytest
import yaml

from fsa_runtime.runs_store import FolderRunStore, RunNotFound, RunStore
from fsa.io.run_writer import RunWriter, make_run_id, sanitize_name

T1 = datetime(2026, 10, 17, 9, 0, 0)
T2 = datetime(2026, 10, 17, 10, 30, 0)


def _write_run(runs_dir, factories, now, n_samples=5, **config_kw):
    run = RunWriter(runs_dir, factories.config(**config_kw), now=now)
    run.append_samples(factories.sample(i=i) for i in range(n_samples))
    run.write_metrics(factories.metrics())
    return run


def test_run_id_format():
    assert make_run_id("gemma3 4b/fsa k=8", T1) == "20261017-090000_gemma3-4b-fsa-k-8"
    assert sanitize_name("qwen2.5") == "qwen2-5"
    with pytest.raises(ValueError):
        sanitize_name("///")


def test_writer_creates_folder_with_lf_files(tmp_path, factories):
    run = _write_run(tmp_path, factories, T1)
    files = sorted(p.name for p in run.path.iterdir())
    assert files == ["config.yaml", "metrics.json", "samples.jsonl"]
    for p in run.path.iterdir():
        assert b"\r\n" not in p.read_bytes(), p.name
    metrics = json.loads((run.path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["run_id"] == run.run_id and metrics["is_mock"] is True
    config = yaml.safe_load((run.path / "config.yaml").read_text(encoding="utf-8"))
    assert list(config)[:2] == ["name", "seed"]  # caller's key order kept


def test_writer_rejects_invalid_and_never_overwrites(tmp_path, factories):
    run = RunWriter(tmp_path, factories.config(), now=T1)
    with pytest.raises(ValueError):
        run.write_metrics({"leakage": factories.leakage()})  # no is_mock, no utility
    with pytest.raises(ValueError):
        run.append_samples([{"secret": "37"}])
    with pytest.raises(FileExistsError):
        RunWriter(tmp_path, factories.config(), now=T1)
    assert not (run.path / "metrics.json").exists()


def test_writer_checks_patching_matches_method(tmp_path, factories):
    run = RunWriter(tmp_path, factories.config(method="patching", k=None, layers=None), now=T1)
    with pytest.raises(ValueError, match="patching"):
        run.write_metrics(factories.metrics())
    run.write_metrics({"is_mock": True, "patching": factories.patching()})


def test_store_satisfies_protocol(tmp_path):
    store: RunStore = FolderRunStore(tmp_path)
    assert store.list_runs() == []
    assert FolderRunStore(tmp_path / "missing").list_runs() == []


def test_list_runs_newest_first_and_flattened(tmp_path, factories):
    _write_run(tmp_path, factories, T1, name="older", method="base", k=None, layers=None)
    newer = _write_run(tmp_path, factories, T2, name="newer")
    runs = FolderRunStore(tmp_path).list_runs()
    assert [r.name for r in runs] == ["newer", "older"]
    top = runs[0]
    assert top.run_id == newer.run_id and top.created_at == T2
    assert top.is_mock and top.method == "fsa" and top.k == 8 and top.layers == [14, 15, 16, 17]
    assert top.attacker_exact_match.chance == 0.01 and top.attacker_per_digit.chance == 0.1
    assert top.n_replies_per_secret == 20 and top.kl_upper_bound_bits is None
    assert top.eval_split == "heldout" and top.attacker_trained_fresh is True
    assert top.kl_to_base_nonsecret == 0.031 and top.task_score == 0.92
    assert runs[1].k is None and runs[1].layers is None


def test_list_runs_skips_incomplete_invalid_and_stray(tmp_path, factories, caplog):
    _write_run(tmp_path, factories, T1, name="good")
    RunWriter(tmp_path, factories.config(name="running"), now=T2)  # no metrics.json yet
    broken = _write_run(tmp_path, factories, T2, name="broken")
    (broken.path / "metrics.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "notes").mkdir()
    (tmp_path / "README.txt").write_text("x", encoding="utf-8")
    runs = FolderRunStore(tmp_path).list_runs()
    assert [r.name for r in runs] == ["good"]
    assert "broken" in caplog.text


def test_get_run(tmp_path, factories):
    run = _write_run(tmp_path, factories, T1, n_samples=7)
    detail = FolderRunStore(tmp_path).get_run(run.run_id)
    assert detail.n_samples == 7 and detail.is_mock
    assert detail.config["eval"]["n_replies_per_secret"] == 20  # raw config incl. extra keys
    assert detail.metrics.leakage.attacker.name == "qwen2.5-1.5b-lora"


def test_get_samples_paging(tmp_path, factories):
    run = _write_run(tmp_path, factories, T1, n_samples=7)
    store = FolderRunStore(tmp_path)
    page = store.get_samples(run.run_id, offset=5, limit=10)
    assert page.total == 7 and [s.reply_index for s in page.items] == [5, 6]
    assert store.get_samples(run.run_id, offset=50).items == []
    with pytest.raises(ValueError):
        store.get_samples(run.run_id, limit=0)


def test_samples_append_across_calls(tmp_path, factories):
    run = RunWriter(tmp_path, factories.config(), now=T1)
    run.append_samples([factories.sample(i=0)])
    run.append_samples([factories.sample(i=1), factories.sample(i=2)])
    run.write_metrics(factories.metrics())
    assert FolderRunStore(tmp_path).get_samples(run.run_id).total == 3


@pytest.mark.parametrize(
    "run_id",
    ["nope", "../secret", "20261017-090000_../../x", "20261017-090000_missing"],
)
def test_unknown_or_unsafe_run_ids(tmp_path, factories, run_id):
    _write_run(tmp_path, factories, T1)
    store = FolderRunStore(tmp_path)
    with pytest.raises(RunNotFound):
        store.get_run(run_id)
    with pytest.raises(RunNotFound):
        store.get_samples(run_id)


def test_incomplete_run_not_found(tmp_path, factories):
    run = RunWriter(tmp_path, factories.config(), now=T1)
    with pytest.raises(RunNotFound):
        FolderRunStore(tmp_path).get_run(run.run_id)
