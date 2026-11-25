"""--dry-run end to end for every config (hence every benchmark), plus output-file checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from core.config import ExperimentConfig, reply_seed
from core.io import read_jsonl
from core.runner import run

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = sorted((ROOT.parent / "configs" / "attack").glob("*.yaml"))
OUTPUT_FILES = ["config.yaml", "replies.jsonl", "guesses.jsonl", "metrics.json", "per_item.json", "git_commit.txt"]


def _run(path: Path, out: Path) -> Path:
    cfg = ExperimentConfig.load(path)
    cfg.output_dir = str(out)
    return run(cfg, dry_run=True)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_dry_run_end_to_end(path, tmp_path):
    out = _run(path, tmp_path)
    assert out.name.endswith("_" + path.stem)
    for f in OUTPUT_FILES:
        assert (out / f).exists(), f
    saved = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
    assert saved["dry_run"] is True and saved["model"]["name"] == "dummy"
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    cfg = ExperimentConfig.load(path)
    assert set(metrics["metrics"]) == {m["name"] for m in cfg.metrics}
    assert metrics["n_items"] > 0
    replies = read_jsonl(out / "replies.jsonl")
    k = cfg.decoding.k
    n_items = metrics["n_items"]
    main = [r for r in replies if not r["control"]]
    assert len(main) == n_items * (1 + k)
    # every sampled reply carries the documented seed
    sampled = [r for r in main if r["sample"] >= 0]
    assert all(r["seed"] is not None and r["seed"] % 100 == r["sample"] for r in sampled)
    assert all(r["seed"] is None for r in main if r["sample"] < 0)
    guesses = read_jsonl(out / "guesses.jsonl")
    labels = {a.get("label", a["name"]) for a in cfg.attackers}
    assert {g["attacker"] for g in guesses} == labels
    git = (out / "git_commit.txt").read_text(encoding="utf-8").splitlines()[0]
    assert len(git) == 40 or git == "None"


def test_dry_run_is_deterministic(tmp_path):
    path = ROOT.parent / "configs" / "attack" / "code_check__none.yaml"
    a = _run(path, tmp_path / "a")
    b = _run(path, tmp_path / "b")
    strip = lambda rows: [{k: v for k, v in r.items() if k != "latency_s"} for r in rows]  # noqa: E731
    assert strip(read_jsonl(a / "replies.jsonl")) == strip(read_jsonl(b / "replies.jsonl"))
    ga = [{k: v for k, v in g.items() if k != "extra"} for g in read_jsonl(a / "guesses.jsonl")]
    gb = [{k: v for k, v in g.items() if k != "extra"} for g in read_jsonl(b / "guesses.jsonl")]
    assert ga == gb


def test_utility_delta_pairs_with_baseline(tmp_path):
    _run(ROOT.parent / "configs" / "attack" / "mmlu__none.yaml", tmp_path)
    out = _run(ROOT.parent / "configs" / "attack" / "mmlu__fsa.yaml", tmp_path)
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    delta = m["metrics"]["utility_delta"]["delta_vs_none"]
    assert delta["diff"] is not None and delta["n"] > 1 and "p" in delta


def test_reply_seed_formula():
    assert reply_seed(0, 0) == 2026_000_000
    assert reply_seed(903, 4) == 2026_000_000 + 90300 + 4
    with pytest.raises(ValueError):
        reply_seed(1, 100)


def test_cli_dry_run(tmp_path):
    import subprocess
    import sys

    res = subprocess.run([sys.executable, str(ROOT / "run.py"), "--config", str(ROOT.parent / "configs" / "attack" / "password_eval__none.yaml"),
                          "--dry-run", "--output-dir", str(tmp_path)], capture_output=True, text=True, cwd=ROOT)  # fmt: skip
    assert res.returncode == 0, res.stderr
    assert Path(res.stdout.strip()).exists()
