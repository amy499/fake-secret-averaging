"""core/cluster.py and its use: cluster detection, the pin file, summary.environment, the team
table's refusal to pair runs on different GPUs / library versions, and the job scripts' guards."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from core import cluster as C
from core import protocol as P
from stats import team_table as TT
from stats.rescore import rescore_run

ROOT = Path(__file__).resolve().parent.parent


def test_cluster_detection():
    # the name comes from HARNESS_CLUSTER (cluster.env CLUSTER_NAME, set by scripts/lib/cluster_env.sh)
    site = C.cluster_info({"SLURM_JOB_ID": "54077", "SLURM_JOB_PARTITION": "gpu", "HARNESS_CLUSTER": "slurm-site"})
    assert site["cluster"] == "slurm-site" and site["scheduler"] == "slurm" and site["job_id"] == "54077"
    pbs = C.cluster_info({"PBS_JOBID": "26245786.pbs101", "PBS_O_HOST": "login01"})
    assert pbs["cluster"] == "pbs" and pbs["scheduler"] == "pbs" and pbs["job_id"] == "26245786.pbs101"  # no name set
    assert C.cluster_info({"PBS_JOBID": "1", "HARNESS_CLUSTER": "pbs-site"})["cluster"] == "pbs-site"
    assert C.cluster_info({"SLURM_JOB_ID": "1"})["cluster"] == "slurm"
    local = C.cluster_info({})
    assert local["cluster"] == "local" and local["scheduler"] is None and local["hostname"]


def test_pin_file_and_comparison():
    pins = C.read_pins()
    assert pins["torch"] == "2.6.0" and pins["transformers"] == "4.51.3" and pins["tokenizers"] == "0.21.1"
    assert pins["huggingface-hub"] == "0.30.2" and pins["nltk"] == "3.9.1" and pins["absl-py"] == "2.1.0"
    # every library the pairing rule compares is pinned
    assert {C._norm(x) for x in C.LIBRARIES} <= set(pins)
    ok = {"torch": "2.6.0+cu124", "transformers": "4.51.3"}
    assert C.compare_pins(ok, {"torch": "2.6.0", "transformers": "4.51.3"}) == []  # +cu124 is not part of the pin
    diffs = C.compare_pins({"torch": "2.1.0+cu121", "transformers": None}, {"torch": "2.6.0", "transformers": "4.51.3"})
    assert diffs == ["torch 2.1.0+cu121 (pinned 2.6.0)", "transformers None (pinned 4.51.3)"]
    # requirements.txt keeps the same IFEval pins
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    for name in ("nltk", "langdetect", "absl-py", "immutabledict"):
        assert f"{name}=={pins[name]}" in req


def _env(gpu="NVIDIA A40", cuda="12.4", cluster="slurm-site", **libs):
    base = {"torch": "2.6.0+cu124", "transformers": "4.51.3", "tokenizers": "0.21.1"}
    return {"cluster": cluster, "hostname": "node1", "gpu": {"name": gpu, "torch_cuda": cuda},
            "libraries": {**base, **libs}, "pins": {"match": True, "differences": []}}  # fmt: skip


def test_pairing_rule():
    assert C.pairing_differences(_env(), _env(cluster="pbs-site")) == []  # same stack, other cluster: fine
    assert C.pairing_differences(_env(), _env(gpu="NVIDIA A100-SXM4-40GB")) == ["GPU NVIDIA A40 vs NVIDIA A100-SXM4-40GB"]
    assert C.pairing_differences(_env(), _env(cuda="12.1")) == ["torch CUDA build 12.4 vs 12.1"]
    assert C.pairing_differences(_env(), _env(transformers="4.57.1")) == ["transformers 4.51.3 vs 4.57.1"]
    assert C.pairing_differences(None, _env()) == ["environment not recorded"]
    assert C.where(_env()) == "slurm-site / NVIDIA A40" and C.where(None) == "?"


def test_summary_records_the_environment(tmp_path):
    from tests.test_protocol import _template_spec

    spec = _template_spec(tmp_path, "envrec")
    out = P.run_protocol(spec["name"], "gemma", "original", benchmarks=["gsm8k_team"], dry_run=True,
                         output_dir=tmp_path / "out", approach_spec=spec)[0]  # fmt: skip
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    env = s["environment"]
    assert s["schema_version"] == 2 and "environment" in P.SUMMARY_KEYS
    assert {"cluster", "hostname", "python", "gpu", "libraries", "pins"} <= set(env)
    assert {"name", "torch_cuda", "driver"} <= set(env["gpu"]) and "transformers" in env["libraries"]
    assert not any("eval/requirements-gpu.txt" in r for r in s["non_compliance"])  # dry run: pins not required


def test_real_run_off_the_pins_is_not_compliant(tmp_path, monkeypatch):
    from models.dummy import DummyModel
    from tests.test_protocol import _template_spec

    bad = _env(gpu=None)
    bad["pins"] = {"match": False, "differences": ["transformers 5.20.1 (pinned 4.51.3)"]}
    monkeypatch.setattr(C, "environment", lambda: copy.deepcopy(bad))
    spec = _template_spec(tmp_path, "offpins")
    out = P.run_protocol(spec["name"], "gemma", "original", benchmarks=["gsm8k_team"], fixtures=True,
                         model=DummyModel(), output_dir=tmp_path / "out", approach_spec=spec)[0]  # fmt: skip
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert any("transformers 5.20.1 (pinned 4.51.3)" in r for r in s["non_compliance"])


@pytest.fixture()
def pair(tmp_path):
    from tests.test_protocol import _template_spec

    spec = _template_spec(tmp_path, "envpair")
    dirs = {}
    for v in P.VARIANTS:
        d = P.run_protocol(spec["name"], "gemma", v, benchmarks=["mmlu_team"], dry_run=True,
                           output_dir=tmp_path / "out", approach_spec=spec)[0]  # fmt: skip
        rescore_run(d)
        dirs[v] = d
    return tmp_path / "out", dirs


def _set_env(d: Path, env: dict) -> None:
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    s["environment"] = env
    (d / "summary.json").write_text(json.dumps(s), encoding="utf-8")


@pytest.mark.parametrize("technique_env, reason", [
    (_env(gpu="NVIDIA A100-SXM4-40GB", cluster="pbs-site"), "GPU NVIDIA A40 vs NVIDIA A100-SXM4-40GB"),
    (_env(torch="2.1.0+cu121"), "torch 2.6.0+cu124 vs 2.1.0+cu121"),
    (_env(transformers="4.57.1"), "transformers 4.51.3 vs 4.57.1"),
])  # fmt: skip
def test_team_table_refuses_pairs_on_different_stacks(pair, technique_env, reason):
    out, dirs = pair
    _set_env(dirs["original"], _env())
    _set_env(dirs["technique"], technique_env)
    row = next(r for r in TT.build(out)["rows"] if r["benchmark"] == "mmlu_team")
    assert "not paired: environment differs" in row["status"] and reason in row["status"]
    assert row["diff"] is None and row["identical_reply_fraction"] is None


def test_team_table_pairs_the_same_stack_on_two_clusters(pair):
    out, dirs = pair
    _set_env(dirs["original"], _env(cluster="slurm-site"))
    _set_env(dirs["technique"], _env(cluster="pbs-site"))
    row = next(r for r in TT.build(out)["rows"] if r["benchmark"] == "mmlu_team")
    assert "environment differs" not in row["status"] and row["diff"] is not None




def test_run_py_accepts_plus_between_benchmarks(tmp_path, capsys):
    import run

    assert run.main(["--approach", "fsa", "--model", "qwen", "--variant", "original", "--benchmarks",
                     "mmlu_team+gsm8k_team", "--dry-run", "--output-dir", str(tmp_path)]) == 0  # fmt: skip
    assert [Path(p).name for p in capsys.readouterr().out.strip().splitlines()] == ["mmlu_team", "gsm8k_team"]


BASH = shutil.which("bash")


@pytest.mark.skipif(BASH is None, reason="bash not on PATH")
@pytest.mark.parametrize("script, unset", [("scripts/pbs/harness_run.pbs", "PBS_JOBID"), ("scripts/slurm/harness_run.sbatch", "SLURM_JOB_ID")])
def test_job_scripts_refuse_outside_a_job_and_check_pins(script, unset):
    import os

    for s in ("scripts/pbs/harness_run.pbs", "scripts/slurm/harness_run.sbatch", "scripts/lib/setup_env.sh"):
        assert subprocess.run([BASH, "-n", s], cwd=ROOT.parent).returncode == 0, s
        assert b"\r\n" not in (ROOT.parent / s).read_bytes(), f"{s} must have LF line ends"
    text = (ROOT.parent / script).read_text(encoding="utf-8")
    assert "python -m core.cluster check" in text and "ALLOW_UNPINNED" in text and "setup_env.sh" in text
    env = {k: v for k, v in os.environ.items() if k not in ("PBS_JOBID", "SLURM_JOB_ID")}
    r = subprocess.run([BASH, script, "--approach", "x"], cwd=ROOT.parent, env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "refusing to run outside" in r.stderr
