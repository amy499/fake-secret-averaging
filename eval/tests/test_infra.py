"""Static checks on the job scripts, env.sh and repo hygiene."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # eval/
REPO = ROOT.parent


def test_sbatch_template_constraints():
    t = (REPO / "scripts" / "slurm" / "harness_run.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:1" in t
    h, m, s = map(int, re.search(r"#SBATCH --time=(\d+):(\d+):(\d+)", t).groups())
    assert h * 3600 + m * 60 + s <= 6 * 3600
    assert "source env.sh" in t and "--device cuda" in t
    assert "SLURM_JOB_ID" in t  # refuses to run on the head node
    assert "source scripts/lib/cluster_env.sh" in t


def test_job_scripts_leave_site_settings_to_cluster_env():
    """No partition, QoS, project or queue in any job script: scripts/submit_{slurm,pbs}.sh pass them
    from cluster.env (git-ignored)."""
    jobs = sorted((REPO / "scripts").rglob("*.sbatch")) + sorted((REPO / "scripts").rglob("*.pbs"))
    assert len(jobs) >= 10
    for p in jobs:
        t = p.read_text(encoding="utf-8")
        assert not re.search(r"(?m)^#SBATCH --(partition|qos|account)=", t), p.name
        assert not re.search(r"(?m)^#PBS -[Pq] ", t), p.name
    for w in ("submit_slurm.sh", "submit_pbs.sh"):
        t = (REPO / "scripts" / w).read_text(encoding="utf-8")
        assert "scripts/lib/cluster_env.sh" in t
    ex = (REPO / "cluster.env.example").read_text(encoding="utf-8")
    for key in ("SLURM_PARTITION", "SLURM_QOS", "PBS_PROJECT", "PBS_QUEUE", "HF_HUB_CACHE", "SCRATCH", "CLUSTER_NAME"):
        assert f"\n{key}=" in ex, key


def test_env_sh_does_not_touch_startup_files():
    t = (ROOT / "env.sh").read_text(encoding="utf-8")
    assert ".bashrc" not in t.replace("Never add this to .bashrc", "")
    assert "HARNESS_DATA" in t and "PYTHONPATH" in t


def test_requirements_marks_extras():
    t = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "Extras" in t and "pyrit" in t
    assert not re.search(r"(?m)^pyrit", t)  # extra, not a core dependency


def test_no_ai_attribution_lines():
    pat = re.compile(r"co-authored-by|generated with claude|🤖|AI-generated \(Claude Code\)", re.I)
    for p in REPO.rglob("*"):
        if p.is_file() and not {".venv", ".git", ".cache", "__pycache__"} & set(p.parts) \
                and p.suffix in {".py", ".md", ".yaml", ".sh", ".sbatch", ".pbs", ".txt"} and p.name != "test_infra.py":
            assert not pat.search(p.read_text(encoding="utf-8", errors="ignore")), p
