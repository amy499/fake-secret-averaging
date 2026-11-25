"""Cluster support: the commit and .gitignore read from the files (some compute nodes have no git
binary), no git calls on compute paths, the PBS job scripts, uv installed by the setup job when
no Python 3.11 exists (UV_AUTO_INSTALL=1), and the FSA adapter chosen from paths.local.yaml, a
per-cluster map or FSA_ADAPTER_DIR, with an optional sha256 pin."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from core.io import REPO, ROOT, git_commit, gitignored

BASH = shutil.which("bash")
GIT = shutil.which("git")
SHA_A, SHA_B = "a" * 40, "b" * 40


def _bash_head_commit(cwd: Path) -> str:
    r = subprocess.run([BASH, "-c", f"source '{(REPO / 'scripts' / 'lib' / 'job_lib.sh').as_posix()}'; head_commit"],
                       cwd=cwd, capture_output=True, text=True, timeout=30)  # fmt: skip
    return r.stdout.strip()


def _fake_repo(root: Path, layout: str) -> Path:
    """A checkout whose .git has the given layout; HEAD is on branch main = SHA_A (detached: SHA_B)."""
    root.mkdir(parents=True)
    if layout == "worktree":
        common = root.parent / f"{root.name}_main.git"
        gitdir = common / "worktrees" / "wt"
        gitdir.mkdir(parents=True)
        (gitdir / "commondir").write_text("../..\n", encoding="utf-8")
        (gitdir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (common / "refs" / "heads").mkdir(parents=True)
        (common / "refs" / "heads" / "main").write_text(SHA_A + "\n", encoding="utf-8")
        (root / ".git").write_text(f"gitdir: {gitdir.as_posix()}\n", encoding="utf-8")
        return root
    g = root / ".git"
    g.mkdir()
    if layout == "detached":
        (g / "HEAD").write_text(SHA_B + "\n", encoding="utf-8")
        return root
    (g / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    if layout == "loose":
        (g / "refs" / "heads").mkdir(parents=True)
        (g / "refs" / "heads" / "main").write_text(SHA_A + "\n", encoding="utf-8")
    elif layout == "packed":  # a fresh clone (and a git bundle clone) keeps refs in packed-refs
        (g / "packed-refs").write_text(f"# pack-refs with: peeled fully-peeled sorted\n{SHA_B} refs/heads/other\n"
                                       f"{SHA_A} refs/heads/main\n", encoding="utf-8")  # fmt: skip
    return root


@pytest.mark.parametrize("layout, sha, branch", [("loose", SHA_A, "main"), ("packed", SHA_A, "main"),
                                                 ("worktree", SHA_A, "main"),
                                                 ("detached", SHA_B, "HEAD (detached)")])  # fmt: skip
def test_commit_is_read_from_dot_git(tmp_path, layout, sha, branch):
    repo = _fake_repo(tmp_path / "repo", layout)
    got = git_commit(repo)
    assert got["commit"] == sha and got["branch"] == branch and got["dirty"] is None
    if BASH:
        assert _bash_head_commit(repo) == sha


def test_commit_of_this_checkout_matches_git():
    if GIT is None:
        pytest.skip("no git binary (e.g. a compute node)")
    r = subprocess.run([GIT, "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("this checkout has no commit yet")
    want = r.stdout.strip()
    assert git_commit()["commit"] == want
    if BASH:
        assert _bash_head_commit(REPO) == want


def test_gitignore_reader_matches_git():
    paths = ["configs/eval/private/held_out_secrets.txt", "eval/experiments/x/private/used_secrets.txt",
             "eval/core/io.py", "scripts/slurm/logs/x.out", "eval/.cache/data/x", "outputs/a/b.json",
             "eval/outputs/a/b.json", "a/b/__pycache__/c.pyc", "eval/experiments/_bridges/fsa.py", ".claude/x",
             "eval/secret_types/generated/x.txt", "cluster.env", "paths.local.yaml", "cluster.env.example",
             "results/fsa_lite_run3/gemma_core/team_table.csv", "results/x/team_table.json",
             ".cache/secrets/digits4_100.txt", "a/b/items.jsonl", "a/dev_replies.jsonl"]  # fmt: skip
    in_git = GIT is not None and (REPO / ".git").exists()
    for p in paths:
        ours = gitignored(REPO / p)
        assert ours is not None, p
        if in_git:
            theirs = subprocess.run([GIT, "check-ignore", "-q", "--no-index", p], cwd=REPO).returncode == 0
            assert ours == theirs, p
    assert gitignored(REPO / "configs/eval/private/held_out_secrets.txt") is True
    assert gitignored(REPO / "eval/secret_types/generated/x.txt") is True and gitignored(REPO / "cluster.env") is True
    assert gitignored(REPO / "eval/core/io.py") is False and gitignored(REPO / "cluster.env.example") is False
    assert gitignored(REPO.parent / "elsewhere.txt") is None


def _code_lines(path: Path):
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if s.startswith("#") or not s:
            continue
        yield i, line


COMPUTE_PATHS = (["eval/run.py", "eval/env.sh"]
                 + [str(p.relative_to(REPO)) for d in ("eval/core", "eval/models", "eval/benchmarks", "eval/stats",
                    "eval/metrics", "eval/attackers", "eval/secret_types", "eval/experiments/_bridges", "baselines")
                    for p in sorted((REPO / d).glob("*.py"))]
                 + [str(p.relative_to(REPO)) for d in ("scripts/lib", "scripts/slurm", "scripts/pbs")
                    for p in sorted((REPO / d).iterdir()) if p.suffix in (".sh", ".pbs", ".sbatch")
                    and not (d == "scripts/slurm" and not p.name.startswith("harness_"))])  # fmt: skip


def test_no_git_calls_on_compute_paths():
    py_call = re.compile(r"""\[\s*["']git["']\s*,|which\(\s*["']git["']""")  # an argv list starting with git
    sh_call = re.compile(r"(^|[\s;|&$(`])git\s+(rev-parse|status|check-ignore|describe|log|show|diff|ls-files|branch)\b")
    bad = []
    for rel in COMPUTE_PATHS:
        p = REPO / rel
        for i, line in _code_lines(p):
            if (p.suffix == ".py" and py_call.search(line)) or (p.suffix != ".py" and sh_call.search(line)):
                bad.append(f"{rel}:{i}: {line.strip()}")
    assert not bad, "git binary calls on compute paths:\n" + "\n".join(bad)
    assert any(r.endswith("harness_run.pbs") for r in COMPUTE_PATHS)
    assert "eval/core/io.py" in [c.replace("\\", "/") for c in COMPUTE_PATHS]


# --------------------------------------------------------------------------- PBS scripts

PBS = ["scripts/pbs/harness_run.pbs", "scripts/pbs/harness_adapter_check.pbs"]


def _pbs(path: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for line in (REPO / path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^#PBS\s+(-\w)\s+(.*)$", line)
        if m:
            out.setdefault(m.group(1), []).append(m.group(2).strip())
    return out


def _walltime(s: str) -> int:
    h, m, sec = (int(x) for x in s.split(":"))
    return h * 3600 + m * 60 + sec


@pytest.mark.parametrize("script", PBS)
def test_pbs_scripts(script):
    text = (REPO / script).read_text(encoding="utf-8")
    assert b"\r\n" not in (REPO / script).read_bytes()
    d = _pbs(script)
    assert "-P" not in d and "-q" not in d  # project and queue: cluster.env, passed by scripts/submit_pbs.sh
    assert "select=1:ncpus=16:ngpus=1:mem=110gb" in d["-l"]
    wall = [x.split("=", 1)[1] for x in d["-l"] if x.startswith("walltime=")]
    assert len(wall) == 1
    if script.endswith("fsa_adapter_check.pbs"):
        assert _walltime(wall[0]) <= 3600  # short-queue
    for flag in ("HF_HUB_CACHE", "HF_HUB_OFFLINE", "PBS_JOBID", "head_commit", "python -m core.cluster check"):
        assert flag in text, flag
    assert "HF_HUB_CACHE:?set HF_HUB_CACHE in cluster.env" in text and "source scripts/lib/cluster_env.sh" in text
    # documented -v values: variables separated by commas, none of the values containing one
    for m in re.finditer(r"-v\s+(\S+=(?:\"[^\"]*\"|\S+)(?:,\S+=(?:\"[^\"]*\"|\S+))*)", text):
        pieces = re.findall(r'(\w+)=("[^"]*"|[^,\s]+)', m.group(1))
        assert pieces and ",".join(f"{k}={v}" for k, v in pieces) == m.group(1), m.group(0)
        assert all("," not in v for _, v in pieces), m.group(0)
    if BASH:
        assert subprocess.run([BASH, "-n", script], cwd=REPO).returncode == 0
        env = {k: v for k, v in os.environ.items() if k != "PBS_JOBID"}
        r = subprocess.run([BASH, script], cwd=REPO, env=env, capture_output=True, text=True, timeout=60)
        assert r.returncode == 2 and "refusing to run outside a PBS job" in r.stderr


def test_run_pbs_modes_and_offline_order():
    text = (REPO / "scripts" / "pbs" / "harness_run.pbs").read_text(encoding="utf-8")
    off = text.index('export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"')
    assert text.index("exec bash scripts/lib/setup_env.sh") < off and text.index("python -m core.data fetch") < off
    assert 'SETUP:-0}" = 1' in text and 'FETCH:-0}" = 1' in text and "python run.py $RUN_ARGS" in text
    assert "walltime=03:00:00" in text  # the default (core suite); per-suite estimates in the header
    for suite in ("core 03:00:00", "second 05:00:00", "synthetic 01:30:00", "secret_in_context 04:00:00"):
        assert suite in text


def test_job_scripts_source_the_git_free_helper():
    for s in ("scripts/pbs/harness_run.pbs", "scripts/pbs/harness_adapter_check.pbs", "scripts/slurm/harness_run.sbatch", "scripts/slurm/harness_adapter_check.sbatch"):
        text = (REPO / s).read_text(encoding="utf-8")
        assert "source scripts/lib/job_lib.sh" in text and "commit=$(head_commit)" in text, s


# --------------------------------------------------------------------------- setup: uv installed on request


def test_setup_env_installs_uv_when_asked(tmp_path):
    from tests.test_fsa_bridge import _setup_env_dry, _setup_vars

    stub_python ='#!/bin/bash\\nif [ "$1" = "-m" ]; then exit 1; fi\\necho "$0 3.11.9"\\n'
    installer = ('mkdir -p "$UV_INSTALL_DIR"\n'
                 "cat > \"$UV_INSTALL_DIR/uv\" <<'UVEOF'\n"
                 '#!/bin/bash\n'
                 'if [ "$1" = --version ]; then echo "uv 0.0.0-stub"; exit 0; fi\n'
                 'if [ "$1" = venv ]; then d="${@: -1}"; mkdir -p "$d/bin"; printf \'' + stub_python + '\' > "$d/bin/python"; chmod +x "$d/bin/python"; fi\n'
                 'UVEOF\n'
                 'chmod +x "$UV_INSTALL_DIR/uv"\n')  # fmt: skip
    curl = "#!/bin/bash\ncat <<'INSTALLER'\n" + installer + "INSTALLER\n"
    # UV unset: setup_env.sh looks at $HOME/.local/bin/uv (HOME = tmp_path), absent until the installer runs
    r = _setup_env_dry(tmp_path, _setup_vars(UV_AUTO_INSTALL="1"), with_python311=False,
                       extra_tools={"curl": curl})  # fmt: skip
    out = r.stdout
    assert r.returncode == 0, r.stderr + out
    assert "UV_AUTO_INSTALL: uv / pip caches under" in out and "installing uv from https://astral.sh/uv/install.sh" in out
    assert "uv installed by this job" in out and "installer: uv pip" in out and "3.11" in out
    assert (tmp_path / ".local" / "bin" / "uv").exists()


def test_setup_env_does_not_install_uv_unless_asked(tmp_path):
    from tests.test_fsa_bridge import _setup_env_dry, _setup_vars

    r = _setup_env_dry(tmp_path, _setup_vars(HARNESS_CLUSTER="slurm-site"), with_python311=False,
                       extra_tools={"curl": "#!/bin/bash\nexit 99\n"})  # fmt: skip
    assert r.returncode == 1 and "no Python 3.11" in r.stderr and "installing uv" not in r.stdout


# --------------------------------------------------------------------------- FSA adapter: paths.local.yaml, cluster map, override


def test_fsa_adapter_override_and_record(monkeypatch, tmp_path):
    from experiments._bridges import fsa as F
    from models.dummy import DummyModel

    cfg = {"adapter": {"slurm-site": "/home/x/adapter", "pbs-site": "/scratch/y/adapter"}}
    monkeypatch.setenv("HARNESS_CLUSTER", "pbs-site")
    monkeypatch.setenv("FSA_ADAPTER_DIR", "/scratch/override/adapter")
    b = F.FSABridge(cfg)
    assert b.adapter.as_posix() == "/scratch/override/adapter" and b.adapter_source == "environment FSA_ADAPTER_DIR"
    b.prepare([], None)
    b.wrap(DummyModel())
    assert b.info()["adapter_source"] == "environment FSA_ADAPTER_DIR" and b.info()["cluster"] == "pbs-site"
    monkeypatch.delenv("FSA_ADAPTER_DIR")
    monkeypatch.setenv("HARNESS_CLUSTER", "local")
    b = F.FSABridge(cfg)
    assert b.adapter is None and "no adapter for cluster 'local'" in b.adapter_source


GEMMA_FSA_SHA256 = "46c8b770ec7d4fe7e0f71ba3bb6656001bd819a1f4beb8296ee972a5c1fc5a27"


def test_spec_pins_the_original_gemma_adapter(monkeypatch):
    from core import protocol as P
    from experiments._bridges import fsa as F

    _, cfg = P.variant_configs(P.load_approach("fsa"), "gemma", "technique")
    assert cfg["adapter_sha256"] == GEMMA_FSA_SHA256  # a string (quoted in the YAML), full length
    assert cfg["adapter"] == {"local": "gemma_fsa0"}
    monkeypatch.delenv("FSA_ADAPTER_DIR", raising=False)
    for cluster in ("slurm-site", "pbs-site"):  # one pin, whichever machine resolves the path
        monkeypatch.setenv("HARNESS_CLUSTER", cluster)
        assert F.FSABridge(cfg).cfg["adapter_sha256"] == GEMMA_FSA_SHA256


def test_local_adapter_key_is_resolved_from_paths_local_yaml(monkeypatch, tmp_path):
    from experiments._bridges import fsa as F

    monkeypatch.delenv("FSA_ADAPTER_DIR", raising=False)
    monkeypatch.setenv("FSA_PATHS_FILE", str(tmp_path / "paths.local.yaml"))
    b = F.FSABridge({"adapter": {"local": "gemma_fsa_lite_run3"}})
    assert b.adapter is None and "not found" in b.adapter_source
    (tmp_path / "paths.local.yaml").write_text("adapters:\n  gemma_fsa_lite_run3: /data/adapters/run3\n", encoding="utf-8")
    b = F.FSABridge({"adapter": {"local": "gemma_fsa_lite_run3"}})
    assert b.adapter.as_posix() == "/data/adapters/run3"
    assert b.adapter_source == "paths.local.yaml adapters.gemma_fsa_lite_run3"
    b = F.FSABridge({"adapter": {"local": "missing_key"}})
    assert b.adapter is None and "no adapters.missing_key" in b.adapter_source
    (tmp_path / "paths.local.yaml").write_text("adapters:\n  rel: outputs/training/x/adapter\n", encoding="utf-8")
    assert F.FSABridge({"adapter": {"local": "rel"}}).adapter == F.REPO / "outputs/training/x/adapter"
    monkeypatch.setenv("FSA_ADAPTER_DIR", "/override/adapter")  # still wins
    assert F.FSABridge({"adapter": {"local": "rel"}}).adapter.as_posix() == "/override/adapter"


def test_fsa_adapter_sha256_pin(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")  # noqa: F841
    from experiments._bridges import fsa as F
    from tests.test_fsa_bridge import make_adapter, tiny

    monkeypatch.setenv("HARNESS_DATA", str(tmp_path / "data"))
    monkeypatch.delenv("FSA_ADAPTER_DIR", raising=False)
    m = tiny("tiny-gemma3")
    ad = make_adapter(tmp_path / "ad", F.live_layout(m.load()[1]), "language_model.model.")  # needs a remap
    real = F.sha256_file(ad / F.WEIGHTS)

    def no_load(*a, **k):
        raise AssertionError("the model must not load before the pin is checked")

    # mismatch: refused before anything loads, with the spec path ...
    monkeypatch.setattr(m, "load", no_load)
    b = F.FSABridge({"adapter": str(ad), "adapter_sha256": GEMMA_FSA_SHA256})
    b.prepare([], None)
    with pytest.raises(F.AdapterError, match=f"sha256 {real} does not match adapter_sha256 {GEMMA_FSA_SHA256}.*refusing"):
        b.wrap(m)
    # ... and when FSA_ADAPTER_DIR overrides the path (the pin still applies)
    monkeypatch.setenv("FSA_ADAPTER_DIR", str(ad))
    b = F.FSABridge({"adapter": {"slurm-site": "/nowhere"}, "adapter_sha256": GEMMA_FSA_SHA256})
    assert b.adapter_source == "environment FSA_ADAPTER_DIR"
    b.prepare([], None)
    with pytest.raises(F.AdapterError, match="does not match adapter_sha256"):
        b.wrap(m)
    # malformed pins are refused too
    for bad in (real[:8], "not-a-hash-at-all-0123456789"):
        b = F.FSABridge({"adapter": str(ad), "adapter_sha256": bad})
        b.prepare([], None)
        with pytest.raises(F.AdapterError, match="not a sha256"):
            b.wrap(m)
    monkeypatch.undo()

    # match (full hash, and through FSA_ADAPTER_DIR): loads; the pin was checked on the original
    # file, and defence_info records the original's hash, not the remapped copy's
    pytest.importorskip("peft")
    monkeypatch.setenv("HARNESS_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("FSA_ADAPTER_DIR", str(ad))
    m = tiny("tiny-gemma3")
    b = F.FSABridge({"adapter": {"slurm-site": "/nowhere"}, "adapter_sha256": real.upper()})
    b.prepare([], None)
    b.wrap(m)
    info = b.info()
    assert info["adapter_sha256"] == real and info["adapter_sha256_expected"] == real
    assert info["adapter_sha256_checked"] is True and info["adapter_source"] == "environment FSA_ADAPTER_DIR"
    assert info["remapped"] is True and info["counts"]["missing"] == 0
    loaded = Path(info["loaded_from"]) / F.WEIGHTS
    assert F.sha256_file(loaded) != real  # the remapped copy is a different file (same tensor bytes)
