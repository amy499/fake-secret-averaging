"""Where a run ran: cluster, host, GPU and library versions, and a check against the pinned GPU
stack (eval/requirements-gpu.txt), the same on the Slurm cluster and the PBS cluster (PBS).

Status: ✅ implemented + tested (tests/test_cluster.py: detection from the scheduler environment,
pin parsing and comparison, the pairing rule). 🟡 on the clusters: the pinned stack has not
been installed on the Slurm cluster or the PBS cluster from this branch yet (scripts/lib/setup_env.sh does it and runs this check).

    python -m core.cluster            # print the environment record (JSON)
    python -m core.cluster check      # exit 1 unless the installed versions equal the pins

Cluster name: $HARNESS_CLUSTER if set (the job scripts set slurm-site / pbs-site), else inferred:
a Slurm job on a partition containing "the Slurm cluster" -> slurm-site (the Slurm cluster's GPU partition is <gpu-partition>); a PBS job
submitted from an asp2a host -> pbs-site; any other Slurm / PBS job -> "slurm" / "pbs";
otherwise "local". `environment()` is what summary.json records under "environment".
"""

from __future__ import annotations

import json
import os
import platform
import re
import socket
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

PIN_FILE = Path(__file__).resolve().parent.parent / "requirements-gpu.txt"
# Libraries whose versions must match between paired runs and the pins (distribution names).
LIBRARIES = ("torch", "transformers", "tokenizers", "accelerate", "peft", "safetensors", "huggingface_hub",
             "numpy", "nltk", "langdetect", "absl-py", "immutabledict")  # fmt: skip


def cluster_info(env: dict[str, str] | None = None) -> dict[str, Any]:
    env = dict(os.environ if env is None else env)
    if env.get("SLURM_JOB_ID"):
        scheduler, job = "slurm", env["SLURM_JOB_ID"]
    elif env.get("PBS_JOBID"):
        scheduler, job = "pbs", env["PBS_JOBID"]
    else:
        scheduler, job = None, None
    name = env.get("HARNESS_CLUSTER") or scheduler or "local"  # HARNESS_CLUSTER: CLUSTER_NAME in cluster.env
    return {"cluster": name, "hostname": socket.gethostname(), "scheduler": scheduler, "job_id": job}


def library_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for dist in LIBRARIES:
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    return out


def _driver() -> str | None:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else None


def gpu_info() -> dict[str, Any]:
    """GPU name, count, memory and compute capability, CUDA build of torch, cuDNN, driver.
    name None = no CUDA GPU visible (CPU run, dry run)."""
    try:
        import torch
    except ImportError:
        return {"name": None, "count": 0, "torch_cuda": None, "cudnn": None, "driver": None}
    info: dict[str, Any] = {"name": None, "count": 0, "torch_cuda": torch.version.cuda,
                            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                            "driver": None}  # fmt: skip
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        info.update(name=p.name, count=torch.cuda.device_count(), memory_gb=round(p.total_memory / 2**30, 1),
                    capability=f"{p.major}.{p.minor}", driver=_driver())  # fmt: skip
    return info


def read_pins(path: Path = PIN_FILE) -> dict[str, str]:
    """name -> version from `name==version` lines (comments and other lines ignored); names are
    normalised (case, - vs _) so they compare with metadata names."""
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)\s*==\s*([^\s;#]+)", line)
        if m:
            pins[_norm(m.group(1))] = m.group(2)
    return pins


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def compare_pins(versions: dict[str, str | None], pins: dict[str, str]) -> list[str]:
    """Differences between installed versions and the pins. torch's local tag (+cu124) is not part
    of the pin; the CUDA build is recorded separately (gpu.torch_cuda)."""
    diffs = []
    inst = {_norm(k): v for k, v in versions.items()}
    for name, want in sorted(pins.items()):
        if name not in inst:
            continue  # pinned but not tracked here (e.g. pytest, pyyaml)
        got = inst[name]
        base = got.split("+")[0] if got else None
        if base != want:
            diffs.append(f"{name} {got} (pinned {want})")
    return diffs


def environment() -> dict[str, Any]:
    """The record summary.json stores under "environment"."""
    versions = library_versions()
    try:
        pins = read_pins()
        diffs = compare_pins(versions, pins)
        pin_rec = {"file": "eval/requirements-gpu.txt", "match": not diffs, "differences": diffs}
    except OSError as e:
        pin_rec = {"file": "eval/requirements-gpu.txt", "match": False, "differences": [f"pin file unreadable: {e}"]}
    return {**cluster_info(), "python": platform.python_version(), "platform": platform.platform(),
            "gpu": gpu_info(), "libraries": versions, "pins": pin_rec}  # fmt: skip


def pairing_differences(a: dict[str, Any] | None, b: dict[str, Any] | None) -> list[str]:
    """Why two runs' environments must not be paired: GPU name, CUDA build of torch, or any
    library version differs (cluster and host may differ as long as these are equal)."""
    if not a or not b:
        return ["environment not recorded"]
    out = []
    ga, gb = a.get("gpu") or {}, b.get("gpu") or {}
    if ga.get("name") != gb.get("name"):
        out.append(f"GPU {ga.get('name')} vs {gb.get('name')}")
    if ga.get("torch_cuda") != gb.get("torch_cuda"):
        out.append(f"torch CUDA build {ga.get('torch_cuda')} vs {gb.get('torch_cuda')}")
    la, lb = a.get("libraries") or {}, b.get("libraries") or {}
    for k in sorted(set(la) | set(lb)):
        if la.get(k) != lb.get(k):
            out.append(f"{k} {la.get(k)} vs {lb.get(k)}")
    return out


def where(env: dict[str, Any] | None) -> str:
    """Short label for tables: cluster / GPU."""
    if not env:
        return "?"
    return f"{env.get('cluster') or '?'} / {(env.get('gpu') or {}).get('name') or 'no GPU'}"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    env = environment()
    if args[:1] == ["check"]:
        print(f"{where(env)} on {env['hostname']}; torch CUDA build {env['gpu'].get('torch_cuda')}")
        if env["pins"]["match"]:
            print("versions match eval/requirements-gpu.txt")
            return 0
        print("versions differ from eval/requirements-gpu.txt:\n  " + "\n  ".join(env["pins"]["differences"]))
        return 1
    print(json.dumps(env, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
