#!/bin/bash
# Build .venv with the pinned GPU stack (eval/requirements-gpu.txt) and check it. Run it INSIDE a
# job, on both clusters, never on a login / head node:
#   Slurm: SBATCH_EXTRA=--export=ALL,SETUP=1 bash scripts/submit_slurm.sh scripts/slurm/harness_run.sbatch --setup
#   PBS:   QSUB_EXTRA="-l walltime=01:00:00 -v SETUP=1" bash scripts/submit_pbs.sh scripts/pbs/harness_run.pbs
# Python 3.11, first found of:
#   1. $PYTHON (pass it with --export / -v), used with `python -m venv` and pip;
#   2. python3.11 on PATH, likewise;
#   3. uv ($UV, default $HOME/.local/bin/uv): `uv venv --python 3.11` (uv may download a managed
#      Python 3.11) and `uv pip install`;
#   4. with UV_AUTO_INSTALL=1 (cluster.env, for a site without Python 3.11 and with a small home quota):
#      if none of the above exists, install uv into $HOME/.local/bin with the official installer
#      (curl https://astral.sh/uv/install.sh; UV_VERSION=x.y.z pins a version), then use it.
# With UV_AUTO_INSTALL=1 the uv / pip caches and uv's managed Pythons also go to eval/.cache (keep the
# checkout on scratch), not $HOME. Otherwise the defaults are kept.
# It prints which interpreter and installer it used. It never deletes an existing .venv: it stops.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../eval" && pwd)"   # the harness
cd "$ROOT"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
UV="${UV:-$HOME/.local/bin/uv}"
AUTO=0
if [ "${UV_AUTO_INSTALL:-0}" = 1 ]; then
    AUTO=1
    export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.cache/uv}"
    export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$ROOT/.cache/uv-python}"
    export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/.cache/pip}"
    echo "UV_AUTO_INSTALL: uv / pip caches under $ROOT/.cache"
fi

install_uv() {  # official installer, into $HOME/.local/bin (no shell profile changes)
    local url="https://astral.sh/uv/install.sh"
    [ -n "${UV_VERSION:-}" ] && url="https://astral.sh/uv/${UV_VERSION}/install.sh"
    command -v curl >/dev/null 2>&1 || { echo "STOP: curl not found, cannot install uv" >&2; exit 1; }
    echo "installing uv from $url into $HOME/.local/bin"
    curl -LsSf "$url" | env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 INSTALLER_NO_MODIFY_PATH=1 sh
    UV="$HOME/.local/bin/uv"
    [ -x "$UV" ] || { echo "STOP: uv not found at $UV after installing" >&2; exit 1; }
    echo "uv: $("$UV" --version)"
}

if [ ! -x .venv/bin/python ]; then
    if [ -e .venv ]; then
        echo "STOP: .venv exists but has no bin/python; remove it yourself if it is broken" >&2
        exit 1
    fi
    if [ -n "${PYTHON:-}" ]; then
        echo "python for .venv: \$PYTHON=$PYTHON"
        "$PYTHON" -m venv .venv
    elif command -v python3.11 >/dev/null 2>&1; then
        echo "python for .venv: $(command -v python3.11)"
        python3.11 -m venv .venv
    elif [ -x "$UV" ]; then
        echo "python for .venv: uv ($UV) venv --python 3.11 (python3.11 not on PATH)"
        "$UV" venv --python 3.11 .venv
    elif [ "$AUTO" = 1 ]; then
        install_uv
        echo "python for .venv: uv ($UV) venv --python 3.11 (python3.11 not on PATH; uv installed by this job)"
        "$UV" venv --python 3.11 .venv
    else
        echo "STOP: no Python 3.11: set PYTHON=/path/to/python3.11, put python3.11 on PATH, or install uv at $UV" >&2
        exit 1
    fi
fi

# Installer: pip if the venv has it (python -m venv), else uv pip (a uv venv has no pip).
if .venv/bin/python -m pip --version >/dev/null 2>&1; then
    INSTALLER=pip
    pip_install() { .venv/bin/python -m pip install "$@"; }
    .venv/bin/python -m pip install --upgrade pip
elif [ -x "$UV" ]; then
    INSTALLER="uv pip ($UV)"
    pip_install() { "$UV" pip install --python .venv/bin/python "$@"; }
else
    echo "STOP: .venv has no pip and uv is not at $UV" >&2
    exit 1
fi
echo "interpreter: $(.venv/bin/python -c 'import sys; print(sys.executable, sys.version.split()[0])')"
echo "installer: $INSTALLER"
.venv/bin/python -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)' \
    || { echo "STOP: .venv is not Python 3.11" >&2; exit 1; }
if [ "${SETUP_DRY:-0}" = 1 ]; then  # tests: stop after choosing the interpreter and installer
    echo "SETUP_DRY: stop before installing"
    exit 0
fi

pip_install torch==2.6.0 --index-url "$TORCH_INDEX"
pip_install -r eval/requirements-gpu.txt
if [ "$INSTALLER" = pip ]; then
    .venv/bin/python -m pip freeze > .venv/PIP_FREEZE.txt
else
    "$UV" pip freeze --python .venv/bin/python > .venv/PIP_FREEZE.txt
fi

# shellcheck disable=SC1091
source env.sh
python -m core.cluster check
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "no CUDA GPU visible: run this inside a GPU job"
x = torch.ones(2, 2, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul on", torch.cuda.get_device_name(0), "->", float((x @ x).sum()))
EOF
# Needs internet from the compute node; if it has none, fetch on a machine that does and copy
# $HARNESS_DATA (default .cache/data) over. The IFEval preflight then finds it.
python -m core.data fetch nltk_punkt_tab || echo "WARNING: could not fetch punkt_tab here (no internet?): copy .cache/data/ over"
python -m pytest -q
echo "setup OK: $(python -m core.cluster check | head -1)"
