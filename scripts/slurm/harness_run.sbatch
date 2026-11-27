#!/bin/bash
# The Slurm cluster (Slurm) job template: one team-protocol run or one experiment config, on one GPU.
# The PBS cluster (PBS) twin is scripts/pbs/harness_run.pbs; both check the pinned stack before running.
#
#   sbatch --export=ALL,SETUP=1 scripts/slurm/harness_run.sbatch --setup                                  # once: build .venv (scripts/lib/setup_env.sh)
#   sbatch --export=ALL,FETCH=1 scripts/slurm/harness_run.sbatch --fetch                                  # once: fetch + verify the pinned data
#   sbatch scripts/slurm/harness_run.sbatch --approach fsa --model gemma --suite core --variant original   # team protocol
#   sbatch scripts/slurm/harness_run.sbatch configs/hidden_code__none.yaml                               # attack study
#   sbatch --export=ALL,EXTRA_ARGS="--output-dir /scratch/$USER/outputs" scripts/slurm/harness_run.sbatch configs/x.yaml
#   sbatch --export=ALL,MODULE=analysis.internals.run,EXTRA_ARGS="--parts 2" scripts/slurm/harness_run.sbatch analysis/internals/config.yaml
#   sbatch --export=ALL,MODULE=stats.rescore scripts/slurm/harness_run.sbatch outputs/protocol/fsa/gemma                 # re-score runs
#   sbatch --export=ALL,MODULE=stats.team_table scripts/slurm/harness_run.sbatch build outputs/protocol --approach fsa --model gemma --suite core --out outputs/team/fsa_gemma_core
# (--export splits on commas: never put a comma inside EXTRA_ARGS; use + in --benchmarks a+b.)
# ALLOW_UNPINNED=1 runs even if the installed versions differ from eval/requirements-gpu.txt
# (the run is then recorded as not protocol-compliant and will not pair with pinned runs).
# HARNESS_HF_HUB_CACHE=/path/to/hub (in --export) reuses an existing model snapshot (env.sh).
# Partition and QoS are fixed (<gpu-partition> / normal), at most 10 CPUs and 05:50:00: never override
# them on the sbatch line. The Slurm cluster allows at most 2 submitted jobs per user at a time.
#
# Never run model code on the head node: submit with sbatch (or srun for an interactive GPU shell).
#SBATCH --job-name=harness
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:50:00
#SBATCH --output=scripts/slurm/logs/%x_%j.out
#SBATCH --error=scripts/slurm/logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"   # the repository root (scripts/submit_slurm.sh)
# shellcheck disable=SC1091
source scripts/lib/cluster_env.sh
mkdir -p scripts/slurm/logs

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "refusing to run outside a Slurm job (head node). Use: sbatch scripts/slurm/harness_run.sbatch ..." >&2
    exit 2
fi

# The Slurm cluster model cache: the existing one, $HOME/hf_cache (models--google--gemma-3-12b-it directly
# under it, snapshot 96b6f1ec...), never a new cache inside the repo. Override with
# --export=ALL,HARNESS_HF_HUB_CACHE=/path/to/cache (or HF_HUB_CACHE=...).
export HF_HUB_CACHE="${HARNESS_HF_HUB_CACHE:-${HF_HUB_CACHE:?set HF_HUB_CACHE in cluster.env}}"

# shellcheck disable=SC1091
source scripts/lib/job_lib.sh   # head_commit: reads .git, no git binary
echo "job=$SLURM_JOB_ID cluster=$HARNESS_CLUSTER node=$(hostname) commit=$(head_commit)"
# shellcheck disable=SC2086  (EXTRA_ARGS is a word list on purpose)
model_cache_report "$@" ${EXTRA_ARGS:-}   # the model of --model (both if none)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

if [ "${SETUP:-0}" = 1 ]; then
    exec bash scripts/lib/setup_env.sh   # builds eval/.venv
fi
if [ "${FETCH:-0}" = 1 ]; then
    # data download + hash check, inside a job (no Python on the login node)
    # shellcheck disable=SC1091
    cd eval   # the harness: run.py, core/, ...
    # shellcheck disable=SC1091
    source env.sh
    python -m core.data fetch
    exec python -m core.data verify
fi

# Model runs load the pinned revision from the cache only (no download); SETUP / FETCH above
# keep network access. HF_HUB_OFFLINE=0 in --export allows a download.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
CONFIG="${1:?usage: sbatch scripts/slurm/harness_run.sbatch configs/<name>.yaml | --approach <name> --model gemma|qwen --variant original|technique [--suite core]}"
# shellcheck disable=SC1091
cd eval   # the harness: run.py, core/, ...
# shellcheck disable=SC1091
source env.sh
if ! python -m core.cluster check; then
    [ "${ALLOW_UNPINNED:-0}" = 1 ] || { echo "stop: install the pinned stack (SETUP=1) or set ALLOW_UNPINNED=1" >&2; exit 3; }
fi

if [ -n "${MODULE:-}" ]; then
    case "$MODULE" in
        stats.*)
            # CPU tools with their own arguments (no --config / --device), e.g.
            #   sbatch --export=ALL,MODULE=stats.rescore scripts/slurm/harness_run.sbatch outputs/protocol/fsa/gemma
            python -m "$MODULE" "$@" ${EXTRA_ARGS:-} ;;
        *)
            # e.g. MODULE=analysis.internals.run (any entry point taking --config and --device)
            python -m "$MODULE" --config "$CONFIG" --device cuda ${EXTRA_ARGS:-} ;;
    esac
elif [ "${CONFIG#--}" != "$CONFIG" ]; then
    # team protocol: every argument goes to run.py (core/protocol.py)
    python run.py "$@" --device cuda ${EXTRA_ARGS:-}
else
    python run.py --config "$CONFIG" --device cuda ${EXTRA_ARGS:-}
fi
