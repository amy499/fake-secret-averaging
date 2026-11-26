#!/bin/bash
# The Slurm cluster job: load google/gemma-3-12b-it + the FSA adapter from configs/eval/fsa.approach.yaml
# through the bridge on the pinned stack, print the verification counts (LoRA A/B attached,
# missing, unexpected, zero lora_B, r/alpha, remapped or not), then generate --n greedy replies on
# secret_in_context items with and without the adapter and print whether each pair is identical.
# It prints item keys, counts and hashes only: never secrets, prompts, replies or tokens.
#
#   sbatch scripts/slurm/harness_adapter_check.sbatch                      # after SETUP=1 and FETCH=1 (scripts/slurm/harness_run.sbatch)
#   sbatch --export=ALL,CHECK_ARGS="--n 8 --benchmark gsm8k_team_sic" scripts/slurm/harness_adapter_check.sbatch
#   sbatch --export=ALL,HARNESS_HF_HUB_CACHE=/path/to/cache scripts/slurm/harness_adapter_check.sbatch   # another model cache
# Needs: the pinned stack in .venv, the MMLU/GSM8K/IFEval data under $HARNESS_DATA,
# the approach's configs/eval/private/ files (held_out_secrets.txt and its used-secrets file), and the Gemma licence + an HF token
# (HF_TOKEN in the environment, or ~/.cache/huggingface/token, used via HF_TOKEN_PATH). Without a
# cached snapshot the job stops (it runs with HF_HUB_OFFLINE=1; the model is about 24 GB).
# Partition and QoS are fixed (<gpu-partition> / normal): never override them on the sbatch line.
# Never run it on the login node: it refuses outside a Slurm job.
#SBATCH --job-name=fsa_adapter_check
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=scripts/slurm/logs/%x_%j.out
#SBATCH --error=scripts/slurm/logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"   # the repository root (scripts/submit_slurm.sh)
# shellcheck disable=SC1091
source scripts/lib/cluster_env.sh
mkdir -p scripts/slurm/logs

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "refusing to run outside a Slurm job (login node). Use: sbatch scripts/slurm/harness_adapter_check.sbatch" >&2
    exit 2
fi

# The Slurm cluster model cache: the existing one, $HOME/hf_cache (models--google--gemma-3-12b-it directly
# under it, snapshot 96b6f1ec...), never a new cache inside the repo. Override with
# --export=ALL,HARNESS_HF_HUB_CACHE=/path/to/cache (or HF_HUB_CACHE=...).
export HF_HUB_CACHE="${HARNESS_HF_HUB_CACHE:-${HF_HUB_CACHE:?set HF_HUB_CACHE in cluster.env}}"

# shellcheck disable=SC1091
source scripts/lib/job_lib.sh   # head_commit: reads .git, no git binary
echo "job=$SLURM_JOB_ID cluster=$HARNESS_CLUSTER node=$(hostname) commit=$(head_commit)"
echo "HF_HUB_CACHE=$HF_HUB_CACHE (gemma-3-12b-it cached: $([ -d "$HF_HUB_CACHE/models--google--gemma-3-12b-it" ] && echo yes || echo no))"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"   # load the pinned revision from the cache, never download
# shellcheck disable=SC1091
cd eval   # the harness: run.py, core/, ...
# shellcheck disable=SC1091
source env.sh   # HF cache: HARNESS_HF_HUB_CACHE / HF_HUB_CACHE / HF_HOME (see env.sh); HF_TOKEN_PATH set there
python -m core.cluster check || { echo "stop: the installed versions differ from eval/requirements-gpu.txt" >&2; exit 3; }

# shellcheck disable=SC2086  (CHECK_ARGS is a word list on purpose)
python -m experiments._bridges.fsa check --model gemma --device cuda ${CHECK_ARGS:-}
