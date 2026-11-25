#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --job-name=fsa-utility
#SBATCH --output=scripts/slurm/logs/%x_%j.out
#
# Utility benchmarks (scripts/utility_eval.py) on the Slurm cluster; see slurm/README.md. The dataset cache
# must exist first (login node: uv run --group ml python -m scripts.utility_eval --prepare
# --prepare-model gemma3-12b-it). Submit from :
#   sbatch --export=ALL,MODEL=gemma3-12b-it scripts/slurm/fsa_utility.sbatch                  # base model
#   sbatch --export=ALL,MODEL=gemma3-12b-it,SECRET_TYPE=digits4,ADAPTER=outputs/training/<run>/adapter scripts/slurm/fsa_utility.sbatch
# Optional: ADAPTER, SECRET_TYPE (digits | digits4; digits4 adds pe_access), EXTRA_ARGS
# (space-separated, no commas; e.g. --n-mmlu 300).

set -euo pipefail

echo "host: $(hostname)"
echo "date: $(date)"
nvidia-smi
echo "git:  $(git rev-parse --short HEAD)"

cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"

optional=()
if [ -n "${ADAPTER:-}" ]; then optional+=(--adapter "$ADAPTER"); fi
if [ -n "${SECRET_TYPE:-}" ]; then optional+=(--secret-type "$SECRET_TYPE"); fi
read -r -a extra <<< "${EXTRA_ARGS:-}"

uv run --group ml python -m scripts.utility_eval \
    --device cuda \
    --base-model "${MODEL:-gemma3-12b-it}" \
    ${optional[@]+"${optional[@]}"} \
    ${extra[@]+"${extra[@]}"}
