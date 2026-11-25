#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --job-name=fsa-train
#SBATCH --output=scripts/slurm/logs/%x_%j.out
#
# LoRA training (FSA / B3 / random-layer control); see slurm/README.md. Submit from
# :
#   sbatch --export=ALL,CONFIG=configs/training/experiments/fsa_gemma4b_debug.yaml scripts/slurm/fsa_train.sbatch
#   sbatch --export=ALL,RESUME=outputs/training/<run> scripts/slurm/fsa_train.sbatch
#
# 06:00:00 is the Slurm cluster's per-job maximum. The trainer checkpoints and exits (code 3) MAX_MINUTES
# (default 330) after it starts, before the limit kills the job; resubmit with RESUME=<run>.

set -euo pipefail
if [ -z "${CONFIG:-}" ] && [ -z "${RESUME:-}" ]; then
    echo "set CONFIG=configs/training/experiments/<name>.yaml or RESUME=outputs/training/<run>" >&2
    exit 1
fi

echo "host: $(hostname)"
echo "date: $(date)"
nvidia-smi
echo "git:  $(git rev-parse --short HEAD)"

cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"

# RESUME wins over CONFIG: a resumed run keeps the config it started with.
if [ -n "${RESUME:-}" ]; then
    source_args=(--resume "$RESUME")
else
    source_args=(--config "$CONFIG")
fi
read -r -a extra <<< "${EXTRA_ARGS:-}"
uv run --group ml python -m scripts.train_fsa "${source_args[@]}" --device cuda \
    --max-minutes "${MAX_MINUTES:-330}" \
    ${extra[@]+"${extra[@]}"}
