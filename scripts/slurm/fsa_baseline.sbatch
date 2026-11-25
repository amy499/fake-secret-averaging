#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --job-name=fsa-baseline
#SBATCH --output=scripts/slurm/logs/%x_%j.out
#
# Baseline leakage on the Slurm cluster (see slurm/README.md). Submit from :
#   sbatch scripts/slurm/fsa_baseline.sbatch
#   sbatch --export=ALL,MODEL=gemma3-4b-it,METHOD=prompt_hiding,PRESET=full scripts/slurm/fsa_baseline.sbatch
# Optional (unset = not passed, so the commands above run exactly as before): ADAPTER (a
# trained adapter folder, with METHOD=fsa/b3/random_layer), SECRET_TYPE (digits | words |
# digits4), PROMPT_ID (a user prompt id), EXTRA_ARGS (more options, space-separated, no commas):
#   sbatch --export=ALL,MODEL=gemma3-12b-it,METHOD=fsa,PRESET=full,SECRET_TYPE=digits4,PROMPT_ID=lottery_4digit,ADAPTER=outputs/training/<run>/adapter scripts/slurm/fsa_baseline.sbatch

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
if [ -n "${PROMPT_ID:-}" ]; then optional+=(--prompt-id "$PROMPT_ID"); fi
read -r -a extra <<< "${EXTRA_ARGS:-}"

# Run as a module (-m): run as a file, Python would not find the src/fsa_lite/fsa_runtime/ and scripts/ packages.
uv run --group ml python -m scripts.baseline_leakage \
    --backend real \
    --device cuda \
    --base-model "${MODEL:-qwen2.5-7b-instruct}" \
    --method "${METHOD:-base}" \
    --preset "${PRESET:-quick}" \
    ${optional[@]+"${optional[@]}"} \
    ${extra[@]+"${extra[@]}"}
