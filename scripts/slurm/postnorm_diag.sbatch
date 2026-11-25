#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=00:40:00
#SBATCH --job-name=postnorm_diag
#SBATCH --output=scripts/slurm/logs/%x_%j.out
#
# postnorm_diag.sbatch (the Slurm cluster, one A40): the REPORT-ONLY post-norm diagnostic (postnorm_diag.py) for the
# Gemma FSA-lite run2 adapter on 50 synthetic key-guarded prompts (not PasswordEval). Two forward
# passes per prompt (adapter off / on), no generation: a few minutes after the model loads. Writes
# outputs/diagnostics/postnorm/gemma_run2.{csv,json} and prints one line per layer and "SUMMARY:".
#   cd $REPO_ROOT && \
#   env -u HF_HOME -u HF_HUB_CACHE -u HF_HUB_OFFLINE -u HF_TOKEN -u VIRTUAL_ENV sbatch scripts/slurm/postnorm_diag.sbatch
set -uo pipefail
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/lib/common_lite.sh"
CONFIG_LITE="${CONFIG_LITE:-configs/training/experiments/fsa_lite_gemma12b_run2.yaml}"
ADAPTER="${ADAPTER:-outputs/training/fsa_lite_gemma12b_run2/adapter}"
OUT="${OUT:-outputs/diagnostics/postnorm/gemma_run2}"
lite_banner
lite_fsa_venv || exit 1
[ -f "$ADAPTER/adapter_model.safetensors" ] || { echo "no adapter at $ADAPTER"; exit 1; }
uv run --group ml python src/fsa_lite/lite/postnorm_diag.py run --config "$CONFIG_LITE" --adapter "$ADAPTER" \
    --out "$OUT" --device cuda
