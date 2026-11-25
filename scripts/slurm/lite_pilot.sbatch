#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH --job-name=lite_pilot
#SBATCH --output=scripts/slurm/logs/%x_%j.out
#
# lite_pilot.sbatch: FSA-lite timing pilot on one A40. Builds the training data if the config's
# data.dir has none (its data.builder, default build_lite.py; CPU, ~6-10 min), re-runs the contamination check
# (stops on FAIL), then v2_train --mode pilot: replies for pilot.n_reply_keys units, one dev
# evaluation, pilot.steps training steps; prints "PILOT:" with sec/step, the projected full run
# and steps_for_1h15; writes <run>/pilot_timing.json. No adapter.
#   env -u HF_HOME -u HF_HUB_CACHE -u HF_HUB_OFFLINE -u HF_TOKEN -u VIRTUAL_ENV sbatch scripts/slurm/lite_pilot.sbatch
#   (Qwen: sbatch --export=ALL,CONFIG_LITE=configs/training/experiments/fsa_lite_qwen7b.yaml scripts/slurm/lite_pilot.sbatch)
#   (run3: sbatch --export=ALL,CONFIG_LITE=configs/training/experiments/fsa_lite_gemma12b_run3.yaml,RUN_DIR=outputs/training/fsa_lite_gemma12b_run3_pilot ...)
set -uo pipefail
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/lib/common_lite.sh"
lite_banner
lite_require_raw || exit 1
lite_fsa_venv || exit 1
lite_build_data || { echo "data build or contamination check FAILED: nothing trained"; exit 1; }
uv run --group ml python src/fsa_lite/lite/v2_train.py --config "$CONFIG_LITE" --mode pilot \
    --device cuda ${RUN_DIR:+--run-dir "$RUN_DIR"}
