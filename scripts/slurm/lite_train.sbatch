#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=05:50:00
#SBATCH --job-name=lite_train
#SBATCH --output=scripts/slurm/logs/%x_%j.out
#
# lite_train.sbatch: FSA-lite training, resumable. RUN_DIR (required) names the run; CONFIG_LITE
# picks the config (default Gemma run1). Needs the data the pilot built (the config's data.dir); the
# contamination check is re-run first and a FAIL stops the job. The first job caches replies,
# trains, logs dev metrics every dev.every steps (<run>/dev_log.jsonl), checkpoints every
# train.checkpoint_every steps and, MAX_MINUTES (default 330) after it started, saves a checkpoint
# and exits with code 3; resubmit the SAME command to continue. The finished run holds adapter/
# (PEFT + training_info.json) and v2_meta.json; with a config `selection` block (run3) also
# selection.json, and adapter/ is the checkpoint the pre-registered dev rule chose ("SELECTION:" line).
#   env -u HF_HOME -u HF_HUB_CACHE -u HF_HUB_OFFLINE -u HF_TOKEN -u VIRTUAL_ENV \
#     sbatch --export=ALL,RUN_DIR=outputs/training/fsa_lite_gemma12b_run3,CONFIG_LITE=configs/training/experiments/fsa_lite_gemma12b_run3.yaml scripts/slurm/lite_train.sbatch
set -uo pipefail
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/lib/common_lite.sh"
RUN_DIR="${RUN_DIR:?set RUN_DIR (the same value for every job of the run)}"
lite_banner
lite_fsa_venv || exit 1
lite_cfg_data || exit 1
[ -f "$BUILT/manifest.json" ] || { echo "no built data at $BUILT: run lite_pilot.sbatch with this CONFIG_LITE first"; exit 1; }
uv run --group ml python src/fsa_lite/lite/contam_check.py --data "$BUILT" \
    --harness-data "$HARNESS_DATA" || { echo "contamination check FAILED: nothing trained"; exit 1; }
lite_heldout_check || { echo "held-out check FAILED: nothing trained"; exit 1; }
uv run --group ml python src/fsa_lite/lite/v2_train.py --config "$CONFIG_LITE" --mode train \
    --device cuda --run-dir "$RUN_DIR" --max-minutes "${MAX_MINUTES:-330}"
rc=$?
if [ $rc = 3 ]; then
    echo "NOT FINISHED (time budget): resubmit the same command to continue from the latest checkpoint:"
    echo "  cd $FSA_DIR && env -u HF_HOME -u HF_HUB_CACHE -u HF_HUB_OFFLINE -u HF_TOKEN -u VIRTUAL_ENV sbatch --export=ALL,RUN_DIR=$RUN_DIR,CONFIG_LITE=$CONFIG_LITE${MAX_MINUTES:+,MAX_MINUTES=$MAX_MINUTES} scripts/slurm/lite_train.sbatch"
fi
exit $rc
