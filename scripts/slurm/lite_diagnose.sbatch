#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --job-name=lite_diag
#SBATCH --output=scripts/slurm/logs/%x_%j.out
#
# lite_diagnose.sbatch: diagnose an FSA-lite run from its own files (lite_diagnose.py): leaking cached
# replies per type, the recomputed draws, the FSA term with / without a leaking reply, dilution,
# adapter weights; with GENERATE=1 also the dev evaluation with the adapter off and on (dev data only).
#   env -u HF_HOME -u HF_HUB_CACHE -u HF_HUB_OFFLINE -u HF_TOKEN -u VIRTUAL_ENV \
#     sbatch --export=ALL,RUN_DIR=outputs/training/fsa_lite_gemma12b_run1,GENERATE=1 scripts/slurm/lite_diagnose.sbatch
set -uo pipefail
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/lib/common_lite.sh"
RUN_DIR="${RUN_DIR:?set RUN_DIR, e.g. outputs/training/fsa_lite_gemma12b_run1}"
lite_banner
lite_fsa_venv || exit 1
uv run --group ml python src/fsa_lite/lite/lite_diagnose.py --run-dir "$RUN_DIR" --device cuda \
    --dev-n "${DEV_N:-112}" ${GENERATE:+--generate}
