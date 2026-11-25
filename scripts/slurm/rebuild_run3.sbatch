#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=00:40:00
#SBATCH --job-name=rebuild_run3
#SBATCH --output=scripts/slurm/logs/%x_%j.out
#
# rebuild_run3.sbatch (Slurm): rebuild the run3 data with the harness held-out secrets excluded
# (build_lite_run3.py --exclude-secrets $HELD_OUT), then the contamination check and the held-out check;
# prints the sha256 of units.jsonl, pools.json, used_secrets.txt and of the held-out FILE (never its
# lines) for the Slurm cluster / the PBS cluster comparison. CPU work (~10 min); a job because Python never runs on the login
# node. Refuses to run over an existing build: rename the old folder first (nothing is deleted).
#   mv ~/fsa_lite_data/built_run3 ~/fsa_lite_data/built_run3_old_heldout
#   cd $REPO_ROOT && \
#   env -u HF_HOME -u HF_HUB_CACHE -u HF_HUB_OFFLINE -u HF_TOKEN -u VIRTUAL_ENV sbatch scripts/slurm/rebuild_run3.sbatch
set -uo pipefail
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/lib/common_lite.sh"
CONFIG_LITE="${CONFIG_LITE:-configs/training/experiments/fsa_lite_gemma12b_run3.yaml}"
lite_banner
lite_require_raw || exit 1
lite_fsa_venv || exit 1
lite_cfg_data || exit 1
[ "$BUILDER" = build_lite_run3.py ] || { echo "$CONFIG_LITE is not a run3 config"; exit 1; }
[ ! -e "$BUILT" ] || { echo "$BUILT exists: rename it first (mv $BUILT ${BUILT}_old_heldout); nothing is deleted"; exit 1; }
lite_build_args || exit 1
echo "held-out file sha256: $(sha256sum "$HELD_OUT" | cut -c1-64)"
uv run --group ml python src/fsa_lite/lite/build_lite_run3.py --raw "$FSA_LITE_DATA/raw" --harness-data "$HARNESS_DATA" \
    --out "$BUILT" "${BUILD_ARGS[@]}" || { echo "REBUILD FAILED"; exit 1; }
uv run --group ml python src/fsa_lite/lite/contam_check.py --data "$BUILT" --harness-data "$HARNESS_DATA" \
    || { echo "contamination check FAILED"; exit 1; }
lite_heldout_check || { echo "held-out check FAILED"; exit 1; }
chmod 600 "$BUILT"/*
echo "SHA256 (compare with the PBS cluster):"
(cd "$BUILT" && sha256sum units.jsonl pools.json used_secrets.txt)
echo "used_secrets lines: $(wc -l < "$BUILT/used_secrets.txt")"
echo "REBUILD OK: $BUILT"
