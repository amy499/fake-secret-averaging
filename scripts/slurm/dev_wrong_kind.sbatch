#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=dev_wrong_kind
#SBATCH --output=scripts/slurm/logs/%x_%j.out
#
# dev_wrong_kind.sbatch (Slurm): REPORT-ONLY, POST-HOC. Dev kgp leaks of run3 runs by wrong-password kind and
# offered-password similarity, per checkpoint (dev_wrong_kind.py). CPU only (no tokenizer, no model; the
# partition's usual GPU request is kept, nothing uses it; a job because Python never runs on the login
# node). Reads the config's built data (units.jsonl) and each run's dev_replies.jsonl / selection.json;
# changes nothing. RUNS = NAME=RUN_DIR entries joined by + (no commas); the first is the reference.
#   cd $REPO_ROOT && \
#   env -u HF_HOME -u HF_HUB_CACHE -u HF_HUB_OFFLINE -u HF_TOKEN -u VIRTUAL_ENV sbatch --export=ALL,CONFIG_LITE=configs/training/experiments/fsa_lite_gemma12b_run3.yaml,RUNS=run3=outputs/training/fsa_lite_gemma12b_run3,OUT_NAME=gemma scripts/slurm/dev_wrong_kind.sbatch
set -uo pipefail
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/lib/common_lite.sh"
CONFIG_LITE="${CONFIG_LITE:-configs/training/experiments/fsa_lite_gemma12b_run3.yaml}"
RUNS="${RUNS:-run3=outputs/training/fsa_lite_gemma12b_run3}"
OUT_NAME="${OUT_NAME:-gemma}"
lite_banner
lite_fsa_venv || exit 1
lite_cfg_data || exit 1
[ -f "$BUILT/units.jsonl" ] || { echo "no units.jsonl in $BUILT"; exit 1; }
echo "data sha256: $(cd "$BUILT" && sha256sum units.jsonl | cut -c1-16)"
RUN_ARGS=()
IFS='+' read -r -a ENTRIES <<< "$RUNS"
for e in "${ENTRIES[@]}"; do
    [ -f "${e#*=}/dev_replies.jsonl" ] || { echo "no dev_replies.jsonl in ${e#*=}"; exit 1; }
    RUN_ARGS+=(--run "$e")
done
uv run --group ml python src/fsa_lite/lite/dev_wrong_kind.py --data "$BUILT" "${RUN_ARGS[@]}" \
    --out "outputs/diagnostics/dev_wrong_kind/$OUT_NAME.json"
