#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=00:20:00
#SBATCH --job-name=preflight_fakes
#SBATCH --output=scripts/slurm/logs/%x_%j.out
#
# preflight_fakes.sbatch (Slurm): checks of one run3 config's built data, CPU only (no model is loaded; a
# job because Python never runs on the login node):
#  1. the contamination and held-out checks of the built data (stops on FAIL);
#  2. (optional, OLD_DATA + OLD_REPLIES + NEW_RUN set) reuse_replies.py: the old run's cached replies whose
#     prompt is byte-identical in the new data go to NEW_RUN/replies.jsonl (refuses to overwrite), the rest
#     is left for the train / pilot job to generate ("REUSE_TO_GENERATE n");
#  3. preflight_fakes.py: the model's tokenizer (offline, pinned revision), the token-length filter, N
#     (default 200) seeded joint fake draws per slot unit (train and dev), valid-tuple rates, fallback
#     outcome. PREFLIGHT_RUN_DIR (optional, a run with ALL train replies) marks the leaking-reply draw pool.
# Prints uids, counts, rates, lengths and token counts only (never a value).
#   cd $REPO_ROOT && \
#   env -u HF_HOME -u HF_HUB_CACHE -u HF_HUB_OFFLINE -u HF_TOKEN -u VIRTUAL_ENV sbatch --export=ALL,CONFIG_LITE=configs/training/experiments/fsa_lite_gemma12b_run3.yaml,OLD_DATA=$HOME/fsa_lite_data/built_run3_old_heldout,OLD_REPLIES=outputs/training/fsa_lite_gemma12b_run3_crashed55932/replies.jsonl,NEW_RUN=outputs/training/fsa_lite_gemma12b_run3 scripts/slurm/preflight_fakes.sbatch
set -uo pipefail
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/lib/common_lite.sh"
CONFIG_LITE="${CONFIG_LITE:-configs/training/experiments/fsa_lite_gemma12b_run3.yaml}"
N="${N:-200}"
lite_banner
lite_fsa_venv || exit 1
lite_cfg_data || exit 1
[ -f "$BUILT/manifest.json" ] || { echo "no built data at $BUILT: run rebuild_run3.sbatch first"; exit 1; }
uv run --group ml python src/fsa_lite/lite/contam_check.py --data "$BUILT" --harness-data "$HARNESS_DATA" \
    || { echo "contamination check FAILED"; exit 1; }
lite_heldout_check || { echo "held-out check FAILED"; exit 1; }
echo "data sha256: $(cd "$BUILT" && sha256sum units.jsonl pools.json used_secrets.txt | cut -c1-16 | tr '\n' ' ')"
if [ -n "${OLD_DATA:-}" ] && [ -n "${OLD_REPLIES:-}" ] && [ -n "${NEW_RUN:-}" ]; then
    [ -f "$OLD_DATA/units.jsonl" ] && [ -f "$OLD_REPLIES" ] || { echo "OLD_DATA/units.jsonl or OLD_REPLIES missing"; exit 1; }
    mkdir -p "$NEW_RUN"
    uv run --group ml python src/fsa_lite/lite/reuse_replies.py --config "$CONFIG_LITE" --old-data "$OLD_DATA" \
        --new-data "$BUILT" --replies "$OLD_REPLIES" --out "$NEW_RUN/replies.jsonl" || exit 1
fi
NAME="$(basename "$CONFIG_LITE" .yaml)"
RUN_ARG=()
if [ -n "${PREFLIGHT_RUN_DIR:-}" ]; then RUN_ARG=(--run-dir "$PREFLIGHT_RUN_DIR"); echo "draw pool from $PREFLIGHT_RUN_DIR/replies.jsonl"; fi
uv run --group ml python src/fsa_lite/lite/preflight_fakes.py --config "$CONFIG_LITE" --n "$N" ${RUN_ARG[@]+"${RUN_ARG[@]}"} \
    --out "outputs/diagnostics/preflight_fakes/$NAME.json"
