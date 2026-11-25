#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=00:30:00
#SBATCH --job-name=train_signal_audit
#SBATCH --output=scripts/slurm/logs/%x_%j.out
#
# train_signal_audit.sbatch (Slurm): H3 report-only audit of a finished run3 run, CPU only (tokenizer, no
# model; a job because Python never runs on the login node): buckets of the cached base replies of the
# FULL train set, the 500 x grad_accum draws replayed and checked against train_log.jsonl, loss shares,
# dev rates per checkpoint. Counts only (never a value). Changes nothing in the run folder.
#   cd $REPO_ROOT && \
#   env -u HF_HOME -u HF_HUB_CACHE -u HF_HUB_OFFLINE -u HF_TOKEN -u VIRTUAL_ENV sbatch --export=ALL,CONFIG_LITE=configs/training/experiments/fsa_lite_gemma12b_run3.yaml,RUN_DIR=outputs/training/fsa_lite_gemma12b_run3 scripts/slurm/train_signal_audit.sbatch
set -uo pipefail
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/lib/common_lite.sh"
CONFIG_LITE="${CONFIG_LITE:-configs/training/experiments/fsa_lite_gemma12b_run3.yaml}"
RUN_DIR="${RUN_DIR:-outputs/training/fsa_lite_gemma12b_run3}"
lite_banner
lite_fsa_venv || exit 1
lite_cfg_data || exit 1
[ -f "$RUN_DIR/train_log.jsonl" ] || { echo "no train_log.jsonl in $RUN_DIR"; exit 1; }
echo "data sha256: $(cd "$BUILT" && sha256sum units.jsonl pools.json | cut -c1-16 | tr '\n' ' ')"
NAME="$(basename "$CONFIG_LITE" .yaml)"
uv run --group ml python src/fsa_lite/lite/train_signal_audit.py --config "$CONFIG_LITE" --run-dir "$RUN_DIR" \
    --out "outputs/diagnostics/train_signal_audit/$NAME.json"
