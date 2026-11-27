#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --job-name=fsa-likelihood
#SBATCH --output=scripts/slurm/logs/%x_%j.out
#
# Likelihood (Bayes) attacker on a finished run (see slurm/README.md). Submit from
# :
#   sbatch --export=ALL,RUN=outputs/runs/<run_id> scripts/slurm/fsa_likelihood.sbatch
#   sbatch --export=ALL,RUN=outputs/runs/<run_id>,SCORER=qwen2.5-7b-instruct scripts/slurm/fsa_likelihood.sbatch
#   sbatch --export=ALL,RUN=outputs/runs/<run_id>,EXTRA_ARGS="--limit-secrets 2 --no-cache" \
#       scripts/slurm/fsa_likelihood.sbatch
# Public-weights attacker on an adapter run (base model only; writes its own numbered file):
#   sbatch --export=ALL,RUN=outputs/runs/<run_id>,EXTRA_ARGS="--scorer-no-adapter --calibrate" \
#       scripts/slurm/fsa_likelihood.sbatch
# A paired inadvertent-leakage config (docs/inadvertent_leakage.md): its newest run, --calibrate,
# THREAT = grey_box or public_weights, in the FSA-lite job environment (scripts/lib/common_lite.sh):
#   INADV_CONFIG=inadv_gemma_rand20_fsa_lite_run3 THREAT=grey_box bash scripts/submit_slurm.sh scripts/slurm/fsa_likelihood.sbatch
# The paired report (aggregates only), once all runs are scored (seconds):
#   INADV_REPORT=gemma bash scripts/submit_slurm.sh scripts/slurm/fsa_likelihood.sbatch
# (per-job walltimes for SBATCH_EXTRA: docs/inadvertent_leakage.md)

set -euo pipefail

if [ -n "${INADV_REPORT:-}" ] || [ -n "${INADV_CONFIG:-}" ]; then
    CONFIG_LITE="${INADV_CONFIG:-report $INADV_REPORT}"   # shown by lite_banner
    # shellcheck disable=SC1091
    source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/lib/common_lite.sh"
    lite_banner
    lite_fsa_venv
fi
if [ -n "${INADV_REPORT:-}" ]; then   # the paired tables (numpy only, seconds): qwen | gemma | all
    uv run --group ml python -m scripts.inadvertent report --model "$INADV_REPORT"
    exit $?
fi
if [ -n "${INADV_CONFIG:-}" ]; then
    : "${THREAT:?set THREAT=grey_box or THREAT=public_weights with INADV_CONFIG}"
    uv run --group ml python -m scripts.inadvertent score "$INADV_CONFIG" --threat "$THREAT" --backend real --device cuda
    exit $?
fi
: "${RUN:?set RUN=outputs/runs/<run_id> via sbatch --export=ALL,RUN=... (or INADV_CONFIG + THREAT)}"

echo "host: $(hostname)"
echo "date: $(date)"
nvidia-smi
echo "git:  $(git rev-parse --short HEAD)"

cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"

# SCORER empty = the run's own model. EXTRA_ARGS holds further likelihood_attack options,
# split on spaces (e.g. "--limit-secrets 2 --no-cache"), so values cannot contain spaces.
# Splitting into an array keeps the quoting intact; empty EXTRA_ARGS adds nothing.
read -r -a extra <<< "${EXTRA_ARGS:-}"
echo "extra args: ${extra[*]:-(none)}"
uv run --group ml python -m scripts.likelihood_attack "$RUN" \
    --device cuda \
    ${SCORER:+--scorer-model "$SCORER"} \
    ${extra[@]+"${extra[@]}"}
