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

set -euo pipefail
: "${RUN:?set RUN=outputs/runs/<run_id> via sbatch --export=ALL,RUN=...}"

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
