#!/bin/bash
# Submit a Slurm job script with the site settings from cluster.env (partition, QoS, account),
# from the repository root, logs in scripts/slurm/logs/. Variables for the job go in the
# environment (sbatch exports it) or in SBATCH_EXTRA (no commas inside one --export value):
#   RUN_DIR=outputs/training/fsa_lite_gemma12b_run3 CONFIG_LITE=configs/training/experiments/fsa_lite_gemma12b_run3.yaml \
#     bash scripts/submit_slurm.sh scripts/slurm/lite_train.sbatch
#   SBATCH_EXTRA="--time=01:00:00" bash scripts/submit_slurm.sh scripts/slurm/harness_run.sbatch --approach fsa_lite_run3 --model gemma --variant technique
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source scripts/lib/cluster_env.sh
job="${1:?usage: bash scripts/submit_slurm.sh <scripts/slurm/job.sbatch> [job arguments]}"
shift
mkdir -p scripts/slurm/logs
# shellcheck disable=SC2086  (SBATCH_EXTRA is a word list on purpose)
exec sbatch --partition="${SLURM_PARTITION:?set SLURM_PARTITION in cluster.env}" \
    ${SLURM_QOS:+--qos="$SLURM_QOS"} ${SLURM_ACCOUNT:+--account="$SLURM_ACCOUNT"} \
    ${SBATCH_EXTRA:-} "$job" "$@"
