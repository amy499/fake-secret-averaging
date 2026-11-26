#!/bin/bash
# Submit a PBS job script with the site settings from cluster.env (project, queue), from the
# repository root, log in scripts/pbs/logs/. Job variables and resources go in QSUB_EXTRA
# (qsub splits the variable list on commas, so no value may contain one):
#   QSUB_EXTRA="-l walltime=01:00:00 -v RUN_DIR=outputs/training/fsa_lite_qwen7b_run3" bash scripts/submit_pbs.sh scripts/pbs/lite_train.pbs
# QSUB_EXTRA is split on spaces, so a value WITH spaces goes in the environment and only its name in -v
# (qsub then takes the value from the environment):
#   RUN_ARGS="--approach fsa --model qwen --variant original" QSUB_EXTRA="-l walltime=01:00:00 -v RUN_ARGS" bash scripts/submit_pbs.sh scripts/pbs/harness_run.pbs
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source scripts/lib/cluster_env.sh
job="${1:?usage: bash scripts/submit_pbs.sh <scripts/pbs/job.pbs>}"
mkdir -p scripts/pbs/logs
log="$PWD/scripts/pbs/logs/$(basename "$job" .pbs)_$(date +%Y%m%d_%H%M%S).log"
# shellcheck disable=SC2086  (QSUB_EXTRA is a word list on purpose)
exec qsub ${PBS_PROJECT:+-P "$PBS_PROJECT"} -q "${PBS_QUEUE:?set PBS_QUEUE in cluster.env}" \
    ${QSUB_EXTRA:-} -o "$log" "$job"
