#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --job-name=fsa-baseline
#SBATCH --output=scripts/slurm/logs/%x_%j.out
#
# Baseline leakage on the Slurm cluster (see slurm/README.md). Submit from :
#   sbatch scripts/slurm/fsa_baseline.sbatch
#   sbatch --export=ALL,MODEL=gemma3-4b-it,METHOD=prompt_hiding,PRESET=full scripts/slurm/fsa_baseline.sbatch
# Optional (unset = not passed, so the commands above run exactly as before): ADAPTER (a
# trained adapter folder, with METHOD=fsa/b3/random_layer), SECRET_TYPE (digits | words |
# digits4), PROMPT_ID (a user prompt id), EXTRA_ARGS (more options, space-separated, no commas):
#   sbatch --export=ALL,MODEL=gemma3-12b-it,METHOD=fsa,PRESET=full,SECRET_TYPE=digits4,PROMPT_ID=lottery_4digit,ADAPTER=outputs/training/<run>/adapter scripts/slurm/fsa_baseline.sbatch
# ADAPTER_KEY instead of ADAPTER: paths.local.yaml adapters.<key>, whose sha256 must match the pin in
# ADAPTER_SPEC (default configs/eval/fsa_lite_run3.approach.yaml), checked before the model loads:
#   sbatch --export=ALL,MODEL=gemma3-12b-it,METHOD=fsa,PRESET=full,ADAPTER_KEY=gemma_fsa_lite_run3 scripts/slurm/fsa_baseline.sbatch
# A paired inadvertent-leakage config (docs/inadvertent_leakage.md) instead of the variables above:
# the FSA-lite job environment (scripts/lib/common_lite.sh: cluster.env, HF_HUB_CACHE, offline):
#   INADV_CONFIG=inadv_gemma_rand20_original bash scripts/submit_slurm.sh scripts/slurm/fsa_baseline.sbatch
# (per-job walltimes for SBATCH_EXTRA: docs/inadvertent_leakage.md)

set -euo pipefail

if [ -n "${INADV_CONFIG:-}" ]; then
    CONFIG_LITE="$INADV_CONFIG"   # shown by lite_banner
    # shellcheck disable=SC1091
    source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/lib/common_lite.sh"
    lite_banner
    lite_fsa_venv
    uv run --group ml python -m scripts.inadvertent generate "$INADV_CONFIG" --backend real --device cuda
    exit $?
fi

echo "host: $(hostname)"
echo "date: $(date)"
nvidia-smi
echo "git:  $(git rev-parse --short HEAD)"

cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"

optional=()
if [ -n "${ADAPTER:-}" ]; then optional+=(--adapter "$ADAPTER"); fi
if [ -n "${ADAPTER_KEY:-}" ]; then optional+=(--adapter-key "$ADAPTER_KEY"); fi
if [ -n "${ADAPTER_SPEC:-}" ]; then optional+=(--adapter-spec "$ADAPTER_SPEC"); fi
if [ -n "${SECRET_TYPE:-}" ]; then optional+=(--secret-type "$SECRET_TYPE"); fi
if [ -n "${PROMPT_ID:-}" ]; then optional+=(--prompt-id "$PROMPT_ID"); fi
read -r -a extra <<< "${EXTRA_ARGS:-}"

# Run as a module (-m): run as a file, Python would not find the src/fsa_lite/fsa_runtime/ and scripts/ packages.
uv run --group ml python -m scripts.baseline_leakage \
    --backend real \
    --device cuda \
    --base-model "${MODEL:-qwen2.5-7b-instruct}" \
    --method "${METHOD:-base}" \
    --preset "${PRESET:-quick}" \
    ${optional[@]+"${optional[@]}"} \
    ${extra[@]+"${extra[@]}"}
