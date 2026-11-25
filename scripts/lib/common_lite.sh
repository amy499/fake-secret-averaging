# common_lite.sh: sourced by the FSA-lite Slurm jobs (never run Python on a login node).
# Site paths come from cluster.env (scripts/lib/cluster_env.sh); each is overridable from the environment. The config's data.dir must be a subfolder of
# FSA_LITE_DATA (default ~/fsa_lite_data): built/ for run1 / run2, built_run3/ for run3; its
# data.builder (default build_lite.py) builds it when missing (lite_cfg_data).

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/lib/cluster_env.sh"
FSA_DIR="$REPO_ROOT"
V2="$FSA_DIR/src/fsa_lite/lite"
FSA_LITE_DATA="${FSA_LITE_DATA:-$HOME/fsa_lite_data}"
HARNESS_DATA="${HARNESS_DATA:-$REPO_ROOT/eval/.cache/data}"
CONFIG_LITE="${CONFIG_LITE:-configs/training/experiments/fsa_lite_gemma12b.yaml}"
# run3: the harness's held-out secrets (secret-in-context suite), read by build_lite_run3.py and
# heldout.py only (never printed, never copied)
HELD_OUT="${HELD_OUT:-$REPO_ROOT/configs/eval/private/held_out_secrets.txt}"
umask 077

lite_banner() {
    echo "=== $(date -Is) host $(hostname) job ${SLURM_JOB_ID:-local} ==="
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "gpu: none"
    echo "FSA repo $(git -C "$FSA_DIR" rev-parse HEAD 2>/dev/null); config $CONFIG_LITE"
    echo "data $FSA_LITE_DATA; harness data $HARNESS_DATA"
}

lite_fsa_venv() {  # the FSA uv venv (torch, transformers, peft, pyarrow)
    export PATH="$HOME/.local/bin:$PATH"
    unset VIRTUAL_ENV
    export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
    cd "$FSA_DIR" || return 1
}

lite_require_raw() {
    for f in tt/raw_dump_defenses.jsonl.bz2 tt/raw_dump_attacks.jsonl.bz2 cimem/cimemories_raw.json; do
        [ -f "$FSA_LITE_DATA/raw/$f" ] || { echo "missing $FSA_LITE_DATA/raw/$f: run bash scripts/fetch_lite.sh on the login node"; return 1; }
    done
    for f in password_eval_test confaide_tier4 mmlu_test ifeval_input tt_extraction tt_hijacking; do
        [ -d "$HARNESS_DATA/url/protocol/$f" ] || { echo "missing $HARNESS_DATA/url/protocol/$f: run the harness FETCH step"; return 1; }
    done
}

lite_cfg_data() {  # BUILT = the config's data.dir (~ expanded), BUILDER = its data.builder
    # (run1 / run2 configs name no builder: build_lite.py, and their data.dir is $FSA_LITE_DATA/built)
    local d b
    d="$(sed -n 's/^  dir: *\([^ #]*\).*/\1/p' "$FSA_DIR/$CONFIG_LITE" | head -1)"
    b="$(sed -n 's/^  builder: *\([^ #]*\).*/\1/p' "$FSA_DIR/$CONFIG_LITE" | head -1)"
    BUILT="${d/#\~/$HOME}"
    BUILT="${BUILT//\$SCRATCH/${SCRATCH:-}}"   # data.dir may use $SCRATCH / $HOME (cluster.env)
    BUILT="${BUILT//\$HOME/$HOME}"
    BUILDER="${b:-build_lite.py}"
    [ -n "$BUILT" ] || { echo "$CONFIG_LITE has no data.dir"; return 1; }
    case "$BUILDER" in build_lite.py|build_lite_run3.py) ;; *) echo "$CONFIG_LITE: unknown data.builder $BUILDER"; return 1 ;; esac
    [ "$(dirname "$BUILT")" = "$FSA_LITE_DATA" ] || { echo "$CONFIG_LITE data.dir $BUILT is not in FSA_LITE_DATA=$FSA_LITE_DATA"; return 1; }
    echo "data: $BUILT (builder $BUILDER)"
}

lite_build_args() {  # run3's builder needs the held-out file (BUILD_ARGS; empty for build_lite.py)
    BUILD_ARGS=()
    if [ "$BUILDER" = build_lite_run3.py ]; then
        [ -f "$HELD_OUT" ] || { echo "run3 needs the harness held-out file: HELD_OUT=$HELD_OUT is missing"; return 1; }
        BUILD_ARGS=(--exclude-secrets "$HELD_OUT")
    fi
}

lite_heldout_check() {  # run3 only: the built data shares no value with the held-out secrets
    [ "$BUILDER" = build_lite_run3.py ] || return 0
    uv run --group ml python src/fsa_lite/lite/heldout.py check --data "$BUILT" --exclude-secrets "$HELD_OUT"
}

lite_build_data() {  # build once (CPU, a few minutes); always re-run the contamination check
    lite_cfg_data || return 1
    lite_build_args || return 1
    if [ ! -f "$BUILT/manifest.json" ]; then
        uv run --group ml python "src/fsa_lite/lite/$BUILDER" --raw "$FSA_LITE_DATA/raw" \
            --harness-data "$HARNESS_DATA" --out "$BUILT" ${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"} || return 1
    fi
    uv run --group ml python src/fsa_lite/lite/contam_check.py --data "$BUILT" \
        --harness-data "$HARNESS_DATA" || return 1
    lite_heldout_check || return 1
}
