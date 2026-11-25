# common_lite_pbs.sh: sourced by the FSA-lite PBS jobs (lite_setup.pbs, lite_pilot.pbs,
# lite_train.pbs, ...). Never run Python on a login node: every job refuses to start without
# PBS_JOBID. Compute nodes may have no git binary: the commit is read from .git
# (scripts/lib/git_head.sh). Site paths come from cluster.env (scripts/lib/cluster_env.sh); caches
# live on $SCRATCH (home quotas are often small); the HF token file is never read (the model is
# loaded offline from the existing cache).

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/lib/cluster_env.sh"
SCR="${SCR:-${SCRATCH:-}}"                                           # checked in lite_pbs_env (after the job guard)
FSA_DIR="$REPO_ROOT"
V2="$FSA_DIR/src/fsa_lite/lite"
FSA_LITE_DATA="${FSA_LITE_DATA:-$SCR/fsa_lite_data}"                 # fetch_lite.sh raw/ + built/ (run1 / run2), built_run3/
HARNESS_DATA="${HARNESS_DATA:-$REPO_ROOT/eval/.cache/data}"          # the harness's verified eval files (FETCH)
MODEL_CACHE="${HF_HUB_CACHE:-}"                                      # models--Qwen--Qwen2.5-7B-Instruct, ...; checked in lite_pbs_env
CACHE="${CACHE:-$SCR/fsa_lite_cache}"                                # uv / pip / torch / triton caches
UV_DIR="${UV_DIR:-$SCR/tools/uv}"                                    # uv binary (installed by lite_setup.pbs)
UV="$UV_DIR/uv"
PY="$FSA_DIR/.venv/bin/python"                                       # built by lite_setup.pbs
CONFIG_LITE="${CONFIG_LITE:-configs/training/experiments/fsa_lite_qwen7b.yaml}"
ADAPTER_OUT="${ADAPTER_OUT:-$SCR/adapters/qwen_fsa_lite}"            # where the finished adapter is copied (paths.local.yaml adapters.<key>)
# run3: the harness's held-out secrets (read by build_lite_run3.py / heldout.py only; never printed)
HELD_OUT="${HELD_OUT:-$REPO_ROOT/configs/eval/private/held_out_secrets.txt}"
umask 077

lite_pbs_require_job() {
    if [ -z "${PBS_JOBID:-}" ]; then
        echo "refusing to run outside a PBS job (login node: no Python). Use qsub." >&2
        exit 2
    fi
}

lite_pbs_env() {  # caches on scratch, offline model loading, the FSA venv's python
    [ -n "$SCR" ] || { echo "set SCRATCH in cluster.env"; return 1; }
    [ -n "$MODEL_CACHE" ] || { echo "set HF_HUB_CACHE in cluster.env"; return 1; }
    mkdir -p "$CACHE"/{uv,uv-python,pip,xdg,triton,torch,hf_home,mpl}
    export UV_CACHE_DIR="$CACHE/uv" UV_PYTHON_INSTALL_DIR="$CACHE/uv-python" PIP_CACHE_DIR="$CACHE/pip"
    export XDG_CACHE_HOME="$CACHE/xdg" TRITON_CACHE_DIR="$CACHE/triton" TORCH_HOME="$CACHE/torch"
    export MPLCONFIGDIR="$CACHE/mpl" HF_HOME="$CACHE/hf_home"
    # backend.settings reads FSA_HF_HOME as the model cache (cache_dir of from_pretrained; refs/main is
    # checked against the config's revision there)
    export FSA_HF_HOME="$MODEL_CACHE" HF_HUB_CACHE="$MODEL_CACHE" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
    unset VIRTUAL_ENV HF_TOKEN
    cd "$FSA_DIR" || return 1
}

lite_pbs_banner() {
    # shellcheck disable=SC1091
    source "$FSA_DIR/scripts/lib/git_head.sh"
    echo "=== $(date -Is) host $(hostname) job ${PBS_JOBID:-none} ==="
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "gpu: none"
    echo "repo $REPO_ROOT commit $(_git_head "$REPO_ROOT" 2>/dev/null || echo unknown); config $CONFIG_LITE"
    echo "data $FSA_LITE_DATA; harness data $HARNESS_DATA; model cache $MODEL_CACHE"
}

lite_pbs_require_venv() {
    [ -x "$PY" ] || { echo "no $PY: run lite_setup.pbs first"; return 1; }
    [ -f "$FSA_DIR/.venv/SETUP_OK" ] || { echo "$FSA_DIR/.venv has no SETUP_OK: lite_setup.pbs did not finish"; return 1; }
}

lite_pbs_require_inputs() {
    for f in tt/raw_dump_defenses.jsonl.bz2 tt/raw_dump_attacks.jsonl.bz2 cimem/cimemories_raw.json; do
        [ -f "$FSA_LITE_DATA/raw/$f" ] || { echo "missing $FSA_LITE_DATA/raw/$f: on the login node run FSA_LITE_DATA=$FSA_LITE_DATA bash $V2/fetch_lite.sh"; return 1; }
    done
    for f in password_eval_test confaide_tier4 mmlu_test ifeval_input tt_extraction tt_hijacking; do
        [ -d "$HARNESS_DATA/url/protocol/$f" ] || { echo "missing $HARNESS_DATA/url/protocol/$f: run the harness FETCH job"; return 1; }
    done
    # the config's data.dir (a subfolder of FSA_LITE_DATA) is what this job builds and checks
    lite_cfg_data || return 1
    rev="$(sed -n 's/^  revision: *\([0-9a-f]\{40\}\).*/\1/p' "$FSA_DIR/$CONFIG_LITE" | head -1)"
    hf="$(sed -n 's/^  hf_id: *\([^ #]*\).*/\1/p' "$FSA_DIR/$CONFIG_LITE" | head -1)"
    ref="$MODEL_CACHE/models--${hf//\//--}/refs/main"
    [ -n "$rev" ] && [ -f "$ref" ] && [ "$(cat "$ref")" = "$rev" ] && [ -d "$MODEL_CACHE/models--${hf//\//--}/snapshots/$rev" ] \
        || { echo "model cache: $ref must name the pinned revision '$rev' and its snapshot must exist"; return 1; }
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

lite_pbs_build_args() {  # run3's builder needs the held-out file (BUILD_ARGS; empty for build_lite.py)
    BUILD_ARGS=()
    if [ "$BUILDER" = build_lite_run3.py ]; then
        [ -f "$HELD_OUT" ] || { echo "run3 needs the harness held-out file: HELD_OUT=$HELD_OUT is missing"; return 1; }
        BUILD_ARGS=(--exclude-secrets "$HELD_OUT")
    fi
}

lite_pbs_heldout_check() {  # run3 only: the built data shares no value with the held-out secrets
    [ "$BUILDER" = build_lite_run3.py ] || return 0
    "$PY" "$V2/heldout.py" check --data "$BUILT" --exclude-secrets "$HELD_OUT"
}

lite_pbs_build_data() {  # build once (CPU, minutes); always re-run the contamination check
    lite_pbs_build_args || return 1
    if [ ! -f "$BUILT/manifest.json" ]; then
        "$PY" "$V2/$BUILDER" --raw "$FSA_LITE_DATA/raw" --harness-data "$HARNESS_DATA" \
            --out "$BUILT" ${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"} || return 1
    fi
    "$PY" "$V2/contam_check.py" --data "$BUILT" --harness-data "$HARNESS_DATA" || return 1
    lite_pbs_heldout_check || return 1
    echo "data units sha256: $(sed -n 's/.*"units_sha256": "\([0-9a-f]*\)".*/\1/p' "$BUILT/manifest.json" | head -1)"
}
