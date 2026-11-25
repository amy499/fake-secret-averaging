# Environment for the harness. Source it by hand:   source env.sh
# (Never add this to .bashrc or any shell startup file.)

HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export HARNESS_ROOT
export PYTHONPATH="$HARNESS_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Downloaded benchmark data (override before sourcing to use scratch space).
export HARNESS_DATA="${HARNESS_DATA:-$HARNESS_ROOT/.cache/data}"

# Hugging Face model cache. Reuse an existing snapshot rather than downloading again (Gemma 3 12B
# is about 24 GB of bf16 safetensors). First match wins:
#   1. HARNESS_HF_HUB_CACHE=/path/to/hub (the folder that holds models--google--gemma-3-12b-it),
#      e.g. sbatch --export=ALL,HARNESS_HF_HUB_CACHE=/path/to/hub ... -> becomes HF_HUB_CACHE;
#   2. HF_HUB_CACHE or HF_HOME already set (passed with --export / -v): left as they are;
#   3. an existing ~/.cache/huggingface/hub: the huggingface_hub default, left as it is;
#   4. otherwise a new cache inside the repo: $HARNESS_ROOT/.cache/hf (git-ignored).
if [ -n "${HARNESS_HF_HUB_CACHE:-}" ]; then
    export HF_HUB_CACHE="$HARNESS_HF_HUB_CACHE"
elif [ -z "${HF_HUB_CACHE:-}" ] && [ -z "${HF_HOME:-}" ] && [ ! -d "$HOME/.cache/huggingface/hub" ]; then
    export HF_HOME="$HARNESS_ROOT/.cache/hf"
fi

# Gemma 3 is gated: accept the licence on huggingface.co, then put a token in
# ~/.cache/huggingface/token (huggingface-cli login) or export HF_TOKEN yourself. When HF_HOME
# points elsewhere, huggingface_hub would not look there, so HF_TOKEN_PATH names that file (the
# token itself is never read or printed here).
if [ -z "${HF_TOKEN:-}" ] && [ -z "${HF_TOKEN_PATH:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"
fi

export TOKENIZERS_PARALLELISM=false
export PYTHONUTF8=1

# Activate the project virtualenv if it exists (python -m venv .venv && pip install -r requirements.txt).
if [ -f "$HARNESS_ROOT/.venv/bin/activate" ]; then
    . "$HARNESS_ROOT/.venv/bin/activate"
elif [ -f "$HARNESS_ROOT/.venv/Scripts/activate" ]; then
    . "$HARNESS_ROOT/.venv/Scripts/activate"
fi
