# Sourced by the job scripts (slurm/*.sbatch, slurm/*.pbs). Plain bash, no git binary: The PBS cluster
# compute nodes have none, so the commit is read from .git directly (as core/io.py git_commit).

# model_cache_report [args...]: one line saying whether $HF_HUB_CACHE holds the model of the job's
# --model (gemma | qwen; the last --model wins, as in argparse), or both models when no --model is
# given (setup / data / stats jobs). Words only: it never prints a value from the data.
model_cache_report() {
    local model="" prev="" a m k dir name out=""
    for a in "$@"; do
        case "$a" in --model=*) model="${a#--model=}" ;; esac
        [ "$prev" = "--model" ] && model="$a"
        prev="$a"
    done
    case "$model" in
        gemma | qwen) k="$model" ;;
        *) k="gemma qwen" ;;
    esac
    for m in $k; do
        case "$m" in
            gemma) dir="models--google--gemma-3-12b-it"; name="gemma-3-12b-it" ;;
            qwen) dir="models--Qwen--Qwen2.5-7B-Instruct"; name="Qwen2.5-7B-Instruct" ;;
        esac
        out="${out:+$out, }$name cached: $([ -d "${HF_HUB_CACHE:-/nonexistent}/$dir" ] && echo yes || echo no)"
    done
    echo "HF_HUB_CACHE=${HF_HUB_CACHE:-unset} ($out)"
}

# head_commit: the checked-out commit (HEAD -> loose ref, or packed-refs), "none" if unknown.
# Works for a normal clone (.git/ folder) and a worktree (.git file "gitdir: ...", commondir).
head_commit() {
    local g=.git c head ref sha
    if [ -f .git ]; then
        g="$(sed -n 's/^gitdir: *//p' .git)"
    fi
    if [ ! -f "$g/HEAD" ]; then
        echo none
        return 0
    fi
    c="$g"
    if [ -f "$g/commondir" ]; then
        c="$(cat "$g/commondir")"
        case "$c" in /*) ;; *) c="$g/$c" ;; esac
    fi
    head="$(cat "$g/HEAD")"
    case "$head" in
        "ref: "*)
            ref="${head#ref: }"
            if [ -f "$g/$ref" ]; then
                cat "$g/$ref"
            elif [ -f "$c/$ref" ]; then
                cat "$c/$ref"
            else
                sha="$(awk -v r="$ref" '$2 == r { print $1; exit }' "$c/packed-refs" 2>/dev/null || true)"
                echo "${sha:-none}"
            fi
            ;;
        *) echo "$head" ;;
    esac
}
