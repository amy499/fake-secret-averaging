# Sourced by the job scripts (slurm/*.sbatch, slurm/*.pbs). Plain bash, no git binary: The PBS cluster
# compute nodes have none, so the commit is read from .git directly (as core/io.py git_commit).

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
