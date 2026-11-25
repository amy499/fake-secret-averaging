# cluster_env.sh: sourced by every job script and by scripts/submit_{slurm,pbs}.sh. Reads the site
# settings from cluster.env at the repository root (git-ignored; copy cluster.env.example) and
# exports them. No job script carries a cluster path, user, partition, QoS, project or queue.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [ -f "$REPO_ROOT/cluster.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$REPO_ROOT/cluster.env"
    set +a
else
    echo "cluster_env.sh: $REPO_ROOT/cluster.env not found: copy cluster.env.example and fill it in" >&2
fi
export REPO_ROOT
export HARNESS_CLUSTER="${HARNESS_CLUSTER:-${CLUSTER_NAME:-local}}"   # recorded in every run's summary.json
