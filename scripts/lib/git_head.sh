# git-free HEAD reader: The PBS cluster compute nodes have no git binary
_git_head() { local g="$1/.git" h r; h=$(cat "$g/HEAD"); if [[ $h == ref:* ]]; then r=${h#ref: }; if [ -f "$g/$r" ]; then cat "$g/$r"; else awk -v r="$r" '$2==r{print $1}' "$g/packed-refs"; fi; else echo "$h"; fi; }
