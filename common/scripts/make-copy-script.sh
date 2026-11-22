#!/bin/bash
#
# make-copy-script.sh <file>... - for the OzSTAR jobs (common/scripts/ozstar.sbatch,
# ncu-ozstar.sbatch), run from the repository root: writes logfiles/copy-results.<jobid>.sh, which
# copies the job's SLURM log and the given files (its build log and packed results, paths from the
# root) to the
# same paths under RESULTS_DEST with one rsync (one login), and prints how to run it.  Run that
# script after the job has finished (the SLURM log is complete then).
#
# RESULTS_DEST: the checkout to copy to (rsync/scp syntax), default
#     ar313:packages/claude/src/difx-prototyping/
# so that e.g. xmac/results-local/<run>.tgz lands in xmac/results-local/ there, and the SLURM log
# (logfiles/ozstar.<jobid>.log) in logfiles/.
# SLURM_LOG: the job's SLURM log, from the root (the sbatch script sets it).

set -u
dest="${RESULTS_DEST:-ar313:packages/claude/src/difx-prototyping/}"
case "$dest" in */) ;; *) dest="$dest/" ;; esac
root=$(pwd)
job="${SLURM_JOB_ID:-manual}"
files=()
[ -n "${SLURM_LOG:-}" ] && files+=("$SLURM_LOG")
for f in "$@"; do [ -e "$f" ] && files+=("$f"); done
[ ${#files[@]} -gt 0 ] || { echo "make-copy-script.sh: nothing to copy"; exit 0; }
mkdir -p "$root/logfiles"
script="$root/logfiles/copy-results.$job.sh"
{
    echo '#!/bin/bash'
    echo "# Copy job $job's results ($(hostname -s), $(date '+%Y-%m-%d %H:%M')) to $dest"
    printf 'cd %q || exit 1\n' "$root"
    printf 'exec rsync -aR --progress'
    printf ' %q' "${files[@]}" "$dest"
    printf '\n'
} > "$script"
chmod +x "$script"
echo
echo "To copy the results to ${dest%%:*} once the job has finished:"
echo "    $script"
echo "which runs (from $root):"
tail -1 "$script" | sed 's/^exec /    /'
