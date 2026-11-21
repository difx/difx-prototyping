#!/bin/bash
#
# make-copy-script.sh <file>... - for the OzSTAR jobs (fusebench-ozstar.sbatch, ncu-ozstar.sbatch):
# writes copy-results.<jobid>.sh, which copies the job's SLURM log and the given files (its packed
# results) to RESULTS_DEST with one scp, and prints how to run it.  Run that script after the job
# has finished (the SLURM log is complete then).
#
# RESULTS_DEST: where to copy to (scp syntax), default
#     ar313:packages/claude/src/difx-prototyping/fft-fusion/results-local/
# SLURM_LOG: the job's SLURM log (the sbatch script sets it from its --output pattern).

set -u
dest="${RESULTS_DEST:-ar313:packages/claude/src/difx-prototyping/fft-fusion/results-local/}"
job="${SLURM_JOB_ID:-manual}"
files=()
[ -n "${SLURM_LOG:-}" ] && files+=("$(realpath -m "$SLURM_LOG")")
for f in "$@"; do [ -e "$f" ] && files+=("$(realpath "$f")"); done
[ ${#files[@]} -gt 0 ] || { echo "make-copy-script.sh: nothing to copy"; exit 0; }
script="$(realpath -m "copy-results.$job.sh")"
{
    echo '#!/bin/bash'
    echo "# Copy job $job's results ($(hostname -s), $(date '+%Y-%m-%d %H:%M')) to $dest"
    printf 'exec scp'
    printf ' %q' "${files[@]}" "$dest"
    printf '\n'
} > "$script"
chmod +x "$script"
echo
echo "To copy the results to ${dest%%:*} once the job has finished:"
echo "    $script"
echo "which runs:"
tail -1 "$script" | sed 's/^exec /    /'
