#!/bin/bash
#
#SBATCH --job-name=fusebench
#SBATCH --output=fusebench.%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8         # for the parallel build (the fused kernel is compiled per FFT length)
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --tmp=8G                  # job-local disk ($JOBFS) for nvcc's intermediate files
#SBATCH --time=0:45:00            # estimate: build 5-10 min, run-bench.sh + run-nchan.sh 3-6 min (5 on an RTX 2070)
#SBATCH --account=oz168
## add a partition if your site needs one:  #SBATCH --partition=<partition>
#
# fusebench on one OzSTAR GPU node (an A100 on the gina nodes): builds fusebench for the GPU the
# job got, then runs the standard set (run-bench.sh) and the channel-count sweep (run-nchan.sh).
# See README.md.
#
# The build environment comes from DIFX_SETUP, a file this job sources: your gpudifx setup (module
# loads, then gpudifx's setup.bash), which sets CUDAROOT or puts nvcc on PATH.  Default:
# $HOME/setup_difx.claude.  The GPU architecture is the job's GPU, whatever NVCCFLAGS says.
#
# Before submitting, on a login node, in a clone of difx-prototyping
# (git clone https://github.com/difx/difx-prototyping.git):
#     cd difx-prototyping/fft-fusion
#     make mathdx           # fetches MathDx (cuFFTDx) from PyPI; the GPU nodes may have no network
#     sbatch fusebench-ozstar.sbatch
# Submit from fft-fusion (or set BENCHDIR).  The SLURM log fusebench.<jobid>.log is written there
# and holds the summary.  The logs of every run are in results-local/<host>-<stamp>/ (run-bench.sh)
# and results-local/<host>-nchan-<stamp>/ (run-nchan.sh), each also packed into a .tgz next to it
# at the end: send those and the SLURM log back.
#
# Other settings, through the environment:
#     sbatch --export=ALL,DIFX_SETUP=<file> fusebench-ozstar.sbatch
#     sbatch --export=ALL,MATHDX=<MathDx dir> fusebench-ozstar.sbatch    # a MathDx you already have
#     sbatch --export=ALL,EXTRA="--reps 50" fusebench-ozstar.sbatch      # extra fusebench options

DIFX_SETUP="${DIFX_SETUP:-$HOME/setup_difx.claude}"
BENCHDIR="${BENCHDIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$DIFX_SETUP" ] || { echo "ERROR: DIFX_SETUP $DIFX_SETUP does not exist" >&2; exit 1; }
[ -x "$BENCHDIR/run-bench.sh" ] || { echo "ERROR: BENCHDIR $BENCHDIR has no run-bench.sh: submit from fft-fusion or set BENCHDIR" >&2; exit 1; }
cd "$BENCHDIR" || exit 1
if [ -n "${MATHDX:-}" ]; then
    [ -f "$MATHDX/include/cufftdx.hpp" ] || { echo "ERROR: MATHDX $MATHDX has no include/cufftdx.hpp" >&2; exit 1; }
    mathdx=(MATHDX="$MATHDX")
else
    [ -f mathdx-25.6.0/.unpacked ] || { echo "ERROR: no MathDx in $BENCHDIR: run 'make mathdx' there on a login node first" >&2; exit 1; }
    mathdx=()
fi

# shellcheck source=/dev/null
. "$DIFX_SETUP" || { echo "ERROR: sourcing $DIFX_SETUP failed" >&2; exit 1; }
command -v nvcc > /dev/null || [ -x "${CUDAROOT:-}/bin/nvcc" ] \
    || { echo "ERROR: after $DIFX_SETUP, no nvcc on PATH and none in CUDAROOT" >&2; exit 1; }
read -r -a extra <<< "${EXTRA:-}"

SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
[ -n "$SM" ] || { echo "ERROR: no GPU visible to nvidia-smi" >&2; exit 1; }
echo "Node $(hostname), job ${SLURM_JOB_ID:-manual}, difx-prototyping $(git log -1 --format='%h %s')"
echo "Environment from $DIFX_SETUP"
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,compute_cap --format=csv,noheader
echo "Building for sm_$SM ..."
t0=$(date +%s)
# nvcc writes ~360 MB of intermediate files to $TMPDIR for each fused FFT length, eight at
# once with -j8: more than a node's /tmp may have free.  Use the job's own disk (OzSTAR: $JOBFS,
# from --tmp above) when there is one.
[ -n "${JOBFS:-}" ] && export TMPDIR="$JOBFS"
tmpfree=$(df -Pk "${TMPDIR:-/tmp}" | awk 'NR==2 {print int($4/1048576)}')
echo "Temporary files in ${TMPDIR:-/tmp} (${tmpfree:-?} GB free)"
[ "${tmpfree:-0}" -ge 3 ] || echo "WARNING: the build may need ~3 GB of temporary space: set TMPDIR, or build on a login node first"
# Jobs started together (fusebench-ozstar.sbatch and ncu-ozstar.sbatch) build in the same
# build-sm<SM> directory, so they take turns (flock); each keeps its own build log.  Where the
# filesystem does not support locks, warn and build without one.
blog="build-sm$SM.${SLURM_JOB_ID:-manual}.log"
lock=(flock "build-sm$SM.lock")
flock -n -E 200 "build-sm$SM.lock" true 2>/dev/null
rc=$?
if [ $rc != 0 ] && [ $rc != 200 ]; then
    echo "WARNING: file locks do not work in $BENCHDIR; building without one, so do not start two of these jobs at once"
    lock=()
fi
"${lock[@]}" make -j"${SLURM_CPUS_PER_TASK:-8}" SM="$SM" "${mathdx[@]}" > "$blog" 2>&1 \
    || { echo "ERROR: build failed, see $BENCHDIR/$blog" >&2; tail -20 "$blog" >&2; exit 1; }
head -1 "$blog"                   # what the Makefile chose: CUDA, MathDx
"${CUDAROOT:+$CUDAROOT/bin/}nvcc" --version | tail -1   # the Makefile's choice: CUDAROOT, else PATH
echo "Built in $(( $(date +%s) - t0 )) s"

# run <script>: run it, then pack the results directory it reports ("Results in <dir> ...")
rc=0
run() {
    local t0=$(date +%s) log="$1.${SLURM_JOB_ID:-manual}.out"
    "./$1" "${extra[@]}" 2>&1 | tee "$log"
    local r=${PIPESTATUS[0]}
    echo "$1: exit $r, $(( $(date +%s) - t0 )) s"
    [ $r = 0 ] || rc=1
    local results=$(sed -n 's/^Results in \([^ ]*\).*/\1/p' "$log" | tail -1)
    if [ -n "$results" ] && [ -d "$results" ]; then
        command mv "$log" "$results/"
        tar czf "$results.tgz" -C results-local "$(basename "$results")"
        echo "Packed: $BENCHDIR/$results.tgz"
    fi
}
run run-bench.sh
run run-nchan.sh
exit $rc
