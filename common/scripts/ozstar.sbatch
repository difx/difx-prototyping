#!/bin/bash
#
#SBATCH --job-name=difx-proto
#SBATCH --output=logfiles/ozstar.%j.log    # from the directory you submit in: the root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8         # for the parallel build (the fused kernel is compiled per FFT length)
#SBATCH --gres=gpu:1
#SBATCH --mem=32G                 # the build: up to ~4 GB per ptxas (N=8192 and up), eight at once
#SBATCH --tmp=8G                  # job-local disk ($JOBFS) for nvcc's intermediate files
#SBATCH --time=1:30:00            # estimate: build ~6 min on a gina node (370 s, job 21387208), the default scripts ~20 min
#SBATCH --account=oz168
## add a partition if your site needs one:  #SBATCH --partition=<partition>
#
# The benchmarks on one OzSTAR GPU node (an A100 on most gina nodes): builds them for the GPU the
# job got (the Makefile in the repository root), then runs the sub-projects' run-*.sh scripts named
# in SCRIPTS (paths from the root), each in its own directory.  Default: fft-fusion's standard set
# (run-bench.sh) and channel-count sweep (run-nchan.sh), and xmac's station-count sweeps
# (run-xmac.sh, run-staged.sh).  See README.md (OzSTAR).
#
# The build environment comes from DIFX_SETUP, a file this job sources: your gpudifx setup (module
# loads, then gpudifx's setup.bash), which sets CUDAROOT or puts nvcc on PATH.  Default:
# $HOME/setup_difx.claude.  The GPU architecture is the job's GPU, whatever NVCCFLAGS says.
#
# Before submitting, on a login node, in a clone of difx-prototyping
# (git clone https://github.com/difx/difx-prototyping.git):
#     cd difx-prototyping
#     make mathdx           # fetches MathDx (cuFFTDx) from PyPI; the GPU nodes may have no network
#     sbatch common/scripts/ozstar.sbatch
# Submit from the checkout's root: the SLURM log, logfiles/ozstar.<jobid>.log, holds the summary, and
# SLURM writes it relative to where you submit (it does not create logfiles/; the checkout has it).
# The build log is logfiles/build-sm<SM>.<jobid>.log.  Each script's results go
# to its sub-project's results-local/<host>-...-<stamp>/, packed into a .tgz next to it at the end.
# The job then writes logfiles/copy-results.<jobid>.sh (make-copy-script.sh): run it once the job
# has finished to copy the SLURM log, the build log and the .tgz files to the same places in RESULTS_DEST
# (default ar313's checkout; --export=ALL,RESULTS_DEST=<host>:<checkout>/).
#
# Other settings, through the environment:
#     sbatch --export=ALL,DIFX_SETUP=<file> common/scripts/ozstar.sbatch
#     sbatch --export=ALL,MATHDX=<MathDx dir> ...           # a MathDx you already have
#     sbatch --export=ALL,EXTRA="--reps 50" ...             # extra options for every script
#     sbatch --export=ALL,TARGETS="xmacbench" ...           # build only these (default: all; fusebench
#                                                           # is most of it)
#     sbatch --export=ALL,SCRIPTS="xmac/run-xmac.sh xmac/run-staged.sh" ...   # only some scripts
#     SCRIPTS=fft-fusion/run-wide.sh      the wider shapes (bits, bands)
#     SCRIPTS=fft-fusion/run-matrix.sh    the big matrix (bits x subbands x channels x stations, 108
#                                         shapes, 15 min on an A100), with a summary CSV in the .tgz
#     SCRIPTS=two-stage/run-twostage.sh   two-stage channelisation, ~64k channels; add
#                                         NCHAN=8192,NWIN=128 for ~8k

DIFX_SETUP="${DIFX_SETUP:-$HOME/setup_difx.claude}"
BENCHDIR="${BENCHDIR:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null)}"
[ -f "$DIFX_SETUP" ] || { echo "ERROR: DIFX_SETUP $DIFX_SETUP does not exist" >&2; exit 1; }
[ -f "$BENCHDIR/Makefile" ] && [ -d "$BENCHDIR/common" ] || { echo "ERROR: BENCHDIR ${BENCHDIR:-?} is not a difx-prototyping checkout: submit from one or set BENCHDIR" >&2; exit 1; }
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
# Jobs started together (ozstar.sbatch and ncu-ozstar.sbatch) build in the same
# build-sm<SM> directory, so they take turns (flock); each keeps its own build log (logfiles/).  Where the
# filesystem does not support locks, warn and build without one.
mkdir -p logfiles
blog="logfiles/build-sm$SM.${SLURM_JOB_ID:-manual}.log"
lock=(flock "logfiles/build-sm$SM.lock")
flock -n -E 200 "logfiles/build-sm$SM.lock" true 2>/dev/null
rc=$?
if [ $rc != 0 ] && [ $rc != 200 ]; then
    echo "WARNING: file locks do not work in $BENCHDIR; building without one, so do not start two of these jobs at once"
    lock=()
fi
read -r -a targets <<< "${TARGETS:-all}"
"${lock[@]}" make -j"${SLURM_CPUS_PER_TASK:-8}" SM="$SM" "${mathdx[@]}" "${targets[@]}" > "$blog" 2>&1 \
    || { echo "ERROR: build failed, see $BENCHDIR/$blog" >&2; tail -20 "$blog" >&2; exit 1; }
head -1 "$blog"                   # what the Makefile chose: CUDA, MathDx
"${CUDAROOT:+$CUDAROOT/bin/}nvcc" --version | tail -1   # the Makefile's choice: CUDAROOT, else PATH
echo "Built in $(( $(date +%s) - t0 )) s"

# run <sub-project>/<script>: run it in its directory, then pack the results directory it reports
# ("Results in <dir> ...", from that directory)
rc=0
packed=()
run() {
    local t0=$(date +%s) dir=$(dirname "$1") name=$(basename "$1")
    local log="$BENCHDIR/$dir/$name.${SLURM_JOB_ID:-manual}.out"
    [ -x "$BENCHDIR/$1" ] || { echo "ERROR: no script $1"; rc=1; return; }
    (cd "$BENCHDIR/$dir" && "./$name" "${extra[@]}") 2>&1 | tee "$log"
    local r=${PIPESTATUS[0]}
    echo "$1: exit $r, $(( $(date +%s) - t0 )) s"
    [ $r = 0 ] || rc=1
    local results=$(sed -n 's/^Results in \([^ ]*\).*/\1/p' "$log" | tail -1)
    if [ -n "$results" ] && [ -d "$dir/$results" ]; then
        command mv "$log" "$dir/$results/"
        tar czf "$dir/$results.tgz" -C "$dir/$(dirname "$results")" "$(basename "$results")"
        echo "Packed: $BENCHDIR/$dir/$results.tgz"
        packed+=("$dir/$results.tgz")
    fi
}
for script in ${SCRIPTS:-fft-fusion/run-bench.sh fft-fusion/run-nchan.sh xmac/run-xmac.sh xmac/run-staged.sh}; do
    run "$script"
done
submit=$(realpath --relative-to="$BENCHDIR" "${SLURM_SUBMIT_DIR:-$BENCHDIR}")
SLURM_LOG="${SLURM_JOB_ID:+$submit/logfiles/ozstar.$SLURM_JOB_ID.log}" common/scripts/make-copy-script.sh "$blog" "${packed[@]}"
exit $rc
