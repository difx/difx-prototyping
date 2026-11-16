#!/bin/bash
#
#SBATCH --job-name=fusebench
#SBATCH --output=fusebench.%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8         # for the parallel build (the fused kernel is compiled per FFT length)
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=0:30:00            # estimate: build 4-8 min, run-bench.sh 1-2 min (70 s on an RTX 2070)
#SBATCH --account=oz168
## add a partition if your site needs one:  #SBATCH --partition=<partition>
#
# fusebench on one OzSTAR GPU node (an A100 on the gina nodes): builds fusebench for the GPU the
# job got, then runs the standard set (run-bench.sh).  See README.md.
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
# and holds the summary.  The logs of every run are in results-local/<host>-<stamp>/, which is also
# packed into a .tgz next to it at the end: send that and the SLURM log back.
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
make -j"${SLURM_CPUS_PER_TASK:-8}" SM="$SM" "${mathdx[@]}" > "build-sm$SM.log" 2>&1 \
    || { echo "ERROR: build failed, see $BENCHDIR/build-sm$SM.log" >&2; tail -20 "build-sm$SM.log" >&2; exit 1; }
head -1 "build-sm$SM.log"                   # what the Makefile chose: CUDA, MathDx
"${CUDAROOT:+$CUDAROOT/bin/}nvcc" --version | tail -1   # the Makefile's choice: CUDAROOT, else PATH
echo "Built in $(( $(date +%s) - t0 )) s"

t0=$(date +%s)
./run-bench.sh "${extra[@]}"
rc=$?
echo "run-bench.sh: exit $rc, $(( $(date +%s) - t0 )) s"
results=$(ls -td results-local/"$(hostname -s)"-* 2>/dev/null | head -1)
if [ -n "$results" ]; then
    tar czf "$results.tgz" -C results-local "$(basename "$results")"
    echo "Packed: $BENCHDIR/$results.tgz"
fi
exit $rc
