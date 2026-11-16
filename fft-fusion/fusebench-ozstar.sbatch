#!/bin/bash
#
#SBATCH --job-name=fusebench
#SBATCH --output=fusebench.%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8         # for the parallel build (the fused kernel is compiled per FFT length)
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=0:15:00            # estimate: build 2-4 min, run-bench.sh about 1 min (30 s on an RTX 2070)
#SBATCH --account=oz168
## add a partition if your site needs one:  #SBATCH --partition=<partition>
#
# fusebench on one OzSTAR GPU node (an A100 on the gina nodes): builds fusebench for the GPU the
# job got, then runs the standard set (run-bench.sh).  See README.md.
#
# Before submitting, on a login node:
#   1. a clone of difx-prototyping:  git clone https://github.com/difx/difx-prototyping.git
#   2. MathDx (cuFFTDx), header-only, unpacked from the PyPI wheel:
#        mkdir -p <dir> && cd <dir>
#        pip download --no-deps nvidia-mathdx==25.6.0 -d .
#        python3 -m zipfile -e nvidia_mathdx-25.6.0-py3-none-any.whl .
#      MATHDX is then <dir>/nvidia/mathdx
#   3. set the paths below (CUDA_HOME: the CUDA toolkit, the directory holding bin/nvcc; for
#      example the CUDAROOT in your gpudifx setup.local.bash)
# Submit from any directory; the SLURM log fusebench.<jobid>.log is written there and holds the
# summary.  The logs of every run are in $BENCHDIR/results-local/<host>-<stamp>/, which is also
# packed into a .tgz next to it at the end: send that and the SLURM log back.
#
# Extra fusebench options for every run go in EXTRA, for example:
#     sbatch --export=ALL,EXTRA="--reps 50" fusebench-ozstar.sbatch

BENCHDIR="SET-ME/difx-prototyping/fft-fusion"
MATHDX="SET-ME/nvidia/mathdx"
CUDA_HOME="SET-ME"
[ -x "$BENCHDIR/run-bench.sh" ] || { echo "ERROR: BENCHDIR $BENCHDIR has no run-bench.sh: set the paths at the top" >&2; exit 1; }
[ -f "$MATHDX/include/cufftdx.hpp" ] || { echo "ERROR: MATHDX $MATHDX has no include/cufftdx.hpp" >&2; exit 1; }
[ -x "$CUDA_HOME/bin/nvcc" ] || { echo "ERROR: CUDA_HOME $CUDA_HOME has no bin/nvcc" >&2; exit 1; }
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
read -r -a extra <<< "${EXTRA:-}"

cd "$BENCHDIR" || exit 1
SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
[ -n "$SM" ] || { echo "ERROR: no GPU visible to nvidia-smi" >&2; exit 1; }
echo "Node $(hostname), job ${SLURM_JOB_ID:-manual}, difx-prototyping $(git log -1 --format='%h %s')"
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,compute_cap --format=csv,noheader
nvcc --version | tail -1
echo "Building for sm_$SM ..."
t0=$(date +%s)
make -j"${SLURM_CPUS_PER_TASK:-8}" SM="$SM" MATHDX="$MATHDX" CUDA_HOME="$CUDA_HOME" > "build-sm$SM.log" 2>&1 \
    || { echo "ERROR: build failed, see $BENCHDIR/build-sm$SM.log" >&2; tail -20 "build-sm$SM.log" >&2; exit 1; }
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
