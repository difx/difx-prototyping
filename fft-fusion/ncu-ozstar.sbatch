#!/bin/bash
#
#SBATCH --job-name=fusebench-ncu
#SBATCH --output=fusebench-ncu.%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8         # for the parallel build
#SBATCH --gres=gpu:1
#SBATCH --mem=32G                 # the build: up to ~4 GB per ptxas (N=8192 and up), eight at once
#SBATCH --tmp=8G                  # job-local disk ($JOBFS) for nvcc's intermediate files
#SBATCH --time=1:00:00            # estimate: build ~25 min on a gina node if not already built, profiles ~3 min
#SBATCH --account=oz168
## add a partition if your site needs one:  #SBATCH --partition=<partition>
#
# Nsight Compute profiles of the fusebench kernels on one OzSTAR GPU node (an A100 on the gina
# nodes): the full set of sections for one launch of each kernel below, to find out why the fused
# kernel stays well below the memory bandwidth (README.md, "What changes on the A100").
#
# Set up and submit as fusebench-ozstar.sbatch (its header): the same DIFX_SETUP, run from
# fft-fusion after "make mathdx" on a login node.  ncu comes from the CUDA toolkit DIFX_SETUP
# selects (NCU=<path to ncu> overrides).  The job writes results-local/ncu-<host>-<stamp>/:
#   <case>.ncu-rep   the reports (open in the Nsight Compute GUI: ncu-ui), with the source
#   <case>.txt       the same as text (ncu --import ... --page details)
#   <case>.run.log   fusebench's own output under the profiler
# and packs it into a .tgz next to it: send that and the SLURM log back.  If the GPU's
# performance counters are restricted to administrators, ncu fails with ERR_NVGPUCTRPERM; the job
# then stops and says so.  ncu locks the GPU to its base clock while it profiles (the A100's
# 1095 MHz rather than 1410), so its durations are longer than fusebench's.
#
# CASES=<extended regex> profiles only the cases whose names match, e.g. the xmacbench ones alone
# (a few minutes once built):
#     sbatch --export=ALL,CASES='^x' ncu-ozstar.sbatch

DIFX_SETUP="${DIFX_SETUP:-$HOME/setup_difx.claude}"
BENCHDIR="${BENCHDIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$DIFX_SETUP" ] || { echo "ERROR: DIFX_SETUP $DIFX_SETUP does not exist" >&2; exit 1; }
[ -f "$BENCHDIR/fusebench.cu" ] || { echo "ERROR: BENCHDIR $BENCHDIR is not fft-fusion: submit from there or set BENCHDIR" >&2; exit 1; }
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
NVCC="${CUDAROOT:+$CUDAROOT/bin/}nvcc"
command -v "$NVCC" > /dev/null || { echo "ERROR: after $DIFX_SETUP, no nvcc on PATH and none in CUDAROOT" >&2; exit 1; }
cudahome=$(dirname "$(dirname "$(command -v "$NVCC")")")
if [ -z "${NCU:-}" ]; then
    NCU=$(command -v ncu || true)
    [ -n "$NCU" ] || NCU=$(ls "$cudahome"/bin/ncu "$cudahome"/nsight-compute*/ncu 2>/dev/null | head -1)
fi
[ -n "$NCU" ] && [ -x "$NCU" ] || { echo "ERROR: no ncu (Nsight Compute) on PATH or in $cudahome; set NCU=<path>" >&2; exit 1; }

SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
[ -n "$SM" ] || { echo "ERROR: no GPU visible to nvidia-smi" >&2; exit 1; }
echo "Node $(hostname), job ${SLURM_JOB_ID:-manual}, difx-prototyping $(git log -1 --format='%h %s')"
echo "Environment from $DIFX_SETUP; $("$NCU" --version | tail -1)"
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,compute_cap --format=csv,noheader
echo "Building for sm_$SM ..."
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
head -1 "$blog"

OUT=results-local/ncu-$(hostname -s)-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
QUICK=(--reps 1 --warmup 0 --check 0)
# big enough not to fit in the A100's 40 MB L2 cache (128 channels: 164 MB of FP32 input per antenna)
C128=(--nchan 128 --nwin 10000)
C1024=(--nchan 1024 --nwin 1000)
fail=0

# run_profile <program> <case> <kernel regex> <options...>: the first launch of the kernel; only the
# cases whose names match $CASES (an extended regex) when it is set
run_profile() {
    local prog=$1 name=$2 kernel=$3
    shift 3
    [ -n "${CASES:-}" ] && ! [[ $name =~ $CASES ]] && return
    echo "== $name: $kernel in $prog $*"
    local t0=$(date +%s)
    "$NCU" --set full --import-source yes -k "regex:$kernel" -c 1 -f -o "$OUT/$name" \
        "build-sm$SM/$prog" "$@" "${QUICK[@]}" > "$OUT/$name.run.log" 2>&1
    local rc=$?
    if grep -q ERR_NVGPUCTRPERM "$OUT/$name.run.log"; then
        echo "ERROR: ncu may not read this GPU's performance counters (ERR_NVGPUCTRPERM): profiling" \
             "needs the driver's NVreg_RestrictProfilingToAdminUsers=0, or ask the OzSTAR admins" >&2
        exit 1
    fi
    if [ $rc != 0 ] || [ ! -f "$OUT/$name.ncu-rep" ]; then
        echo "   FAILED (exit $rc), see $OUT/$name.run.log"
        fail=1
        return
    fi
    "$NCU" --import "$OUT/$name.ncu-rep" --page details > "$OUT/$name.txt" 2>&1
    grep -E "^  [a-zA-Z_].*\(|Duration|DRAM Throughput|Memory Throughput|Compute \(SM\) Throughput|Achieved Occupancy|Registers Per Thread" \
        "$OUT/$name.txt" | head -8 | cut -c1-150
    echo "   ($(( $(date +%s) - t0 )) s)"
}
profile() { run_profile fusebench "$@"; }     # profile <case> <kernel regex> <fusebench options...>
profilex() { run_profile xmacbench "$@"; }    # profilex <case> <kernel regex> <xmacbench options...>

# the fused kernel with an FP32 and an FP16 buffer as input; a plain streaming kernel (the
# fixed-point rotation, post_kernel) for comparison
profile c128-fused-fp32            fused_fft_frac "${C128[@]}"
profile c128-rotation-streaming    post_kernel    "${C128[@]}"
profile c128-fused-fp16-wola       fused_fft_frac "${C128[@]}" --input fp16 --wola
profile c128-fused-fp16-wola-ept-half fused_fft_frac "${C128[@]}" --input fp16 --wola --ept half
# the end-to-end kernel from the packed data, at both elements-per-thread settings (the lean
# decode, fusebench's default; job 21038155 has the same profiles with gpudifx's per-sample
# decode arithmetic, now --decode direct), with 40 registers, and from 8-bit data
profile c128-packed-wola           fused_fft_frac "${C128[@]}" --input packed --wola
profile c128-packed-wola-ept-half fused_fft_frac "${C128[@]}" --input packed --wola --ept half
profile c128-packed-wola-ept-half-cap1536 fused_fft_frac "${C128[@]}" --input packed --wola --ept half --fpb 2 --regcap 1536
profile c128-packed-wola-ept-half-nbit8 fused_fft_frac "${C128[@]}" --input packed --wola --ept half --nbit 8
# the first stage of the two-kernel pipeline, and gpudifx's decode kernel (FP64 phases)
profile c128-decode-to-fp16        decode_to      "${C128[@]}" --input packed --wola
profile c128-gpudifx-decode        gpu_fused_fringe_tiled "${C128[@]}" --input packed --rotator double
# registers capped for 1024 threads per SM (64 registers a thread on the A100)
profile c128-fused-fp16-wola-cap1024 fused_fft_frac "${C128[@]}" --input fp16 --wola --regcap 1024
profile c128-packed-wola-cap1024     fused_fft_frac "${C128[@]}" --input packed --wola --regcap 1024
profile c1024-fused-fp32           fused_fft_frac "${C1024[@]}"
profile c1024-packed-wola          fused_fft_frac "${C1024[@]}" --input packed --wola
profile c1024-packed-wola-ept-half fused_fft_frac "${C1024[@]}" --input packed --wola --ept half
profile c1024-packed-wola-cap1024  fused_fft_frac "${C1024[@]}" --input packed --wola --regcap 1024
# gpudifx's XMAC kernel (fusebench --xmac, 10 stations)
profile c128-xmac-gpudifx          gpu_fuse_xmac_and_average "${C128[@]}" --input packed --xmac --fpb 16 --ept 16
# the end-to-end kernel writing FP16 spectra in its own order, and the staged tensor XMAC that
# reshuffles them itself (at the variants fastest on the A100 for the FP32 output)
profile c128-half-out-fpb4         fused_fft_half "${C128[@]}" --input packed --xmac --fpb 4 --ept 16
profile c1024-half-out-fpb1        fused_fft_half "${C1024[@]}" --input packed --xmac --fpb 1 --ept 16
profile c128-xmac-staged           xmac_staged_kernel "${C128[@]}" --input packed --xmac --fpb 4 --ept 16
profile c1024-xmac-staged          xmac_staged_kernel "${C1024[@]}" --input packed --xmac --fpb 1 --ept 16

# the staged tensor XMAC alone (xmacbench, its own spectra), at each super-block size it chooses
# (KS 1 at 4 stations, 2 at 10, 4 from 24), and KS 2 forced at 64 for comparison
profilex xstaged-c128-n4           xmac_staged_kernel --nchan 128 --nwin 10000 --nant 4 --only staged
profilex xstaged-c128-n10          xmac_staged_kernel --nchan 128 --nwin 10000 --nant 10 --only staged
profilex xstaged-c128-n64          xmac_staged_kernel --nchan 128 --nwin 10000 --nant 64 --only staged
profilex xstaged-c128-n64-ks2      xmac_staged_kernel --nchan 128 --nwin 10000 --nant 64 --only staged --staged-tile 2
profilex xstaged-c128-n100         xmac_staged_kernel --nchan 128 --nwin 10000 --nant 100 --only staged
profilex xstaged-c1024-n32         xmac_staged_kernel --nchan 1024 --nwin 1000 --nant 32 --only staged

tar czf "$OUT.tgz" -C results-local "$(basename "$OUT")"
echo "Packed: $BENCHDIR/$OUT.tgz"
SLURM_LOG="${SLURM_JOB_ID:+${SLURM_SUBMIT_DIR:-$BENCHDIR}/fusebench-ncu.$SLURM_JOB_ID.log}" ./make-copy-script.sh "$OUT.tgz"
[ $fail = 0 ] || { echo "SOME PROFILES FAILED: see the .run.log files" >&2; exit 1; }
