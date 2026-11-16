#!/bin/bash
#
# run-bench.sh - the standard fusebench set, for comparing GPUs.  Build first (make, see README.md).
#
# Usage:  ./run-bench.sh [extra fusebench options, e.g. --reps 50]
#
# Two shapes (10 antennas, 8 bands, real data, 64 MHz):
#   c128   128 channels (256-point FFT), 10000 FFT lengths per subint (20 ms)
#   c1024  1024 channels (2048-point FFT), 1000 FFT lengths per subint (16 ms)
# No autocorrelations (they are to move to the XMAC). For each shape:
#   mpifxcorr      FP32 input, mpifxcorr's FP64-phase rotation kernel in the unfused pipeline
#   fp32, fp16     FFT input precision, fixed-point phases in both pipelines
#   wola-*         --wola: Vorbis-taper window 12.5% / 75% / 12.5%, each window overlapping each
#                  neighbour over its taper (hop 0.875 FFT lengths), FP32 and FP16 input; and,
#                  for comparison, the same window with hop 0.75 (consecutive windows share 25%)
#   packed-*       from the packed data (2-bit VDIF): decode + fringe rotation as well. The
#                  unfused pipeline is gpudifx today (its decode kernel and mpifxcorr's rotation,
#                  FP64 phases; fixed-point with --wola, which gpudifx's decode cannot do); the
#                  fused kernel does everything, with direct and with tiled decode; the 2-kernel
#                  lines are decode + fringe into an FP16 or FP32 buffer, then the fused FFT
#   *-chunk<W>     the packed-data pipelines one antenna and W windows at a time, the 2-kernel
#                  buffer reused for every chunk so that it can stay in L2 (--chunk)
# all FFTs-per-block and elements-per-thread variants of the fused kernel, and (wola-fp16,
# packed-wola) the register caps (--regcap).  Logs go to results-local/<host>-<stamp>/, with
# summary.txt the lines worth reading.  The channel-count sweep is run-nchan.sh.

set -u
cd "$(dirname "$0")"
[ -x ./fusebench ] || { echo "ERROR: build fusebench first (make MATHDX=...)" >&2; exit 1; }
OUT=results-local/$(hostname -s)-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
{ echo "Host $(hostname), $(date), commit $(git log -1 --format='%h %s' 2>/dev/null)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,compute_cap --format=csv,noheader
} > "$OUT/system.txt"
cat "$OUT/system.txt"

EXTRA=("$@")
fail=0
run() {   # run <name> <fusebench options...>
    local name=$1
    shift
    ./fusebench "$@" "${EXTRA[@]}" > "$OUT/$name.log" 2>&1
    local rc=$?
    [ $rc = 0 ] || fail=1
    { echo "== $name (exit $rc): fusebench $* ${EXTRA[*]}"
      grep -E "^FFT input|^Input|^check: (PASS|FAIL)$|FAIL|unfused total|^    |^  fused|^  2 kernels|^  end-to-end" "$OUT/$name.log"
    } | tee -a "$OUT/summary.txt"
}

for shape in "c128:--nchan 128 --nwin 10000" "c1024:--nchan 1024 --nwin 1000"; do
    tag=${shape%%:*}
    read -r -a opts <<< "${shape#*:}"
    run "$tag-mpifxcorr"       "${opts[@]}" --rotator double
    run "$tag-fp32"            "${opts[@]}" --fpb sweep --ept sweep
    run "$tag-fp16"            "${opts[@]}" --input fp16 --fpb sweep --ept sweep
    run "$tag-wola-fp32"       "${opts[@]}" --wola --fpb sweep --ept sweep
    run "$tag-wola-fp16"       "${opts[@]}" --wola --input fp16 --fpb sweep --ept sweep --regcap sweep
    run "$tag-wola-fp16-hop75" "${opts[@]}" --wola --overlap 0.25 --input fp16 --ept sweep
    run "$tag-packed"          "${opts[@]}" --input packed --rotator double --fpb sweep --ept sweep
    run "$tag-packed-tiled"    "${opts[@]}" --input packed --decode tiled --fpb sweep --ept sweep
    run "$tag-packed-wola"     "${opts[@]}" --input packed --wola --fpb sweep --ept sweep --regcap sweep
    run "$tag-packed-wola-ept-half" "${opts[@]}" --input packed --wola --ept half
    for w in 4096 2048 1024 512 256; do
        run "$tag-packed-wola-chunk$w" "${opts[@]}" --input packed --wola --ept half --chunk $w --check 4
    done
done
echo "Results in $OUT (summary.txt)"
[ $fail = 0 ] || { echo "SOME RUNS FAILED: see the logs" >&2; exit 1; }
