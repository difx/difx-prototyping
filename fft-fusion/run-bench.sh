#!/bin/bash
#
# run-bench.sh - the standard fusebench set, for comparing GPUs.  Build first (make in the repository root, see ../README.md).
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
#                  fused kernel does everything; the 2-kernel lines are decode + fringe into an
#                  FP16 or FP32 buffer, then the fused FFT.  The decode is the lean one (per-window
#                  work once per thread, bits per sample fixed at compile time); the *-direct runs
#                  repeat two of them with gpudifx's per-sample decode arithmetic, for comparison
#   packed-wola-nbit<N>  the same with 1-, 4- and 8-bit samples (2 bits: packed-wola)
#   *-chunk<W>     the packed-data pipelines W windows at a time, each antenna on its own
#                  stream, each antenna's part of the 2-kernel buffer reused for every chunk
#                  (--chunk): all antennas' chunks in flight at once, 84, 21, 10.5, 5.2 and
#                  2.6 MB in FP16 for both shapes, so that they can stay in L2
# all FFTs-per-block and elements-per-thread variants of the fused kernel, and (wola-fp16,
# packed-wola) the register caps (--regcap).  Logs go to results-local/<host>-<stamp>/, with
# summary.txt the lines worth reading.  The channel-count sweep is run-nchan.sh.

set -u
cd "$(dirname "$0")"
. ../common/scripts/bench-bin.sh
[ -x "$FUSEBENCH" ] || { echo "ERROR: build fusebench first (make in the repository root, see ../README.md)" >&2; exit 1; }
OUT=results-local/$(hostname -s)-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
{ echo "Host $(hostname), $(date), commit $(git log -1 --format='%h %s' 2>/dev/null)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,compute_cap,enforced.power.limit,power.default_limit --format=csv,noheader
  # a max-clock lock (nvidia-smi -lgc) does not show in nvidia-smi's queries: ar313 sets it in gpu-powercap.service
  systemctl show -p Environment --value gpu-powercap.service 2>/dev/null | grep -o 'GPU_MAX_CLOCK=[0-9]*' | sed 's/^/SM clock locked: /'
} > "$OUT/system.txt"
cat "$OUT/system.txt"

EXTRA=("$@")
fail=0
run() {   # run <name> <fusebench options...>
    local name=$1
    shift
    "$FUSEBENCH" "$@" "${EXTRA[@]}" > "$OUT/$name.log" 2>&1
    local rc=$?
    [ $rc = 0 ] || fail=1
    { echo "== $name (exit $rc): fusebench $* ${EXTRA[*]}"
      grep -E "^FFT input|^Input|^check: (PASS|FAIL)$|FAIL|unfused total|^    |^  fused|^  2 kernels|^  end-to-end" "$OUT/$name.log"
    } | tee -a "$OUT/summary.txt"
}

# shape: name, fusebench options, chunk sizes in windows
for shape in "c128:--nchan 128 --nwin 10000:1024 256 128 64 32" "c1024:--nchan 1024 --nwin 1000:128 32 16 8 4"; do
    tag=${shape%%:*}
    rest=${shape#*:}
    read -r -a opts <<< "${rest%%:*}"
    read -r -a chunks <<< "${rest#*:}"
    run "$tag-mpifxcorr"       "${opts[@]}" --rotator double
    run "$tag-fp32"            "${opts[@]}" --fpb sweep --ept sweep
    run "$tag-fp16"            "${opts[@]}" --input fp16 --fpb sweep --ept sweep
    run "$tag-wola-fp32"       "${opts[@]}" --wola --fpb sweep --ept sweep
    run "$tag-wola-fp16"       "${opts[@]}" --wola --input fp16 --fpb sweep --ept sweep --regcap sweep
    run "$tag-wola-fp16-hop75" "${opts[@]}" --wola --overlap 0.25 --input fp16 --ept sweep
    run "$tag-packed"          "${opts[@]}" --input packed --rotator double --fpb sweep --ept sweep
    run "$tag-packed-direct"   "${opts[@]}" --input packed --decode direct --fpb sweep --ept sweep
    run "$tag-packed-wola"     "${opts[@]}" --input packed --wola --fpb sweep --ept sweep --regcap sweep
    run "$tag-packed-wola-direct" "${opts[@]}" --input packed --wola --decode direct --fpb sweep --ept sweep --regcap sweep
    for nb in 1 4 8; do
        run "$tag-packed-wola-nbit$nb" "${opts[@]}" --input packed --wola --nbit $nb --fpb sweep --ept sweep --regcap sweep
    done
    run "$tag-packed-wola-ept-half" "${opts[@]}" --input packed --wola --ept half
    for w in "${chunks[@]}"; do
        run "$tag-packed-wola-chunk$w" "${opts[@]}" --input packed --wola --ept half --chunk $w --check 4
    done
done
echo "Results in $OUT (summary.txt)"
[ $fail = 0 ] || { echo "SOME RUNS FAILED: see the logs" >&2; exit 1; }
