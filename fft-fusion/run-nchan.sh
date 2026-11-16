#!/bin/bash
#
# run-nchan.sh - how the cost of a subint depends on the number of channels.  Build first (make,
# see README.md).
#
# Usage:  ./run-nchan.sh [extra fusebench options, e.g. --reps 50]
#
# The subint is the same length of data for every channel count: 2,560,000 samples per antenna
# and band (20 ms of 64 MHz real data; 10 antennas, 8 bands), so the times compare directly.
# For nchan = 32 ... 8192 (FFT lengths 64 ... 16384), from the packed data (2-bit VDIF), with
# and without --wola: the end-to-end kernel (best of all FFTs-per-block and elements-per-thread
# variants) and the two 2-kernel pipelines (cuFFTDx's variant for their FFT).  Logs go to
# results-local/<host>-nchan-<stamp>/, with table.txt the summary: ms per subint, and ms per ms of
# data (lower is better; below 1 is faster than real time for this array).

set -u
cd "$(dirname "$0")"
[ -x ./fusebench ] || { echo "ERROR: build fusebench first (make, see README.md)" >&2; exit 1; }
OUT=results-local/$(hostname -s)-nchan-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
{ echo "Host $(hostname), $(date), commit $(git log -1 --format='%h %s' 2>/dev/null)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,compute_cap --format=csv,noheader
} > "$OUT/system.txt"
cat "$OUT/system.txt"

SAMPLES=2560000
EXTRA=("$@")
fail=0
printf "%-6s %-5s %7s %9s %9s %9s %9s %11s\n" nchan wola windows subint_ms end2end 2k_fp16 2k_fp32 end2end/ms > "$OUT/table.txt"
for nchan in 32 64 128 256 512 1024 2048 4096 8192; do
    nwin=$(( SAMPLES / (2 * nchan) ))
    for wola in no yes; do
        name="nchan$nchan-$([ $wola = yes ] && echo wola || echo plain)"
        opts=(--nchan "$nchan" --nwin "$nwin" --input packed --fpb sweep --ept sweep --check 4)
        [ $wola = yes ] && opts+=(--wola)
        ./fusebench "${opts[@]}" "${EXTRA[@]}" > "$OUT/$name.log" 2>&1
        rc=$?
        [ $rc = 0 ] || { fail=1; echo "$name: exit $rc, see $OUT/$name.log"; }
        # the subint length, the best fused line, the two 2-kernel lines (median times)
        awk -v nchan="$nchan" -v wola="$wola" '
            / -> subint / { for (i = 1; i <= NF; i++) if ($i == "subint") sub_ms = $(i + 1) }
            /windows per antenna/ { for (i = 1; i <= NF; i++) if ($(i + 1) == "windows" && $(i + 2) == "per") nw = $i }
            /^  fused,/ { t = $0; sub(/.*\) +/, "", t); split(t, a, " "); if (best == "" || a[1] + 0 < best + 0) best = a[1] }
            /^  2 kernels: decode \+ fringe -> FP16/ { t = $0; sub(/.*FFT +/, "", t); split(t, a, " "); k16 = a[1] }
            /^  2 kernels: decode \+ fringe -> FP32/ { t = $0; sub(/.*FFT +/, "", t); split(t, a, " "); k32 = a[1] }
            END { printf "%-6s %-5s %7s %9s %9s %9s %9s %11.4f\n", nchan, wola, nw, sub_ms, best, k16, k32,
                         (sub_ms > 0 ? best / sub_ms : 0) }' "$OUT/$name.log" >> "$OUT/table.txt"
        tail -1 "$OUT/table.txt"
    done
done
echo
cat "$OUT/table.txt"
echo "Results in $OUT (table.txt)"
[ $fail = 0 ] || { echo "SOME RUNS FAILED: see the logs" >&2; exit 1; }
