#!/bin/bash
#
# run-staged.sh - the staged tensor XMAC (xmacbench) against the number of stations, with each of
# its super-block sizes, and the tensor XMAC from its own layout for comparison.  Build first
# (make, see README.md).
#
# Usage:  ./run-staged.sh [extra xmacbench options, e.g. --reps 50]
#
# 4 dual-polarisation frequencies (8 bands), 128 channels for 4 ... 128 stations and 1024 channels
# for 10 ... 100, 10000 windows at 128 channels and 1000 at 1024 (20 and 16 ms of 64 MHz real
# data).  On a GPU with less than 40 GB the larger station counts use fewer windows (the times per
# ms of data still compare).  Each row: the tensor XMAC (its transpose and the XMAC timed apart),
# and the staged XMAC as it chooses (the super-block size it chose: KS x 16 receivers), then forced
# to KS 1 and 2 (8 channels a block) and 4 (4 channels a block).  Every run checks against double
# precision.  Logs go to results-local/<host>-staged-<stamp>/, with table.txt the summary: ms per
# subint, and ms per ms of data (below 1: faster than real time).

set -u
cd "$(dirname "$0")"
[ -x ./xmacbench ] || { echo "ERROR: build xmacbench first (make, see README.md)" >&2; exit 1; }
OUT=results-local/$(hostname -s)-staged-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
{ echo "Host $(hostname), $(date), commit $(git log -1 --format='%h %s' 2>/dev/null)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,compute_cap,enforced.power.limit,power.default_limit --format=csv,noheader
} > "$OUT/system.txt"
cat "$OUT/system.txt"
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

EXTRA=("$@")
fail=0
printf "%-6s %-5s %6s %9s %9s %10s %9s %3s %9s %9s %9s %11s %11s %11s\n" nchan nant nwin subint_ms tensor_x transpose \
    staged ks ks1 ks2 ks4 tensor_x/ms tr+tens/ms staged/ms > "$OUT/table.txt"
# the median time of a variant's line in a log
t() { awk -v pat="$2" '$0 ~ pat { match($0, /[0-9.]+ +\[/); print substr($0, RSTART, RLENGTH) + 0; exit }' "$1"; }
one() {   # one <log> <xmacbench options...>: run, note a failure
    local log=$1
    shift
    ./xmacbench "$@" --check 2 "${EXTRA[@]}" > "$log" 2>&1
    local rc=$?
    { [ $rc = 0 ] && grep -q '^check: PASS$' "$log"; } || { fail=1; echo "$log: exit $rc or a failed check"; }
}
row() {   # row <nchan> <nant> <nwin>
    local base="$OUT/nchan$1-nant$2" opts=(--nchan "$1" --nant "$2" --nwin "$3")
    one "$base-tensor.log" "${opts[@]}" --only tensor
    one "$base-staged.log" "${opts[@]}" --only staged
    for k in 1 2 4; do one "$base-ks$k.log" "${opts[@]}" --only staged --staged-tile $k; done
    local sub_ms tx tr sg ks k1 k2 k4
    sub_ms=$(sed -n 's/.*(subint \([0-9.]*\) ms).*/\1/p' "$base-tensor.log" | head -1)
    tx=$(t "$base-tensor.log" "of which the tensor-core XMAC")
    tr=$(t "$base-tensor.log" "of which the transpose")
    sg=$(t "$base-staged.log" "^  staged")
    ks=$(sed -n 's/.*super-blocks of \([0-9]*\) x 16.*/\1/p' "$base-staged.log")
    k1=$(t "$base-ks1.log" "^  staged")
    k2=$(t "$base-ks2.log" "^  staged")
    k4=$(t "$base-ks4.log" "^  staged")
    awk -v nchan="$1" -v nant="$2" -v nwin="$3" -v s="${sub_ms:-0}" -v tx="${tx:--}" -v tr="${tr:--}" -v sg="${sg:--}" \
        -v ks="${ks:--}" -v k1="${k1:--}" -v k2="${k2:--}" -v k4="${k4:--}" 'BEGIN {
        printf "%-6s %-5s %6s %9s %9s %10s %9s %3s %9s %9s %9s %11.4f %11.4f %11.4f\n", nchan, nant, nwin, s, tx, tr, sg,
               ks, k1, k2, k4, (s > 0 ? tx / s : 0), (s > 0 ? (tx + tr) / s : 0), (s > 0 ? sg / s : 0) }' >> "$OUT/table.txt"
    tail -1 "$OUT/table.txt"
}
big=$([ "$MEM" -ge 40000 ] && echo 1 || echo 0)
for nant in 4 6 10 16 24 32 48 64 100 128; do
    nwin=10000
    [ $big = 0 ] && [ $nant -gt 16 ] && nwin=4000
    [ $big = 0 ] && [ $nant -gt 64 ] && nwin=2000
    row 128 $nant $nwin
done
for nant in 10 32 64 100; do
    nwin=1000
    [ $big = 0 ] && [ $nant -gt 32 ] && nwin=500
    [ $big = 0 ] && [ $nant -gt 64 ] && nwin=250
    row 1024 $nant $nwin
done
echo
cat "$OUT/table.txt"
echo "Results in $OUT (table.txt)"
[ $fail = 0 ] || { echo "SOME RUNS FAILED: see the logs" >&2; exit 1; }
