#!/bin/bash
#
# run-wide.sh - the whole chain (fusebench --xmac: station processing from packed data + XMAC)
# over a wider range of shapes: 1, 2 and 4 bits a sample x 1, 4 and 32 dual-polarisation
# subbands (2, 8 and 64 bands) x 128 channels at 4, 10, 32 (and 64) stations and 1024 channels at
# 10 (and 32).  Build first (make in the repository root, see ../README.md).
#
# Usage:  ./run-wide.sh [extra fusebench options, e.g. --reps 50]
#
# Each run times every variant (--fpb sweep --ept sweep) and checks everything fusebench --xmac
# checks: the station processing against the reference, and the staged XMAC from FP16 spectra
# against gpudifx's XMAC on the same FP16 spectra and on the FP32 ones.  The windows scale with the bands (8 / bands x 10000 at 128 channels up to
# 10 stations, at most 10000; a tenth of that at 1024 channels), less on a GPU with less than
# 40 GB, which also skips 64 stations at 128 channels and 32 at 1024.  Times per ms of data
# compare between window counts, not between band counts (the data rate grows with the bands).
# Logs go to results-local/<host>-wide-<stamp>/, with wide.txt the summary: ms per ms of data for
# each chain, and gpudifx today / staged.

set -u
cd "$(dirname "$0")"
. ../common/scripts/bench-bin.sh
[ -x "$FUSEBENCH" ] || { echo "ERROR: build fusebench first (make in the repository root, see ../README.md)" >&2; exit 1; }
OUT=results-local/$(hostname -s)-wide-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
{ echo "Host $(hostname), $(date), commit $(git log -1 --format='%h %s' 2>/dev/null)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,compute_cap,enforced.power.limit,power.default_limit --format=csv,noheader
  # a max-clock lock (nvidia-smi -lgc) does not show in nvidia-smi's queries: ar313 sets it in gpu-powercap.service
  systemctl show -p Environment --value gpu-powercap.service 2>/dev/null | grep -o 'GPU_MAX_CLOCK=[0-9]*' | sed 's/^/SM clock locked: /'
} > "$OUT/system.txt"
cat "$OUT/system.txt"
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
big=$([ "$MEM" -ge 40000 ] && echo 1 || echo 0)

EXTRA=("$@")
fail=0
printf "%-4s %-5s %-5s %-4s %6s %9s %9s %9s %11s %8s\n" nbit nband nchan nant nwin subint_ms gpudifx \
    e2e+gdx "e2e>16+stag" gain > "$OUT/wide.txt"
chain() {   # chain <nbit> <nband> <nchan> <nant> <nwin>
    local name="nbit$1-nband$2-nchan$3-nant$4"
    "$FUSEBENCH" --input packed --xmac --rotator double --fpb sweep --ept sweep --check 4 --nbit "$1" --nband "$2" \
        --nchan "$3" --nant "$4" --nwin "$5" "${EXTRA[@]}" > "$OUT/$name.log" 2>&1
    local rc=$?
    { [ $rc = 0 ] && ! grep -q "FAIL" "$OUT/$name.log"; } || { fail=1; echo "$name: exit $rc or a failed check, see $OUT/$name.log"; }
    awk -v nbit="$1" -v nband="$2" -v nchan="$3" -v nant="$4" -v nwin="$5" '
        / -> subint / { for (i = 1; i <= NF; i++) if ($i == "subint") sub_ms = $(i + 1) }
        function t(line) { match(line, /[0-9.]+ +\[/); return substr(line, RSTART, RLENGTH) + 0 }
        /station \+ XMAC: gpudifx today/ { a = t($0) }
        /station \+ XMAC: end-to-end \+ gpudifx XMAC/ { b = t($0) }
        /station \+ XMAC: end-to-end -> FP16 spectra \+ staged tensor/ { h = t($0) }
        END { if (sub_ms > 0 && h > 0)
                  printf "%-4s %-5s %-5s %-4s %6s %9s %9.4f %9.4f %11.4f %8.1f\n", nbit, nband, nchan, nant, nwin, sub_ms,
                         a / sub_ms, b / sub_ms, h / sub_ms, a / h }' "$OUT/$name.log" >> "$OUT/wide.txt"
    tail -1 "$OUT/wide.txt"
}
for nbit in 1 2 4; do
    for nband in 2 8 64; do
        for nant in 4 10 32 64; do
            [ $big = 0 ] && [ $nant -gt 32 ] && continue
            nwin=$((80000 / nband)); [ $nwin -gt 10000 ] && nwin=10000
            [ $big = 0 ] && [ $nant -gt 10 ] && nwin=$((nwin / 4))
            chain $nbit $nband 128 $nant $nwin
        done
        for nant in 10 32; do
            [ $big = 0 ] && [ $nant -gt 10 ] && continue
            nwin=$((8000 / nband)); [ $nwin -gt 1000 ] && nwin=1000
            chain $nbit $nband 1024 $nant $nwin
        done
    done
done
echo
cat "$OUT/wide.txt"
echo "Results in $OUT (wide.txt)"
[ $fail = 0 ] || { echo "SOME RUNS FAILED: see the logs" >&2; exit 1; }
