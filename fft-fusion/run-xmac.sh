#!/bin/bash
#
# run-xmac.sh - the XMAC stage on its own (xmacbench) against the number of stations.  Build
# first (make, see README.md).
#
# Usage:  ./run-xmac.sh [extra xmacbench options, e.g. --reps 50]
#
# 4 dual-polarisation frequencies (8 bands), 128 channels, 10000 windows (20 ms of 64 MHz real
# data), for 4 ... 64 stations; and 1024 channels, 1000 windows (16 ms) for 10 and 32 stations.
# On a GPU with less than 40 GB the larger station counts use 4000 windows (the times per ms of
# data still compare).  Each run: gpudifx's XMAC kernel and the staged tensor-core kernel from
# FP16 spectra.  Then the whole chain (fusebench --xmac: the station processing from 2-bit packed
# data, no window, and the XMAC) for 4 ... 64 stations at 128 channels (and 100, 4000 windows, on
# a GPU with 40 GB or more) and 10 and 32 at 1024: gpudifx today (both stages), and the end-to-end
# kernel followed by gpudifx's XMAC, or writing FP16 spectra + the staged tensor one (the fastest
# variant of each kind).  Logs go to
# results-local/<host>-xmac-<stamp>/, with table.txt (the XMAC alone) and chain.txt (the whole
# chain) the summaries: ms per subint and ms per ms of data (below 1: faster than real time).

set -u
cd "$(dirname "$0")"
. ./bench-bin.sh
[ -x "$XMACBENCH" ] || { echo "ERROR: build xmacbench first (make, see README.md)" >&2; exit 1; }
OUT=results-local/$(hostname -s)-xmac-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
{ echo "Host $(hostname), $(date), commit $(git log -1 --format='%h %s' 2>/dev/null)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,compute_cap,enforced.power.limit,power.default_limit --format=csv,noheader
  # a max-clock lock (nvidia-smi -lgc) does not show in nvidia-smi's queries: ar313 sets it in gpu-powercap.service
  systemctl show -p Environment --value gpu-powercap.service 2>/dev/null | grep -o 'GPU_MAX_CLOCK=[0-9]*' | sed 's/^/SM clock locked: /'
} > "$OUT/system.txt"
cat "$OUT/system.txt"
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

EXTRA=("$@")
fail=0
printf "%-6s %-5s %6s %9s %9s %9s %11s %11s\n" nchan nant nwin subint_ms gpudifx staged gpudifx/ms staged/ms \
    > "$OUT/table.txt"
run() {   # run <nchan> <nant> <nwin>
    local name="nchan$1-nant$2"
    "$XMACBENCH" --nchan "$1" --nant "$2" --nwin "$3" --check 2 "${EXTRA[@]}" > "$OUT/$name.log" 2>&1
    local rc=$?
    [ $rc = 0 ] || { fail=1; echo "$name: exit $rc, see $OUT/$name.log"; }
    awk -v nchan="$1" -v nant="$2" -v nwin="$3" '
        /subint [0-9.]+ ms/ { for (i = 1; i <= NF; i++) if ($i == "(subint") sub_ms = $(i + 1) }
        function t(line) { match(line, /[0-9.]+ +\[/); return substr(line, RSTART, RLENGTH) + 0 }
        /^  gpudifx/ { g = t($0) }
        /^  staged/ { sg = t($0) }
        END { printf "%-6s %-5s %6s %9s %9s %9s %11.4f %11.4f\n", nchan, nant, nwin, sub_ms, g, (sg == "" ? "-" : sg),
                     (sub_ms > 0 ? g / sub_ms : 0), (sub_ms > 0 ? sg / sub_ms : 0) }' "$OUT/$name.log" >> "$OUT/table.txt"
    tail -1 "$OUT/table.txt"
}
for nant in 4 6 10 16 24 32 48 64; do
    nwin=10000
    [ "$MEM" -lt 40000 ] && [ $nant -gt 16 ] && nwin=4000
    run 128 $nant $nwin
done
run 1024 10 1000
run 1024 32 1000

# the whole chain: station processing + XMAC (fusebench --xmac), ms per ms of data
printf "%-6s %-5s %6s %9s %10s %10s %12s %9s\n" nchan nant nwin subint_ms gpudifx e2e+gdx "e2e>16+stag" gain \
    > "$OUT/chain.txt"
chain() {   # chain <nchan> <nant> <nwin>
    local name="chain-nchan$1-nant$2"
    [ -x "$FUSEBENCH" ] || { echo "no fusebench: skipping the chain runs"; return; }
    "$FUSEBENCH" --input packed --xmac --rotator double --fpb sweep --ept sweep --check 4 --nchan "$1" --nant "$2" \
        --nwin "$3" "${EXTRA[@]}" > "$OUT/$name.log" 2>&1
    local rc=$?
    [ $rc = 0 ] || { fail=1; echo "$name: exit $rc, see $OUT/$name.log"; }
    awk -v nchan="$1" -v nant="$2" -v nwin="$3" '
        / -> subint / { for (i = 1; i <= NF; i++) if ($i == "subint") sub_ms = $(i + 1) }
        function t(line) { match(line, /[0-9.]+ +\[/); return substr(line, RSTART, RLENGTH) + 0 }
        /station \+ XMAC: gpudifx today/ { a = t($0) }
        /station \+ XMAC: end-to-end \+ gpudifx XMAC/ { b = t($0) }
        /station \+ XMAC: end-to-end -> FP16 spectra \+ staged tensor/ { h = t($0) }
        END { if (sub_ms > 0 && h > 0) printf "%-6s %-5s %6s %9s %10.4f %10.4f %12.4f %9.1f\n", nchan, nant, nwin,
                  sub_ms, a / sub_ms, b / sub_ms, h / sub_ms, a / h }' "$OUT/$name.log" \
        >> "$OUT/chain.txt"
    tail -1 "$OUT/chain.txt"
}
for nant in 4 10 16 32 64; do
    nwin=10000
    [ "$MEM" -lt 40000 ] && [ $nant -gt 10 ] && nwin=2500
    [ "$MEM" -lt 40000 ] && [ $nant -gt 32 ] && continue
    chain 128 $nant $nwin
done
[ "$MEM" -lt 40000 ] || chain 128 100 4000
chain 1024 10 1000
[ "$MEM" -lt 40000 ] || chain 1024 32 1000
echo
cat "$OUT/table.txt"
echo
cat "$OUT/chain.txt"
echo "Results in $OUT (table.txt, chain.txt)"
[ $fail = 0 ] || { echo "SOME RUNS FAILED: see the logs" >&2; exit 1; }
