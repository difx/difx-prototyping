#!/bin/bash
#
# run-station.sh - the station stage at binbench's shape (fusebench, ../fft-fusion): 10 stations, 4
# dual-pol 32 MHz bands (8 bands), 2-bit packed VDIF, a 25.6 ms subint (6400 windows at 128
# channels, 800 at 1024), with the staged tensor XMAC after it (--xmac).  For the pulsar binning's
# cost against the whole chain: the station stage's time against the XMAC's says how much the
# binned XMAC may cost for the chain to stay within a factor of the unbinned one.  Build first
# (make fusebench in the repository root, see ../README.md).
#
# Usage:  ./run-station.sh [extra fusebench options]
#
# Logs and station.csv (every timing) go to results-local/<host>-station-<stamp>/, with table.txt
# the summary.

set -u
cd "$(dirname "$0")"
. ../common/scripts/bench-bin.sh
[ -x "$FUSEBENCH" ] || { echo "ERROR: build fusebench first (make fusebench in the repository root, see ../README.md)" >&2; exit 1; }
OUT=results-local/$(hostname -s)-station-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
{ echo "Host $(hostname), $(date), commit $(git log -1 --format='%h %s' 2>/dev/null)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,compute_cap,enforced.power.limit,power.default_limit --format=csv,noheader
  systemctl show -p Environment --value gpu-powercap.service 2>/dev/null | grep -o 'GPU_MAX_CLOCK=[0-9]*' | sed 's/^/SM clock locked: /'
} > "$OUT/system.txt"
cat "$OUT/system.txt"
CSV="$OUT/station.csv"
fail=0
for n in 128 1024; do
    w=$((6400 * 128 / n))
    log="$OUT/n$n.log"
    "$FUSEBENCH" --nant 10 --nband 8 --bw 32 --nchan $n --nwin $w --input packed --nbit 2 --xmac --reps 20 \
        --csv "$CSV" "$@" > "$log" 2>&1
    rc=$?
    { [ $rc = 0 ] && grep -q '^check: PASS$' "$log"; } || { fail=1; echo "n$n: exit $rc or a failed check"; }
    printf "%-6s done\n" "n$n"
done

python3 - "$CSV" > "$OUT/table.txt" <<'EOF'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r['kind'] == 'time']
print("10 stations, 8 x 32 MHz bands (4 dual-pol), 2-bit, 25.6 ms subint; ms per subint (median)")
print("station: the end-to-end kernel writing FP16 spectra (what the staged XMAC reads); fastest: the")
print("fastest whole station stage of any kind (FP32 or FP16 spectra); X: the staged tensor XMAC alone; chain:")
print("the two measured together.")
print("Within a factor F of the unbinned chain, the binned XMAC (with the bins) may cost up to")
print("F (S/X + 1) - S/X times the unbinned XMAC: for F = 2, S/X + 2")
print("%6s %9s %9s %9s %9s %7s %12s" % ("nchan", "station", "fastest", "X", "chain", "S/X", "binned max"))
for n in sorted({r['nchan'] for r in rows}, key=int):
    d = [r for r in rows if r['nchan'] == n]
    get = lambda pre: min(float(r['median_ms']) for r in d if r['name'].startswith(pre))
    s = get('end-to-end -> FP16 spectra')
    fastest = min(float(r['median_ms']) for r in d if r['name'].startswith(('fused,', '2 kernels', 'end-to-end')))
    x = get('XMAC: staged tensor')
    chain = get('station + XMAC: end-to-end -> FP16 spectra + staged tensor')
    print("%6s %9.3f %9.3f %9.3f %9.3f %7.2f %11.2fx" % (n, s, fastest, x, chain, s / x, s / x + 2))
EOF
cat "$OUT/table.txt"
echo "Results in $OUT (table.txt, station.csv)"
exit $fail
