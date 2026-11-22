#!/bin/bash
#
# run-bins.sh - pulsar binning in the XMAC (binbench, README.md): the test matrix, every version
# that applies, each checked against the host.  10 stations, 4 dual-pol 32 MHz bands, a 25.6 ms
# subint; P = 1.5 and 500 ms; 128 and 1024 channels; the gate, 16 bins scrunched, 16 and 128 bins
# separate.  DM 30, the output at full resolution (zeroed each subint) and with 8 channels averaged,
# accumulated (no zeroing each subint); and a high DM (300 for 1.5 ms, 1000 for 500 ms: at 1024
# channels the 1.5 ms pulsar's bin changes at every channel), full resolution.  Build first (make in the repository root, see ../README.md).
#
# Usage:  ./run-bins.sh [extra binbench options]
#
# Logs and bins.csv (every timing) go to results-local/<host>-bins-<stamp>/, with table.txt the
# summary.

set -u
cd "$(dirname "$0")"
. ../common/scripts/bench-bin.sh
BENCH=$BINBENCH
[ -x "$BENCH" ] || { echo "ERROR: build binbench first (make in the repository root, see ../README.md)" >&2; exit 1; }
OUT=results-local/$(hostname -s)-bins-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
{ echo "Host $(hostname), $(date), commit $(git log -1 --format='%h %s' 2>/dev/null)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,compute_cap,enforced.power.limit,power.default_limit --format=csv,noheader
  # a max-clock lock (nvidia-smi -lgc) does not show in nvidia-smi's queries: ar313 sets it in gpu-powercap.service
  systemctl show -p Environment --value gpu-powercap.service 2>/dev/null | grep -o 'GPU_MAX_CLOCK=[0-9]*' | sed 's/^/SM clock locked: /'
} > "$OUT/system.txt"
cat "$OUT/system.txt"
CSV="$OUT/bins.csv"
EXTRA=("$@")
fail=0
run() {   # run <name> <options...>
    local name="$1"; shift
    echo "$name: $*" >> "$OUT/runs.txt"
    "$BENCH" "$@" --reps 10 --check 64 --csv "$CSV" "${EXTRA[@]}" > "$OUT/$name.log" 2>&1
    local rc=$?
    { [ $rc = 0 ] && grep -q '^check: PASS$' "$OUT/$name.log"; } || { fail=1; echo "$name: exit $rc or a failed check"; }
    printf "%-28s done\n" "$name"
}
for p in 1.5 500; do
    hidm=300; [ $p = 500 ] && hidm=1000
    for n in 128 1024; do
        for o in "avg1" "avg8" "dm$hidm"; do
            out=(--dm 30); [ $o = avg8 ] && out=(--dm 30 --avg 8 --accumulate); [ $o = dm$hidm ] && out=(--dm $hidm)
            run "p$p-n$n-gate-$o"      --period-ms $p --nchan $n --bins gate "${out[@]}"
            run "p$p-n$n-16s-$o"       --period-ms $p --nchan $n --bins 16 --scrunch "${out[@]}"
            run "p$p-n$n-16-$o"        --period-ms $p --nchan $n --bins 16 "${out[@]}"
            run "p$p-n$n-128-$o"       --period-ms $p --nchan $n --bins 128 "${out[@]}"
        done
    done
done

python3 - "$CSV" > "$OUT/table.txt" <<'EOF'
import csv, sys
from collections import OrderedDict
rows = list(csv.DictReader(open(sys.argv[1])))
runs = OrderedDict()
for r in rows:
    k = (r['period_ms'], r['dm'], r['nchan'], r['bins'], r['scrunch'], r['avg'], r['accumulate'])
    runs.setdefault(k, {})[r['what']] = float(r['ms'])
vers = ['naive', 'stage', 'passes', 'sorted', 'gather', 'gatherwm']
print("ms per subint (25.6 ms); best / xmac: the fastest binned version against the unbinned staged XMAC (gatherwm: gather")
print("from the window-major spectra; sorted and gather start from channel-major ones, not counted)")
print("%-6s %5s %5s %-10s %4s %4s  %8s %7s %7s  " % ("P ms", "DM", "nchan", "bins", "avg", "acc", "xmac", "bins", "weights") +
      " ".join("%8s" % v for v in vers) + "  %-7s %6s" % ("best", "/xmac"))
for (p, dm, n, b, sc, a, ac), d in runs.items():
    name = b + (" scr" if sc == '1' else "")
    best = min((d[v], v) for v in vers if v in d)
    print("%-6s %5s %5s %-10s %4s %4s  %8.3f %7.3f %7.3f  " % (p, dm, n, name, a, ac, d['xmac'], d['bins'], d.get('weights', 0)) +
          " ".join("%8.2f" % d[v] if v in d else "%8s" % "-" for v in vers) + "  %-7s %6.2f" % (best[1], best[0] / d['xmac']))
EOF
cat "$OUT/table.txt"
echo "Results in $OUT (table.txt, bins.csv)"
exit $fail
