#!/bin/bash
#
# run-centres.sh - the shift to many phase centres after the XMAC (phasecentrebench, README.md),
# swept one parameter at a time from the default shape: 10 stations, 4 dual-pol 32 MHz bands of
# 8192 channels (3.9 kHz), a 25.6 ms subint of 100 windows in one piece, 100 centres, 64 channels
# averaged (250 kHz out).  Every run is checked (the delays, every shift version against the
# double-precision shift, the source's centre against the exact coherent sum).  Build first (make
# in the repository root, see ../README.md).
#
# Usage:  ./run-centres.sh [extra phasecentrebench options]
#
# Logs and centres.csv (every timing) go to results-local/<host>-centres-<stamp>/, with table.txt
# the summary.

set -u
cd "$(dirname "$0")"
. ../common/scripts/bench-bin.sh
BENCH=$PHASECENTREBENCH
[ -x "$BENCH" ] || { echo "ERROR: build phasecentrebench first (make in the repository root, see ../README.md)" >&2; exit 1; }
OUT=results-local/$(hostname -s)-centres-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
{ echo "Host $(hostname), $(date), commit $(git log -1 --format='%h %s' 2>/dev/null)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,compute_cap,enforced.power.limit,power.default_limit --format=csv,noheader
  # a max-clock lock (nvidia-smi -lgc) does not show in nvidia-smi's queries: ar313 sets it in gpu-powercap.service
  systemctl show -p Environment --value gpu-powercap.service 2>/dev/null | grep -o 'GPU_MAX_CLOCK=[0-9]*' | sed 's/^/SM clock locked: /'
} > "$OUT/system.txt"
cat "$OUT/system.txt"
CSV="$OUT/centres.csv"
EXTRA=("$@")
fail=0
run() {   # run <sweep> <value> <options...>
    local name="$1-$2"; shift 2
    echo "$name: $*" >> "$OUT/runs.txt"
    "$BENCH" "$@" --reps 10 --check 128 --csv "$CSV.$name" "${EXTRA[@]}" > "$OUT/$name.log" 2>&1
    local rc=$?
    { [ $rc = 0 ] && grep -q '^check: PASS$' "$OUT/$name.log"; } || { fail=1; echo "$name: exit $rc or a failed check"; }
    # the sweep's name and value in front of each row
    awk -v s="${name%-*}" -v v="${name##*-}" 'NR == 1 { print "sweep,value," $0; next } { print s "," v "," $0 }' \
        "$CSV.$name" >> "$CSV.all" && rm -f "$CSV.$name"
    printf "%-16s %s\n" "$name" "$(grep -E '^shift / XMAC' "$OUT/$name.log")"
}
for s in 1 10 100 1000; do run centres $s --centres $s; done
for k in 1 2 4 10; do run pieces $k --pieces $k --only simt,tensor; done           # 100, 50, 25, 10 windows a piece
for a in 8 16 32 128 256; do run avg $a --avg $a --only simt,tensor; done
# the channel count at the same subint (25.6 ms) and output resolution (250 kHz)
run nchan 1024 --nchan 1024 --nwin 800 --avg 8 --only simt,tensor
run nchan 65536 --nchan 65536 --nwin 13 --avg 512 --only simt,tensor
run nant 32 --nant 32 --only simt,tensor
awk 'NR == 1 || !/^sweep,value,gpu/' "$CSV.all" > "$CSV" && rm -f "$CSV.all"

python3 - "$CSV" > "$OUT/table.txt" <<'EOF'
import csv, sys
from collections import OrderedDict
rows = list(csv.DictReader(open(sys.argv[1])))
runs = OrderedDict()
for r in rows:
    k = (r['sweep'], r['value'])
    runs.setdefault(k, {'r': r})[r['what']] = float(r['ms_per_ms'])
print("ms per ms of data (per subint: XMAC over its pieces, the delays, each shift); shift / XMAC for the tensor shift;")
print("work: centres / windows per piece")
print("%-8s %6s %6s %5s %5s %6s %5s  %8s %8s %8s %8s %8s  %7s %6s %8s" % ("sweep", "value", "nchan", "nant", "win", "pieces", "avg",
      "xmac", "delays", "naive", "simt", "tensor", "sh/xmac", "work", "total"))
for (sw, v), d in runs.items():
    r = d['r']
    def g(k): return "%8.4f" % d[k] if k in d else "%8s" % "-"
    nw, k = int(r['nwin']), int(r['pieces'])
    t = d.get('tensor')
    tot = d['xmac'] + d['delays'] + t if t is not None else None
    print("%-8s %6s %6s %5s %5d %6d %5s  %s %s %s %s %s  %7s %6.2f %8s" % (sw, v, r['nchan'], r['nant'], nw, k, r['avg'],
          g('xmac'), g('delays'), g('naive'), g('simt'), g('tensor'),
          "%.2f" % (t / d['xmac']) if t is not None else "-", int(r['centres']) * k / nw,
          "%.4f" % tot if tot is not None else "-"))
EOF
cat "$OUT/table.txt"
echo "Results in $OUT (table.txt, centres.csv)"
exit $fail
