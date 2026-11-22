#!/bin/bash
#
# run-seglen.sh - the gathering XMAC's segment length (--seg-len: slot blocks of 16 a block) for
# separate bins of the 1.5 ms pulsar.  The segments run in time order, then bin, so shorter segments
# keep the blocks running together on the same windows: at high DM a window's channels in a group
# fall in different bins, and with a block a bin over the whole subint each sector of the spectra
# comes from DRAM once for each (Nsight Compute: 10x the spectra at DM 300).  Shorter segments add
# blocks and output atomics.  Build first (make binbench in the repository root).
#
# Usage:  ./run-seglen.sh
#
# Logs and seglen.csv go to results-local/<host>-seglen-<stamp>/, with table.txt the summary.

set -u
cd "$(dirname "$0")"
[ -x ./binbench ] || { echo "ERROR: build binbench first (make binbench in the repository root)" >&2; exit 1; }
OUT=results-local/$(hostname -s)-seglen-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
{ echo "Host $(hostname), $(date), commit $(git log -1 --format='%h %s' 2>/dev/null)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,compute_cap,enforced.power.limit,power.default_limit --format=csv,noheader
  systemctl show -p Environment --value gpu-powercap.service 2>/dev/null | grep -o 'GPU_MAX_CLOCK=[0-9]*' | sed 's/^/SM clock locked: /'
} > "$OUT/system.txt"
cat "$OUT/system.txt"
CSV="$OUT/seglen.csv"
echo "seglen,bins,nchan,dm,what,ms,ratio" > "$CSV"
fail=0
for b in even:32 16; do
    for n in 128 1024; do
        for dm in 30 300; do
            for L in 32 8 4 2; do
                name="b${b#even:}-n$n-dm$dm-L$L"
                log="$OUT/$name.log"
                ./binbench --period-ms 1.5 --bins $b --nchan $n --dm $dm --seg-len $L --only gather,gatherwm > "$log" 2>&1
                rc=$?
                { [ $rc = 0 ] && grep -q '^check: PASS$' "$log"; } || { fail=1; echo "$name: exit $rc or a failed check"; }
                awk -v L=$L -v b=${b#even:} -v n=$n -v dm=$dm -v csv="$CSV" '
                    /^  xmac \(staged/ { x = $(NF-3) }
                    /^  binned: gather/ { what = $2; ms = $(NF-3); printf "%s,%s,%s,%s,%s,%s,%.3f\n", L, b, n, dm, what, ms, ms / x >> csv }
                ' "$log"
                printf "%-24s done\n" "$name"
            done
        done
    done
done
python3 - "$CSV" > "$OUT/table.txt" <<'EOF'
import csv, sys
from collections import OrderedDict
rows = list(csv.DictReader(open(sys.argv[1])))
Ls = sorted({r['seglen'] for r in rows}, key=lambda v: -int(v))
t = OrderedDict()
for r in rows:
    t.setdefault((r['bins'], r['nchan'], r['dm'], r['what']), {})[r['seglen']] = float(r['ratio'])
print("1.5 ms pulsar, separate bins: the gathering XMAC (with the sort) against the unbinned XMAC, by segment length")
print("(slot blocks of 16 a block; segments in time order, then bin).  gather: from channel-major; gatherwm: window-major")
print("%5s %6s %5s %-9s " % ("bins", "nchan", "DM", "") + " ".join("%7s" % ("L" + L) for L in Ls))
for (b, n, dm, w), d in t.items():
    print("%5s %6s %5s %-9s " % (b, n, dm, w) + " ".join("%7.2f" % d[L] if L in d else "%7s" % "-" for L in Ls))
EOF
cat "$OUT/table.txt"
echo "Results in $OUT (table.txt, seglen.csv)"
exit $fail
