#!/bin/bash
#
# run-twostage.sh - two-stage channelisation (twostagebench, docs/two-stage.md) over the PFB's
# parameters: 16, 32, 64 coarse channels x oversampling 4:3 and 2 x 4, 8, 16 taps, for about 64k
# final channels (65536 at O = 2, 49152 at 4:3: the nearest the oversampling allows), with the
# double-precision check of the whole chain in each run.  For comparison: gpudifx today at the same
# channel counts, and the single-stage end-to-end kernel at its largest size, 8192 channels
# (fusebench --xmac: the FP16-spectra kernel, the fastest variant), per ms of data.  Build first
# (make in the repository root, see ../README.md).
#
# Usage:  [NCHAN=8192 NWIN=128] ./run-twostage.sh [extra twostagebench options, e.g. --nant 10]
#   NCHAN: the channel count wanted (default 65536); NWIN: stage-2 windows per subint (default 16;
#   128 at NCHAN=8192 keeps the data per run the same)
#
# Logs and twostage.csv (every timing) go to results-local/<host>-twostage-<stamp>/, with
# table.txt the summary.

set -u
cd "$(dirname "$0")"
. ../common/scripts/bench-bin.sh
TWOSTAGE=$TWOSTAGEBENCH
[ -x "$TWOSTAGE" ] || { echo "ERROR: build twostagebench first (make in the repository root, see ../README.md)" >&2; exit 1; }
NCHAN=${NCHAN:-65536} NWIN=${NWIN:-16}
OUT=results-local/$(hostname -s)-twostage$([ $NCHAN = 65536 ] || echo -$NCHAN)-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
{ echo "Host $(hostname), $(date), commit $(git log -1 --format='%h %s' 2>/dev/null)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,compute_cap,enforced.power.limit,power.default_limit --format=csv,noheader
  # a max-clock lock (nvidia-smi -lgc) does not show in nvidia-smi's queries: ar313 sets it in gpu-powercap.service
  systemctl show -p Environment --value gpu-powercap.service 2>/dev/null | grep -o 'GPU_MAX_CLOCK=[0-9]*' | sed 's/^/SM clock locked: /'
} > "$OUT/system.txt"
cat "$OUT/system.txt"
EXTRA=("$@")
NANT=6 NBAND=8 NBIT=2
fail=0
CSV="$OUT/twostage.csv"
for os in 4:3 2; do
    ref=()
    for nc in 16 32 64; do
        for taps in 4 8 16; do
            name="nc$nc-os${os/:/to}-taps$taps"
            noref=--noref
            [ ${#ref[@]} = 0 ] && noref= && ref=(done)       # gpudifx today once per oversampling
            "$TWOSTAGE" --nc $nc --os $os --taps $taps --nchan $NCHAN --nant $NANT --nband $NBAND --nbit $NBIT --nwin $NWIN --check 4 \
                $noref --csv "$CSV" "${EXTRA[@]}" > "$OUT/$name.log" 2>&1
            rc=$?
            { [ $rc = 0 ] && grep -q '^check: PASS$' "$OUT/$name.log"; } || { fail=1; echo "$name: exit $rc or a failed check"; }
            printf "%-24s %s\n" "$name" "$(grep -E 'two stages: total' "$OUT/$name.log" | awk '{print $(NF-3) " ms, " $NF " ms/ms"}')"
        done
    done
done

# the single-stage end-to-end kernel at 8192 channels (FP16 spectra), and gpudifx today there
if [ -x "$FUSEBENCH" ]; then
    "$FUSEBENCH" --input packed --xmac --rotator double --fpb sweep --ept sweep --check 0 --nant $NANT --nband $NBAND \
        --nbit $NBIT --nchan 8192 --nwin 128 > "$OUT/single-8192.log" 2>&1 || { fail=1; echo "single-stage 8192: failed"; }
fi

python3 - "$CSV" "$OUT/single-8192.log" > "$OUT/table.txt" <<'EOF'
import csv, re, sys
rows = list(csv.DictReader(open(sys.argv[1])))
single = None
try:
    log = open(sys.argv[2]).read()
    sub = float(re.search(r"subint ([0-9.]+) ms", log).group(1))
    v = [float(m.group(1)) for m in re.finditer(r"end-to-end -> FP16 spectra, [^\n]*?\s([0-9.]+) \[", log)]
    single = min(v) / sub if v else None
    u = re.search(r"unfused total\s+([0-9.]+) \[", log)
    single_gdx = float(u.group(1)) / sub if u else None
except (OSError, AttributeError):
    single_gdx = None
ref = {r["os"]: float(r["ms_per_ms"]) for r in rows if r["name"].startswith("gpudifx today")}
print("ms per ms of data; slowdown: two-stage total / the single-stage end-to-end kernel at 8192 channels "
      f"({single:.4f} ms/ms)" if single else "ms per ms of data")
print(f"{'Nc':>3} {'O':>4} {'taps':>4} {'nchan':>6} {'stage 1':>8} {'stage 2':>8} {'total':>7} "
      f"{'slowdown':>8} {'vs gpudifx':>10}")
key = lambda r: (r["os"], int(r["nc"]), int(r["taps"]))
seen = {}
for r in rows:
    seen.setdefault(key(r), {})[r["name"]] = float(r["ms_per_ms"])
    seen[key(r)]["nchan"] = int(r["nchan"])
for (os, nc, taps), v in sorted(seen.items(), key=lambda kv: (kv[0][0] != "4:3", kv[0][1], kv[0][2])):
    tot = v.get("two stages: total")
    print(f"{nc:3d} {os:>4} {taps:4d} {v['nchan']:6d} {v.get('two stages: stage 1 (PFB, coarse FFT, corrections)', 0):8.4f} "
          f"{v.get('two stages: stage 2 (fine FFT)', 0):8.4f} "
          f"{tot:7.4f} {tot / single if single else 0:8.2f} {ref.get(os, 0) / tot:9.2f}x")
print(f"gpudifx today: {', '.join(f'{v:.4f} ms/ms at O = {k}' for k, v in ref.items())}"
      + (f"; at 8192 channels {single_gdx:.4f}" if single_gdx else ""))
EOF
cat "$OUT/table.txt"
echo "Results in $OUT (table.txt, twostage.csv)"
[ $fail = 0 ] || { echo "SOME RUNS FAILED" >&2; exit 1; }
