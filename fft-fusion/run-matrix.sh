#!/bin/bash
#
# run-matrix.sh - the whole chain (fusebench --xmac: station processing from packed data + XMAC)
# over a big matrix of shapes, looking for corner cases: 1, 2 and 4 bits a sample x 1, 8 and 64
# dual-polarisation subbands x 32, 512 and 4096 channels x 4, 16, 64 and 256 stations (108 shapes).
# Build first (make, see README.md).
#
# Usage:  ./run-matrix.sh [extra fusebench options]
#         NBITS="2" NSUBS="1 8" NCHANS="32" NANTS="4 16" ./run-matrix.sh     # a subset
#
# Each shape gets the most windows (FFT lengths a subint) that fit three budgets, at most 10000:
#   - GPU memory: fusebench's buffers take about 96 B per spectral value (station x window x band x
#     channel), plus two copies of the visibilities (8 B per baseline x product x channel x
#     frequency), within 80% of the GPU's memory; the visibilities also within 20% of the host's
#     (fusebench copies them back to compare)
#   - work: about 2e10 complex multiply-accumulates (baselines x 4 products x channels x
#     frequencies x windows), so gpudifx's XMAC, timed some 70 times a shape, stays in seconds
#   - data: 1e8 spectral values
# The window counts are not rounded, so most are not a multiple of 16 (the tensor XMACs' block of
# windows): the padding is exercised too.  At least 16 windows; a shape whose work at 16 windows is
# over the budget runs with --reps 5 --warmup 1, and one that does not fit in memory at 16 is
# skipped.
#
# Every run checks everything fusebench --xmac checks (--check 4).  A run fails on a non-zero exit
# or a failed check.  Results:
#   results-local/<host>-matrix-<stamp>/   the logs, fusebench's CSV of every timing and check
#                                          (raw.csv), and summary.csv
#   results/<host>-matrix-<stamp>.csv      summary.csv again, in the repository (results-local is
#                                          not): one row per shape, see matrix_summary.py

set -u
cd "$(dirname "$0")"
. ./bench-bin.sh
[ -x "$FUSEBENCH" ] || { echo "ERROR: build fusebench first (make, see README.md)" >&2; exit 1; }
STAMP=$(date +%Y%m%d-%H%M%S)
NAME=$(hostname -s)-matrix-$STAMP
OUT=results-local/$NAME
mkdir -p "$OUT" results
{ echo "Host $(hostname), $(date), commit $(git log -1 --format='%h %s' 2>/dev/null)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,compute_cap,enforced.power.limit,power.default_limit --format=csv,noheader
  # a max-clock lock (nvidia-smi -lgc) does not show in nvidia-smi's queries: ar313 sets it in gpu-powercap.service
  systemctl show -p Environment --value gpu-powercap.service 2>/dev/null | grep -o 'GPU_MAX_CLOCK=[0-9]*' | sed 's/^/SM clock locked: /'
} > "$OUT/system.txt"
cat "$OUT/system.txt"
GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
HOST_MB=$(awk '/MemTotal/ {print int($2 / 1024)}' /proc/meminfo)
# the job's memory limit, where Slurm sets one (cgroup v2, then v1), if lower
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
    [ -r "$f" ] || continue
    lim=$(cat "$f"); [ "$lim" = max ] && continue
    lim=$((lim / 1048576)); [ "$lim" -gt 0 ] && [ "$lim" -lt "$HOST_MB" ] && HOST_MB=$lim
done
POWER=$(nvidia-smi --query-gpu=enforced.power.limit --format=csv,noheader,nounits | head -1)
CLOCK=$(systemctl show -p Environment --value gpu-powercap.service 2>/dev/null | grep -o 'GPU_MAX_CLOCK=[0-9]*' | cut -d= -f2)
echo "GPU memory ${GPU_MB} MB, host memory for the visibilities: ${HOST_MB} MB"

EXTRA=("$@")
NBITS=${NBITS:-1 2 4}
NSUBS=${NSUBS:-1 8 64}
NCHANS=${NCHANS:-32 512 4096}
NANTS=${NANTS:-4 16 64 256}
RUNS="$OUT/runs.csv"
echo "nbit,nsub,nchan,nant,nwin,reps,status,exit,note" > "$RUNS"
fail=0

# plan <nbit> <nsub> <nchan> <nant>: prints "<nwin> <reps option> <note>", nwin 0 to skip
plan() {
    awk -v nsub="$2" -v nchan="$3" -v nant="$4" -v gpu="$GPU_MB" -v host="$HOST_MB" 'BEGIN {
        nbl = nant * (nant - 1) / 2; nband = 2 * nsub
        vis = 8 * nsub * nbl * 4 * nchan                     # bytes, one copy
        perwin = nant * nband * nchan                         # spectral values a window
        mem = 0.8 * gpu * 1048576 - 2 * vis - 512 * 1048576   # bytes left for the per-value buffers
        if (vis > 0.2 * host * 1048576 || mem < 96 * perwin * 16) {
            printf "0 - visibilities %.1f GB, %.1f GB at 16 windows\n", 2 * vis / 1e9, (2 * vis + 96 * perwin * 16) / 1e9; exit }
        n = 10000
        m = int(mem / (96 * perwin)); if (m < n) n = m
        d = int(1e8 / perwin); if (d < n) n = d
        w = int(2e10 / (nbl * 4 * nchan * nsub)); if (w < n) n = w
        if (n < 16) n = 16
        reps = nbl * 4 * nchan * nsub * n > 2e10 ? "few" : "-"
        printf "%d %s -\n", n, reps }'
}

for nbit in $NBITS; do
    for nsub in $NSUBS; do
        for nchan in $NCHANS; do
            for nant in $NANTS; do
                read -r nwin reps note < <(plan "$nbit" "$nsub" "$nchan" "$nant")
                name="nbit$nbit-nsub$nsub-nchan$nchan-nant$nant"
                if [ "$nwin" = 0 ]; then
                    echo "$name: skipped ($note)"
                    echo "$nbit,$nsub,$nchan,$nant,,,skip,,\"$note\"" >> "$RUNS"
                    continue
                fi
                ropt=(); nreps=20; [ "$reps" = few ] && { ropt=(--reps 5 --warmup 1); nreps=5; }
                t0=$(date +%s)
                "$FUSEBENCH" --input packed --xmac --rotator double --fpb sweep --ept sweep --check 4 \
                    --nbit "$nbit" --nband $((2 * nsub)) --nchan "$nchan" --nant "$nant" --nwin "$nwin" \
                    --csv "$OUT/raw.csv" "${ropt[@]}" "${EXTRA[@]}" > "$OUT/$name.log" 2>&1
                rc=$?
                status=ok
                if grep -q "cudaErrorMemoryAllocation\|out of memory" "$OUT/$name.log"; then status=oom
                elif [ $rc != 0 ] || grep -q "FAIL" "$OUT/$name.log"; then status=fail; fi
                [ $status = ok ] || fail=1
                echo "$nbit,$nsub,$nchan,$nant,$nwin,$nreps,$status,$rc," >> "$RUNS"
                printf "%-34s nwin %5d  %-4s exit %d  %4d s\n" "$name" "$nwin" "$status" $rc $(( $(date +%s) - t0 ))
            done
        done
    done
done

python3 -I matrix_summary.py "$RUNS" "$OUT/raw.csv" --host "$(hostname -s)" \
    --commit "$(git log -1 --format=%h 2>/dev/null)" --date "$STAMP" --power "${POWER:-}" --clock "${CLOCK:-}" \
    > "$OUT/summary.csv"
cp "$OUT/summary.csv" "results/$NAME.csv"
echo
python3 -I -c '
import csv, sys
cols = "nbit nsub nchan nant nwin status worst_ratio chain_layout_tensor chain_staged staged_over_best gain_vs_gpudifx".split()
print(" ".join(f"{c:>12}" for c in cols))
for r in csv.DictReader(open(sys.argv[1])):
    print(" ".join(f"{r[c]:>12}" for c in cols))' "$OUT/summary.csv"
echo "Results in $OUT (summary.csv; also results/$NAME.csv for the repository)"
[ $fail = 0 ] || { echo "SOME RUNS FAILED: see runs.csv and the logs" >&2; exit 1; }
