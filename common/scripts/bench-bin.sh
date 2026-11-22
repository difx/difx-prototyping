# bench-bin.sh - sourced by the run-*.sh scripts: FUSEBENCH, XMACBENCH, TWOSTAGEBENCH,
# PHASECENTREBENCH and BINBENCH, the binaries built for the GPU this runs on (<repo>/build-sm<compute
# capability>/, see the Makefile), so that jobs on different GPUs sharing a checkout never run each
# other's.  Falls back to the last build's links (fft-fusion/fusebench, xmac/xmacbench,
# two-stage/twostagebench, phase-centres/phasecentrebench, pulsar-binning/binbench), e.g. for a build with SM=90a.  REPO is
# the checkout's root.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BENCH_SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
benchBin() {   # benchBin <name> <sub-project>
    if [ -n "$BENCH_SM" ] && [ -x "$REPO/build-sm$BENCH_SM/$1" ]; then echo "$REPO/build-sm$BENCH_SM/$1"
    else echo "$REPO/$2/$1"; fi
}
FUSEBENCH=$(benchBin fusebench fft-fusion)
XMACBENCH=$(benchBin xmacbench xmac)
TWOSTAGEBENCH=$(benchBin twostagebench two-stage)
PHASECENTREBENCH=$(benchBin phasecentrebench phase-centres)
BINBENCH=$(benchBin binbench pulsar-binning)
