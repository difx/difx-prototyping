# bench-bin.sh - sourced by the run-*.sh scripts: FUSEBENCH and XMACBENCH, the binaries built for
# the GPU this runs on (build-sm<compute capability>/, see the Makefile), so that jobs on different
# GPUs sharing this directory never run each other's.  Falls back to ./fusebench and ./xmacbench
# (the last build's links), e.g. for a build with SM=90a.
BENCH_SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
FUSEBENCH=./fusebench XMACBENCH=./xmacbench
[ -n "$BENCH_SM" ] && [ -x "build-sm$BENCH_SM/fusebench" ] && FUSEBENCH="build-sm$BENCH_SM/fusebench"
[ -n "$BENCH_SM" ] && [ -x "build-sm$BENCH_SM/xmacbench" ] && XMACBENCH="build-sm$BENCH_SM/xmacbench"
