# phasecentrebench: the shift after the XMAC (RTX 2070 SUPER, A100)

The benchmark (`phasecentrebench.cu`, [README](../README.md)) runs:
1. The staged tensor XMAC once per piece, writing each piece's fine visibilities.
2. The delays per (piece, baseline, centre), with mpifxcorr's decorrelation estimate.
3. The shift and average for every centre, in three versions:
   - `naive`: a thread per output and a sincos per fine channel;
   - `simt`: an FP32 matrix product;
   - `tensor`: an FP16 WMMA matrix product.

## The optimisation round (2026-11-22, RTX 2070 SUPER)

The backlog's items in turn, each measured on the 2070 (150 W, 1500 MHz) and kept only if it paid
for itself.  The resulting sweep (`results-local/ar313-adeller-centres-20261122-000946`, every
run passing), as the tables below:

| sweep | value | windows a piece | xmac | delays | naive | simt | tensor | sh/xmac | work | total |
|---|---|---|---|---|---|---|---|---|---|---|
| centres | 1 | 100 | 0.0707 | 0.0004 | 0.0086 | 0.0185 | 0.0068 | 0.10 | 0.01 | 0.0779 |
| centres | 10 | 100 | 0.0711 | 0.0005 | 0.0789 | 0.0187 | 0.0071 | 0.10 | 0.10 | 0.0788 |
| centres | **100** | 100 | 0.0704 | 0.0005 | 0.7775 | 0.0708 | 0.0254 | 0.36 | 1.00 | **0.0963** |
| centres | 1000 | 100 | 0.0702 | 0.0032 | 7.6998 | 0.5617 | 0.1959 | 2.79 | 10.00 | 0.2693 |
| pieces | 2 | 50 | 0.0862 | 0.0009 | - | 0.1345 | 0.0477 | 0.55 | 2.00 | 0.1348 |
| pieces | 4 | 25 | 0.1175 | 0.0015 | - | 0.2652 | 0.0910 | 0.77 | 4.00 | 0.2101 |
| pieces | 10 | 10 | 0.2298 | 0.0034 | - | 0.6505 | 0.2242 | 0.98 | 10.00 | 0.4575 |
| avg | 8 | 100 | 0.0709 | 0.0006 | - | 0.1573 | 0.1074 | 1.51 | 1.00 | 0.1789 |
| avg | 16 | 100 | 0.0706 | 0.0006 | - | 0.1061 | 0.0579 | 0.82 | 1.00 | 0.1290 |
| avg | 32 | 100 | 0.0710 | 0.0006 | - | 0.0811 | 0.0346 | 0.49 | 1.00 | 0.1062 |
| avg | 128 | 100 | 0.0713 | 0.0006 | - | 0.0646 | 0.0206 | 0.29 | 1.00 | 0.0925 |
| avg | 256 | 100 | 0.0710 | 0.0006 | - | 0.0618 | 0.0179 | 0.25 | 1.00 | 0.0894 |
| nchan | 1024 (avg 8, 800 windows) | 800 | 0.0583 | 0.0006 | - | 0.0193 | 0.0134 | 0.23 | 0.12 | 0.0722 |
| nchan | 65536 (avg 512, 13 windows) | 13 | 0.1812 | 0.0005 | - | 0.4555 | 0.1424 | 0.79 | 7.69 | 0.3241 |
| nant | 32 | 100 | 0.3832 | 0.0036 | - | 0.7662 | 0.2721 | 0.71 | 1.00 | 0.6588 |

**Against the first version (below): the tensor shift 1.4-3.0x faster, the XMAC 0-20%.**
- Tensor shift: 2.0-3.0x with 100 or more centres in one piece: 0.0598 -> 0.0254 ms per ms at the
  default shape (2.4x), 0.4985 -> 0.1959 at 1000 centres (2.5x), 0.6535 -> 0.2721 at 32 stations
  (2.4x), 0.3235 -> 0.1074 at 8 channels averaged (3.0x).  1.4-1.5x at 1-10 centres (where it was
  cheap already), 1.7-2.0x with 2-10 pieces, 1.7x at 65536 channels.
- With 100 or more centres it runs at 0.9-1.29 Tcmac/s: 23-34% of the 2070's tensor-core rate
  (FP16 in, FP32 sums: about 3.8 Tcmac/s at 1500 MHz).
- XMAC: 5-6% with one piece (the zeroing skipped, below), 10-20% with 2-10 pieces, 65536 channels
  (13 windows) or 32 stations.
- Whole chain at 100 centres: 0.135 -> 0.096 ms per ms, +37% over the XMAC alone (was +80%).
  10 centres: +11%.  1000 centres: 3.8x the XMAC (was 7.7x).
- **Shift / XMAC ≈ 0.25-0.36 x centres / windows per piece** for averages of 64 or more (was
  0.5-0.8).

What each step did, at the default shape (tensor shift, ms per subint of 25.6 ms; the first
version 1.55):

| step | default | 1000 centres | kept |
|---|---|---|---|
| 32 centres a block instead of 64 (101 registers, 28 KB: 2 blocks an SM instead of 1) | 1.22 | 10.0 | yes |
| the tensor kernel's shared memory sized for itself (it had the SIMT kernel's), and `__launch_bounds__(256, 3)`: 80 registers, 21 KB, 3 blocks an SM; 73% achieved occupancy | 1.00 | 8.30 | yes |
| store each output once instead of zeroing it and adding (every output has one owner; the pieces are summed in registers) | 0.69 | 5.35 | yes |
| inner phasors by a short recurrence (a thread's 4 channels of one centre: 1 sincos and 3 multiplies instead of 4 sincos) | 0.67 | 5.11 | yes (3-4%, no more code) |
| the centre tiles fastest in the grid (blocks sharing fine visibilities together in L2) | 0.67 | 5.17 | no (no gain) |
| outer phasors by recurrence (tried on the first version) | slower | slower | no |

- **Nsight Compute after the steps** (`--clock-control none`, 1500 MHz): 74% occupancy, 55% of the
  SM's compute throughput, 42% of DRAM bandwidth, the stalls spread over fixed latency, global
  loads, the math pipes and barriers; 0.56 instructions issued a cycle per scheduler.  About 24%
  of the instructions are the outer phasors, 16% were the inner phasors (before their
  recurrence), 6% the loads of V; the MMAs themselves are few.  No single limit is left.
- **Not tried: double buffering** of the chunks' loads.  With 3 blocks an SM one block's loads
  overlap the others' products already, there are only A / 32 chunks a piece (2 at 64 averaged),
  and a second buffer would cost the third block.

**The XMAC with the pieces in one launch** (the backlog item: each piece to its own slot, grid z =
pieces x window chunks): bitwise the same visibilities, but only 0-1% faster than a launch a
piece.  The short pieces' cost is not the launches: it is the 16-window MMA tile (10 windows pay
for 16) and the output, a full set of fine visibilities a piece.  Reverted.  What it showed instead:
**the staged XMAC zeroed its output first even when nothing needed it.**  Unsplit (one chunk of
windows a block) each block stores its sums, and with a complete map every visibility is written,
so `common/xmac.cu` now skips the zeroing then (`XmacMap::complete`).  XMAC: 6% faster with one
piece, 14% with 4, 16% with 10.  xmacbench (regular, `--messy`, `--invalid`, 32 stations) and
fusebench `--xmac` checks pass.

**The shift in the XMAC's epilogue: ruled out.**  A staged XMAC block holds 8 (at most 16)
channels of sums, a shift output averages A >= 64.  So the epilogue could only form partial sums
over its channels and add them atomically into every centre's outputs: nfreq x baselines x 4 x
nchan / 8 x centres complex atomic adds a piece.  Those adds alone (a one-off kernel doing nothing else,
a thread per (baseline-product, centre, 8-channel group), default shape): 2.5 ms at 10 centres, 113 ms at 100, 1363 ms at 1000 with 8
channels a block (0.37 / 58 / 687 with 16), against 0.19 / 0.67 / 5.2 ms for the whole shift from
DRAM.  A block holding all A channels (no atomics) would need 8x the XMAC's accumulators.  The
fine visibilities' round trip through DRAM is the right design.

**Lower sideband** (`--sideband L` or `mixed`): mpifxcorr's rotator puts an LSB band's channel c
at LO - (nchan - 1 - c) df, the same as an upper sideband from LO - (nchan - 1) df, so the
kernels only take a base frequency per band (`cBase`); the decorrelation estimate keeps the LO
(as mpifxcorr's).  Every check passes for USB, LSB and mixed.  As for USB, the convention is
mpifxcorr's, tested for consistency, not against the sky.

**The lag-window term of mpifxcorr's decorrelation estimate is right** (`lagwindow.py`): in a
time-domain simulation (white noise, a fractional delay d between two stations, the same
rectangular N-sample windows, the shift to d, averaged) the coherence is 1 - |d| / N to within
5e-4, for d from 1% to 50% of the window, at 512 and 128 channels.  So the gap between the
estimate and the coherence phasecentrebench measures (0.949 against 0.997) is that term, which
its per-channel synthetic data does not have.  (For windows that overlap or taper, as WOLA, the
term would differ.)

## The first version (2026-11-21)

RTX 2070 SUPER, ar313, at the 150 W cap and the 1500 MHz clock lock (2026-11-21, every run
passing: `results-local/ar313-adeller-centres-20261121-230101`, `run-centres.sh`).

The default shape:
- 10 stations, 4 dual-pol 32 MHz bands from 1400 MHz;
- 8192 channels (3.9 kHz);
- a 25.6 ms subint of 100 windows in one piece;
- 100 centres, |Δτ| up to 40 µs;
- 64 channels averaged (250 kHz out).

Each sweep below changes one parameter from that.  The table is in ms per ms of data:
- xmac: the XMAC over its pieces, including zeroing each piece's output;
- delays: the delays kernel;
- naive / simt / tensor: the shift versions;
- sh/xmac: the tensor shift over the XMAC;
- work: that ratio's arithmetic, centres / windows per piece;
- total: xmac + delays + tensor.

| sweep | value | windows a piece | xmac | delays | naive | simt | tensor | sh/xmac | work | total |
|---|---|---|---|---|---|---|---|---|---|---|
| centres | 1 | 100 | 0.0750 | 0.0004 | 0.0088 | 0.0192 | 0.0094 | 0.13 | 0.01 | 0.0848 |
| centres | 10 | 100 | 0.0748 | 0.0006 | 0.0796 | 0.0200 | 0.0107 | 0.14 | 0.10 | 0.0861 |
| centres | **100** | 100 | 0.0748 | 0.0006 | 0.7780 | 0.0787 | 0.0598 | 0.80 | 1.00 | **0.1352** |
| centres | 1000 | 100 | 0.0744 | 0.0034 | 7.8510 | 0.6334 | 0.4985 | 6.70 | 10.00 | 0.5763 |
| pieces | 2 | 50 | 0.0950 | 0.0009 | - | 0.1465 | 0.0952 | 1.00 | 2.00 | 0.1911 |
| pieces | 4 | 25 | 0.1344 | 0.0016 | - | 0.2787 | 0.1693 | 1.26 | 4.00 | 0.3053 |
| pieces | 10 | 10 | 0.2755 | 0.0036 | - | 0.6716 | 0.3875 | 1.41 | 10.00 | 0.6665 |
| avg | 8 | 100 | 0.0745 | 0.0006 | - | 0.2585 | 0.3235 | 4.34 | 1.00 | 0.3986 |
| avg | 16 | 100 | 0.0749 | 0.0006 | - | 0.1535 | 0.1722 | 2.30 | 1.00 | 0.2476 |
| avg | 32 | 100 | 0.0747 | 0.0006 | - | 0.1073 | 0.0973 | 1.30 | 1.00 | 0.1726 |
| avg | 128 | 100 | 0.0747 | 0.0006 | - | 0.0717 | 0.0432 | 0.58 | 1.00 | 0.1184 |
| avg | 256 | 100 | 0.0748 | 0.0006 | - | 0.0652 | 0.0351 | 0.47 | 1.00 | 0.1105 |
| nchan | 1024 (avg 8, 800 windows) | 800 | 0.0585 | 0.0006 | - | 0.0330 | 0.0403 | 0.69 | 0.12 | 0.0994 |
| nchan | 65536 (avg 512, 13 windows) | 13 | 0.2164 | 0.0005 | - | 0.4700 | 0.2371 | 1.10 | 7.69 | 0.4541 |
| nant | 32 | 100 | 0.4318 | 0.0037 | - | 0.8596 | 0.6535 | 1.51 | 1.00 | 1.0890 |

## What it shows

- **The matrix product is the way to do it.**
  - At the default shape the tensor shift is 13x the naive port of mpifxcorr's loop: 384 against
    29 Gcmac/s, where a cmac is one fine visibility for one centre.
  - It reaches 460-730 Gcmac/s with long averages or many centres.  The staged XMAC runs at
    about 310 here.
  - That is still only 10-20% of the 2070's tensor-core rate (FP16 in, FP32 sums, at 1500 MHz).
    The overheads per chunk and per piece (loads, inner phasors, outer phasors) dominate, not the
    MMAs.
  - The SIMT FP32 version is 1.3-2x slower, except at averages of 8-16 (below).
- **Shift / XMAC ≈ 0.5-0.8 x centres / windows per piece** at 8k channels and averages of 64 or
  more.
  - At the default 100 centres and 100 windows, the chain costs 0.135 ms per ms against the
    XMAC's 0.075 (+80%).
  - At 10 centres it costs +15%.  At 1000 centres it is 7.7x the XMAC alone.
- **Short pieces hit the XMAC itself, as the BACKLOG's "Watch" item predicted.**
  - Per piece it zeroes and writes a full set of fine visibilities (44 MB here).
  - At 50 windows a piece the XMAC costs 1.27x, at 25 1.8x, at 10 3.7x.
  - The shift's cost tracks the work (centres x pieces), so short pieces cost twice.
- **Short averages favour the CUDA cores**, but only matter with many centres, which does not
  happen in practice.
  - The per-output work (the outer phasor, the product through shared memory, the running sums,
    the write-out) outweighs the product when only 8 or 16 channels are averaged, and the tensor
    version then loses to the SIMT one.
  - With few centres it is cheap all the same: at 8 channels averaged the shift costs 0.037 ms
    per ms for 1 centre and 0.045 for 10 (the XMAC 0.075); at 16 averaged, 0.021 and 0.025.
- **Tried and dropped: the outer phasor by recurrence.**
  - Each thread took one centre and 8-16 consecutive output channels, with one sincos per run
    instead of one per element, and padded centres skipped.
  - It was 4-10% slower everywhere, also at short averages: 1.69 against 1.54 ms at the default
    shape, 9.1 against 8.2 at 8 averaged.
  - Nsight Compute (`--clock-control none`, 1500 MHz): 4% more instructions.  The kernel is
    latency-bound, not sincos-bound: 25% occupancy (151 registers at 64 centres a block, one
    block an SM), 8.9 cycles per issued instruction, 22% of the SM's compute throughput, 32% of
    DRAM bandwidth.
- **Long channels are no problem.**
  - 65536 channels at 512 averaged: shift / XMAC is 1.10 at a work ratio of 7.7, 0.14 per unit of
    work.  The tensor shift runs at 730 Gcmac/s there, the best of any shape.
  - The XMAC with 13 windows a subint is short-piece-bound itself (0.216 ms per ms against 0.075
    at 8k).
- **One centre costs 0.009 ms per ms**, padded to 32 centres a block: nothing next to the XMAC.
- **32 stations: shift / XMAC 1.5 at work 1.**  The XMAC is more efficient with more stations
  (569 Gcmac/s); the shift is not.

## A100-SXM4-80GB (OzSTAR gina2), after the optimisation round

SLURM job 21370182, commit 7bcfe24, every run passing (`results-local/gina2-centres-20261122-115100`).
Columns as above, ms per ms of data:

| sweep | value | windows a piece | xmac | delays | naive | simt | tensor | sh/xmac | work | total |
|---|---|---|---|---|---|---|---|---|---|---|
| centres | 1 | 100 | 0.0228 | 0.0004 | 0.0025 | 0.0084 | 0.0029 | 0.13 | 0.01 | 0.0261 |
| centres | 10 | 100 | 0.0228 | 0.0005 | 0.0187 | 0.0085 | 0.0031 | 0.13 | 0.10 | 0.0264 |
| centres | **100** | 100 | 0.0224 | 0.0005 | 0.1858 | 0.0295 | 0.0096 | 0.43 | 1.00 | **0.0325** |
| centres | 1000 | 100 | 0.0209 | 0.0005 | 1.7302 | 0.2083 | 0.0643 | 3.08 | 10.00 | 0.0857 |
| pieces | 2 | 50 | 0.0291 | 0.0005 | - | 0.0508 | 0.0161 | 0.55 | 2.00 | 0.0457 |
| pieces | 4 | 25 | 0.0461 | 0.0007 | - | 0.0996 | 0.0301 | 0.65 | 4.00 | 0.0769 |
| pieces | 10 | 10 | 0.1018 | 0.0011 | - | 0.2460 | 0.0723 | 0.71 | 10.00 | 0.1753 |
| avg | 8 | 100 | 0.0209 | 0.0004 | - | 0.0561 | 0.0372 | 1.78 | 1.00 | 0.0585 |
| avg | 16 | 100 | 0.0208 | 0.0004 | - | 0.0395 | 0.0205 | 0.99 | 1.00 | 0.0418 |
| avg | 32 | 100 | 0.0209 | 0.0004 | - | 0.0309 | 0.0120 | 0.58 | 1.00 | 0.0333 |
| avg | 128 | 100 | 0.0209 | 0.0004 | - | 0.0250 | 0.0076 | 0.36 | 1.00 | 0.0289 |
| avg | 256 | 100 | 0.0208 | 0.0004 | - | 0.0251 | 0.0071 | 0.34 | 1.00 | 0.0284 |
| nchan | 1024 (avg 8, 800 windows) | 800 | 0.0146 | 0.0004 | - | 0.0073 | 0.0049 | 0.34 | 0.12 | 0.0199 |
| nchan | 65536 (avg 512, 13 windows) | 13 | 0.0764 | 0.0004 | - | 0.1947 | 0.0485 | 0.63 | 7.69 | 0.1254 |
| nant | 32 | 100 | 0.1674 | 0.0006 | - | 0.3250 | 0.0943 | 0.56 | 1.00 | 0.2622 |

**Against the A100's first version (below): the tensor shift 1.1-2.1x faster, less than the
2070's 1.4-3.0x.**
- Tensor shift: 1.55x at the default shape (0.0149 -> 0.0096 ms per ms), 1.8x at 1000 centres,
  1.5x at 32 stations, 1.8-2.1x at averages of 8-32 and at 1024 channels, 1.3x at averages of
  128-256.  1.1-1.2x at 1-10 centres.
- The gain shrinks as the matrix product's share grows: 1.54x / 1.30x / 1.17x / 1.08x at 1 / 2 /
  4 / 10 pieces, 1.08x at 65536 channels (13 windows, 512 averaged).  So on the A100 the gain is
  mostly the per-output work (each output stored once); the matrix product itself barely moved.
  On the 2070 it gained 1.7-2.0x with 2-10 pieces.
- It runs at 2.4-3.7 Tcmac/s with 100 or more centres (was 1.5-3.4): 6-9% of the A100's FP16
  tensor rate (about 39 Tcmac/s), against 23-34% on the 2070.
- Whole chain at 100 centres: 0.0325 ms per ms, +45% over the XMAC alone (was +65%).  10 centres:
  +16%.  1000 centres: 4.1x the XMAC (was 5.8x).
- **Shift / XMAC ≈ 0.3-0.45 x centres / windows per piece** at 10 stations, averages of 64 or more
  (0.56 at 32 stations); was 0.5-0.65.
- Against the 2070 after the same round: the tensor shift 2.3-3.1x faster, the XMAC 2.3-4.0x.
- XMAC: 4-9% faster than the first A100 run in most rows (13% at 256 averaged, 16% at 65536
  channels), 4% slower at 32 stations.  Its time for the same shape varies by 9% between runs
  (0.0208-0.0228 here), so only the trend is clear: smaller than the 2070's 5-20%.  (`xmac/run-staged.sh` in the same job is
  unchanged from the 18 November runs within noise: its small shapes split the windows over
  chunks, which still zero, and at its large ones the zeroing is a small share of a long XMAC.)

**Why the A100 gains less: occupancy, by registers.**  Built for sm_80 the kernel is the same 80
registers and 21 KB, so 3 blocks of 256 threads an SM: 24 warps of the A100's 64 (37.5%),
against 24 of the 2070's 32 (75%).  Shared memory would allow 7 blocks; registers allow 3.
- Four blocks (`--blocks 4`: `__launch_bounds__(256, 4)`) compiles to 64 registers with a 32-byte
  spill (sm_75 and sm_80).  On the 2070 that is 3-12% slower (default 899 -> 842 Gcmac/s, 1000
  centres 1159 -> 1125, 8 averaged 218 -> 192): shared memory (64 KB an SM on Turing) still
  allows only 3 blocks there (ncu: occupancy limit 3 by shared memory, 4 by registers; 73%
  achieved either way), so it is the registers' cost without the occupancy.  On the A100 it would
  be 32 of 64 warps: to be measured (`common/scripts/ncu-ozstar.sbatch`, the pc-* cases).  Five
  blocks spills 164 bytes.
- The A100 also has 16x the tensor rate of its FP32 cores (the 2070 about 4x), so the kernel's
  FP32 work (phasors, conversion to FP16, the outer phasor: about a quarter of the instructions
  on the 2070) weighs more there.  Not profiled on the A100.

## A100-SXM4-80GB (OzSTAR gina3), the first version

SLURM job 21341101, commit 7fc7d05 (the same kernels as the 2070 run), every run passing
(`results-local/gina3-centres-20261121-233134`).  Columns as above, ms per ms of data:

| sweep | value | windows a piece | xmac | delays | naive | simt | tensor | sh/xmac | work | total |
|---|---|---|---|---|---|---|---|---|---|---|
| centres | 1 | 100 | 0.0238 | 0.0004 | 0.0026 | 0.0088 | 0.0032 | 0.14 | 0.01 | 0.0274 |
| centres | 10 | 100 | 0.0238 | 0.0005 | 0.0189 | 0.0091 | 0.0036 | 0.15 | 0.10 | 0.0278 |
| centres | **100** | 100 | 0.0238 | 0.0005 | 0.1918 | 0.0327 | 0.0149 | 0.63 | 1.00 | **0.0393** |
| centres | 1000 | 100 | 0.0217 | 0.0005 | 1.7414 | 0.2304 | 0.1149 | 5.29 | 10.00 | 0.1371 |
| pieces | 2 | 50 | 0.0310 | 0.0005 | - | 0.0562 | 0.0209 | 0.68 | 2.00 | 0.0524 |
| pieces | 4 | 25 | 0.0499 | 0.0006 | - | 0.1053 | 0.0351 | 0.70 | 4.00 | 0.0856 |
| pieces | 10 | 10 | 0.1115 | 0.0011 | - | 0.2549 | 0.0778 | 0.70 | 10.00 | 0.1904 |
| avg | 8 | 100 | 0.0217 | 0.0004 | - | 0.0878 | 0.0698 | 3.21 | 1.00 | 0.0920 |
| avg | 16 | 100 | 0.0217 | 0.0004 | - | 0.0539 | 0.0381 | 1.76 | 1.00 | 0.0603 |
| avg | 32 | 100 | 0.0217 | 0.0004 | - | 0.0390 | 0.0215 | 0.99 | 1.00 | 0.0437 |
| avg | 128 | 100 | 0.0217 | 0.0004 | - | 0.0278 | 0.0101 | 0.47 | 1.00 | 0.0323 |
| avg | 256 | 100 | 0.0238 | 0.0005 | - | 0.0292 | 0.0094 | 0.39 | 1.00 | 0.0337 |
| nchan | 1024 (avg 8, 800 windows) | 800 | 0.0157 | 0.0005 | - | 0.0126 | 0.0103 | 0.66 | 0.12 | 0.0264 |
| nchan | 65536 (avg 512, 13 windows) | 13 | 0.0912 | 0.0005 | - | 0.2020 | 0.0524 | 0.57 | 7.69 | 0.1440 |
| nant | 32 | 100 | 0.1613 | 0.0005 | - | 0.3205 | 0.1454 | 0.90 | 1.00 | 0.3072 |

(The XMAC's time for the same shape varies by 9% between runs here: 0.0217-0.0238.)

- **Faster than the 2070 by 2.4-3.7x for the XMAC, 2.9-5.0x for the tensor shift, 2.2-2.9x for
  the SIMT one.**
  - The tensor shift runs at 1.5-3.4 Tcmac/s: 1.5 at the default, 2.0 at 1000 centres, 3.4 at
    65536 channels.  The XMAC runs at about 1.0.
  - That is 4-9% of the A100's FP16 tensor rate, so the shift is again far from its arithmetic
    limit.
- **Shift / XMAC ≈ 0.5-0.65 x centres / windows per piece**, a little below the 2070's 0.5-0.8.
  - 100 centres at 100 windows: 0.039 ms per ms for the whole chain, against 0.024 for the
    XMAC alone (+65%).
  - 10 centres: +17%.  1000 centres: 5.8x.
- **The tensor version gains more over the others than on the 2070**: 13x the naive port at the
  default shape (as on the 2070), and 2.2x the SIMT version (1.3x on the 2070).  At 65536
  channels it is 3.9x the SIMT version.
- **Short pieces cost the XMAC more on the A100**: 1.4x at 50 windows a piece, 2.3x at 25, 5.1x
  at 10 (2070: 1.27x, 1.8x, 3.7x).  The XMAC alone is faster, so its fixed cost per piece (zeroing
  and writing the fine visibilities) weighs more.  The shift scales better than the work: 10
  pieces cost 5.6x one piece.
- **Short averages**, as on the 2070: SIMT and tensor close at 8-16, the tensor version ahead from
  32 on.
- **The checks** match the 2070's: tensor 9e-5 rms against the double shift, 3e-5 for the
  source centre.

## The checks

Every version, every run:
- **Against the double-precision shift of the same fine visibilities**, on 128 random (frequency,
  baseline, centre, output channel) picks, all 4 products.
  - FP32 versions: rms 2-12e-7.
  - Tensor: 3-17e-5 (the FP16 inputs).
- **The source's centre against the exact coherent sum.**
  - The simulated sky has one source, at centre 1: unit-modulus signals, each station delayed by
    D_a(t) per window.
  - The coherent sum after the shift is exact in double from the delay model.
  - FP32 versions: rms 3e-6 (the FP16 station spectra).  Tensor: 3e-5.
- **The delays and the decorrelation estimate against the host**: Δτ to 1e-20 s.

## Caveats

- **The sign convention is mpifxcorr's**: rotator e^{+2πiΔτ f}, Δτ = D_ant2 − D_ant1.
  - The simulated data is made to match it.  So the source check tests the GPU against mpifxcorr's
    formula, not the formula against the sky.
  - The same holds for the (1 + rate) factor: the data applies it as mpifxcorr does.
- **The decorrelation estimate is lower than the coherence measured** (default shape: 0.954
  against 0.996).
  - mpifxcorr's estimate includes the lag-window term, 1 − |Δτ| / (nchan / bw).
  - The simulated spectra are made per channel, so they do not suffer it.  Real station
    processing (an FFT of delayed time samples) would.
  - A check of that term needs time-domain data through a station stage.
- **Upper sideband only.**  mpifxcorr's LSB channel frequencies are not implemented yet.
