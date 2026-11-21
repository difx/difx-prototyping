# phasecentrebench: the shift after the XMAC, first results (RTX 2070 SUPER, A100)

The benchmark (`phasecentrebench.cu`, [README](../README.md)) runs:
1. The staged tensor XMAC once per piece, writing each piece's fine visibilities.
2. The delays per (piece, baseline, centre), with mpifxcorr's decorrelation estimate.
3. The shift and average for every centre, in three versions:
   - `naive`: a thread per output and a sincos per fine channel;
   - `simt`: an FP32 matrix product;
   - `tensor`: an FP16 WMMA matrix product.

## The run

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

## A100-SXM4-80GB (OzSTAR gina3)

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
