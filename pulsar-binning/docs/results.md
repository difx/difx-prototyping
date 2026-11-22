# binbench results

## The bins and the naive port (2026-11-22, RTX 2070 SUPER, 150 W / 1500 MHz)

`binbench` (README.md): 10 stations, 4 dual-pol 32 MHz bands from 1400 MHz, a 25.6 ms subint,
DM 30, the phase at the window's start (mpifxcorr's convention), the pulse (a Gaussian, FWHM 0.03
turns, at phase 0.5) in the subint's middle.  ms per subint; every case passing its checks (the bins
against mpifxcorr's rule in double: no differences; the naive port against the host's double sums
of the same FP16 spectra; the sum of its bins against the unbinned XMAC's sums).

| P | channels (window) | bins | empty bins | XMAC (unbinned) | bins kernel | naive | naive / XMAC |
|---|---|---|---|---|---|---|---|
| 1.5 ms | 128 (4 µs) | gate | 0/2 | 1.219 | 0.039 | 4.66 | 3.8 |
| | | 16 separate | 0/16 | 1.205 | 0.063 | 8.29 | 6.9 |
| | | 16 scrunched | 0/16 | 1.208 | 0.063 | 4.73 | 3.9 |
| | | 128 separate | 0/128 | 1.210 | 0.105 | 226.9 | 188 |
| | 1024 (32 µs) | gate | 0/2 | 1.451 | 0.037 | 9.02 | 6.2 |
| | | 16 separate | 0/16 | 1.485 | 0.049 | 23.7 | 16.0 |
| | | 16 scrunched | 0/16 | 1.462 | 0.050 | 9.10 | 6.2 |
| | | 128 separate | 0/128 | 1.450 | 0.084 | 138.9 | 96 |
| 500 ms | 128 (4 µs) | gate | 1/2 | 1.212 | 0.038 | 4.70 | 3.9 |
| | | 16 separate | 2/16 | 1.215 | 0.067 | 6.52 | 5.4 |
| | | 16 scrunched | 2/16 | 1.220 | 0.067 | 4.77 | 3.9 |
| | | 128 separate | 118/128 | 1.206 | 0.086 | 6.65 | 5.5 |
| | 1024 (32 µs) | gate | 1/2 | 1.495 | 0.038 | 8.92 | 6.0 |
| | | 16 separate | 2/16 | 1.499 | 0.065 | 11.3 | 7.5 |
| | | 16 scrunched | 2/16 | 1.481 | 0.064 | 8.86 | 6.0 |
| | | 128 separate | 118/128 | 1.487 | 0.084 | 12.2 | 8.2 |

- **The naive port** (a thread per output, over the windows): 3.8-6.2x the unbinned tensor XMAC
  when one sum suffices (the gate, scrunched, or the 500 ms pulsar's long bins), which is about a
  CUDA-core XMAC's cost; 7-16x for 16 separate bins at 1.5 ms; 96-188x for 128 separate bins at
  1.5 ms, where the bin changes every 0.4-3 windows and each change adds the sum into global memory.
  The reference, not a candidate.
  - Dispersion is what hurts it: at DM 0 the 128-bin case (1.5 ms, 128 channels) takes 12.4 ms,
    not 227.  Without it a warp's channels change bin on the same window and their adds into
    global memory coalesce; at DM 30 adjacent channels' bins differ by 5.7 windows, so they do not.
- Also checked: `--phase mid` (the window's middle) and 32 stations (16 bins scrunched: naive
  21x the XMAC's 4.8 ms) pass.
- **The bins kernel**: 0.04-0.11 ms, 3-9% of the unbinned XMAC at 10 stations (less with more
  stations: the XMAC grows with the baselines, the bins do not).  More than "negligible": the
  binary search over the ends sets it (0.04 ms with 2 bins, 0.1 with 128).  A lookup table on the
  phase's top bits would make it flat.
- **The 500 ms pulsar**: a subint covers 5% of the period, so the gate's subint is all on-pulse,
  and 118 of the 128 bins are empty (2 of the 16).
- **The signal check** (each bin's mean power against the profile's, over every baseline): within
  1.6-5.9 "sigma", where the sigma treats the products as independent; they share each station's
  noise, so it is an underestimate.  Informational, not a pass/fail.

## Step 3: weights in the staged XMAC's stage (2026-11-22, RTX 2070 SUPER)

`xmacTensorStagedWeighted` (common/xmac.cu, a template flag: the unweighted kernel compiles as
before, its registers and speed unchanged in xmacbench at 4-64 stations): the copy into the stage
scales each (window, channel) by sqrt|w| of its bin (FP16), negative weights' signs go into the A
operand (per window, from nibbles the copy writes to shared memory).  Each block first classifies
its 16-window blocks from the bins: skipped (every weight 0), plain (every weight 1: copied as is),
scaled, or scaled with signs.  `stage`: one pass (the gate, scrunched output); `passes`: one pass
per bin with a one-hot table (separate bins).  ms per subint, against the unbinned XMAC; every
case passing (weights 0 and 1 exact; others against the host with the stage's FP16 rounding, and
3.8e-5 from the exact weights at 16 bins on - off).

| P | channels | bins | XMAC | weighted | / XMAC |
|---|---|---|---|---|---|
| 1.5 ms | 128 | gate (stage) | 1.210 | 0.553 | 0.46 |
| | | 16 scrunched (stage) | 1.205 | 1.621 | 1.35 |
| | | 16 separate (passes) | 1.190 | 7.11 | 6.0 |
| | | 128 separate (passes) | 1.209 | 50.1 | 41 |
| | 1024 | gate | 1.468 | 1.035 | 0.71 |
| | | 16 scrunched | 1.464 | 1.919 | 1.31 |
| | | 16 separate | 1.492 | 13.8 | 9.3 |
| | | 128 separate | 1.466 | 103.2 | 70 |
| 500 ms | 128 | gate | 1.207 | 1.376 | 1.14 |
| | | 16 scrunched | 1.218 | 1.380 | 1.13 |
| | | 16 separate | 1.217 | 3.97 | 3.3 |
| | | 128 separate | 1.210 | 19.6 | 16 |
| | 1024 | gate | 1.479 | 1.686 | 1.14 |
| | | 16 scrunched | 1.486 | 1.663 | 1.12 |
| | | 16 separate | 1.485 | 5.37 | 3.6 |
| | | 128 separate | 1.479 | 30.3 | 20 |

- **The gate and scrunched output: one pass, 0.46-1.35x the unbinned XMAC.**  The 1.5 ms gate
  skips the 16-window blocks with no on-pulse window in any of a block's channels (0.46x at 128
  channels, 0.71x at 1024).  With nothing to skip (the 500 ms pulsar's on-pulse subint) the cost
  is 1.12-1.14x: the classification pass over the bins, and the weighted kernel's few spilled
  registers (KS 2, at its 128).  On - off with negative weights at 1.5 ms: 1.31-1.35x.
- **What the overhead was** (ncu, the 500 ms gate): the first version executed 1.9x the
  unweighted kernel's instructions (a scale per value in FP32, the bins of every window re-read for
  the signs): 1.36-1.69x the XMAC.  Scaling in FP16, the plain mode (weights all 1: no scaling)
  and the signs from shared memory brought it to the figures above.
- **Separate bins by one pass a bin: 3.3-70x.**  Each pass skips the blocks without its bin, but
  at 1.5 ms (DM 30) a bin is in a large share of a block's 16-window blocks: 41x for 128 bins at
  128 channels, 70x at 1024 (a bin narrower than a window, in nearly every block).  The 500 ms
  pulsar's 118 empty bins still cost a pass each (the classification).  Not the way for many bins.

## Step 4: bins sorted per channel (2026-11-22, RTX 2070 SUPER)

Separate bins in one pass: each channel's windows sorted by bin into slots (a stable counting sort,
a warp per (frequency, channel), the bins computed on the fly), each bin padded to a multiple of 16
slots with the same slots across the XMAC's channel group (8 channels); the staged XMAC
(`xmacTensorStagedSorted`) runs a block per segment of a bin (up to 32 slot blocks) and writes its
sums to that bin once (atomically when the bin has several segments).  Every case checked against
the host (and the sum of its bins against the unbinned sums).

- **The XMAC over the slots is cheap; getting the data into slot order is not.**  From the usual
  spectra ([station][window][band][channel]) a permuted copy cost 17-28 ms (1.5 ms, 128 channels):
  a channel's windows are 4-byte gathers, a sector each, and dispersion (5.7 windows a channel)
  keeps neighbouring channels from sharing them.  The XMAC over the permuted slots was 1.5-2.5 ms.
- **Channel-major spectra** ([station][band][channel][window], a channel's windows contiguous; the
  station kernel would write them in pulsar mode, here a transpose, 1.4-1.5 ms, not counted): the
  permute 3.0-5.8 ms.  And `gather`: no permuted copy, the XMAC's copy gathers each slot's window
  through the slot table (`xmacTensorStagedSorted` with slotWin).
- **Weighted / sorted path / gather, ms per subint** (10 stations; the sort includes a
  device-to-host sync, for the permute's size):

| P | ch | bins | XMAC | passes | sorted: sort + permute + XMAC | gather: sort + XMAC | slots padding |
|---|---|---|---|---|---|---|---|
| 1.5 ms | 128 | 16 | 1.215 | 7.11 (6.0x) | 5.58 (4.6x): 0.35 + 3.02 + 2.19 | 3.93 (3.2x): 0.35 + 3.58 | 5% |
| | 128 | 128 | 1.202 | 50.1 (41x) | 7.76 (6.5x): 0.91 + 3.64 + 3.22 | 14.4 (12x): 0.91 + 13.5 | 28% |
| | 1024 | 16 | 1.470 | 13.8 (9.3x) | 6.69 (4.6x): 0.31 + 3.26 + 3.09 | 5.62 (3.8x): 0.30 + 5.33 | 22% |
| | 1024 | 128 | 1.466 | 103 (70x) | 19.7 (13x): 1.78 + 5.78 + 12.0 | 26.8 (18x): 1.78 + 25.1 | 156% |
| 500 ms | 128 | 16 | 1.204 | 4.00 (3.3x) | 5.49 (4.6x): 0.42 + 2.95 + 2.13 | 3.16 (2.6x): 0.42 + 2.75 | 3% |
| | 128 | 128 | 1.209 | 19.9 (16x) | 5.79 (4.8x): 0.59 + 2.88 + 2.32 | 3.51 (2.9x): 0.59 + 2.92 | 2% |
| | 1024 | 16 | 1.478 | 5.41 (3.7x) | 6.01 (4.1x): 0.31 + 3.04 + 2.63 | 3.72 (2.5x): 0.31 + 3.41 | 8% |
| | 1024 | 128 | 1.493 | 30.3 (20x) | 8.16 (5.5x): 0.58 + 3.03 + 4.56 | 5.88 (3.9x): 0.58 + 5.30 | 8% |

- **Gather wins when a bin's windows come in long runs** (16 bins; the 500 ms pulsar): 2.5-3.9x.
  With short runs (1.5 ms, 128 bins: 1.5-3 windows) its 4-byte loads do not coalesce, and the
  permuted copy wins (6.5x at 128 channels).
- **The output is a large part of what remains**: separate bins zero and write nbins x the
  visibilities every subint (128 bins x 1024 channels: 755 MB, twice), most of the 12 ms of the
  XMAC over the slots there, and the 500 ms pulsar's 118 empty bins are zeroed too.  So channel
  averaging in the XMAC and accumulating across subints (step 6) come next.
- **The padding** (16-slot blocks per bin, the most windows any channel of a group has): 2-8% for
  long runs, 28% at 1.5 ms / 128 bins / 128 channels, 156% at 1024 channels (6 windows a bin a
  subint, padded to 16).  Sorting over several subints would cut the last.
- **Writing a bin's sums at each bin change inside the window loop** (the first version of the
  sorted XMAC, and what splitting 16-window blocks at bin boundaries, step 5 of the plan, needs)
  put a second copy of the store code in the loop: 0.5-1.9 KB of spills at KS 2.  The segment
  grid (one bin a block, one store at the end) has none; step 5 was not built further: for long
  runs `gather` already does its job, and for short runs it is the passes' cost again.

## Step 6: the output, and the whole matrix (2026-11-22, RTX 2070 SUPER)

The binned XMAC modes (weighted, sorted, gather) can sum `--avg` adjacent channels into each output
channel and `--accumulate` into the output (an integration's sum, no zeroing each subint); when the
averaging covers a block's channels (8 here), the block's warps add their sums in shared memory
first and one warp writes them.  (A first layout of that buffer put a warp's 32 lanes in one bank:
40x slower.)

`run-bins.sh` (results-local/ar313-adeller-bins-20261122-140307, every run passing), ms per 25.6 ms
subint, the fastest binned version against the unbinned staged XMAC; `sorted` counts the sort, the
permute and its XMAC but not the channel-major spectra it starts from (1.4-1.5 ms here as a
transpose, nothing if the station kernel writes them):

| P | ch | bins | avg / acc | XMAC | bins kernel | naive | stage | passes | sorted | gather | best / XMAC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.5 ms | 128 | gate | 1 / no | 1.207 | 0.039 | 6.20 | 0.56 | | | | 0.46 |
| | | 16 scrunched | 1 / no | 1.194 | 0.062 | 6.17 | 1.63 | | | | 1.37 |
| | | 16 | 1 / no | 1.211 | 0.063 | 8.38 | | 7.36 | 5.54 | 3.98 | 3.28 (gather) |
| | | 128 | 1 / no | 1.195 | 0.104 | 225.6 | | 52.1 | 7.77 | 14.4 | 6.50 (sorted) |
| | | 128 | 8 / yes | 1.194 | 0.104 | 118.8 | | 50.7 | 7.25 | 13.7 | 6.07 (sorted) |
| | 1024 | gate | 1 / no | 1.474 | 0.037 | 8.57 | 1.04 | | | | 0.70 |
| | | 16 scrunched | 1 / no | 1.497 | 0.049 | 8.62 | 1.89 | | | | 1.26 |
| | | 16 | 1 / no | 1.479 | 0.049 | 23.4 | | 14.1 | 6.78 | 6.38 | 4.31 (gather) |
| | | 128 | 1 / no | 1.471 | 0.084 | 141.9 | | 105.6 | 21.0 | 36.6 | 14.3 (sorted) |
| 500 ms | 128 | gate | 1 / no | 1.207 | 0.039 | 6.08 | 1.37 | | | | 1.13 |
| | | 16 scrunched | 1 / no | 1.214 | 0.067 | 6.19 | 1.39 | | | | 1.14 |
| | | 16 | 1 / no | 1.213 | 0.067 | 6.53 | | 4.12 | 5.47 | 3.20 | 2.64 (gather) |
| | | 128 | 1 / no | 1.204 | 0.087 | 6.74 | | 21.1 | 5.79 | 3.53 | 2.93 (gather) |
| | 1024 | gate | 1 / no | 1.465 | 0.038 | 8.34 | 1.61 | | | | 1.10 |
| | | 16 scrunched | 1 / no | 1.486 | 0.064 | 8.44 | 1.61 | | | | 1.08 |
| | | 16 | 1 / no | 1.479 | 0.064 | 10.8 | | 5.69 | 6.05 | 4.16 | 2.81 (gather) |
| | | 128 | 1 / no | 1.471 | 0.084 | 11.9 | | 33.2 | 8.25 | 6.12 | 4.16 (gather) |
| | | 128 | 8 / yes | 1.465 | 0.084 | 11.2 | | 31.1 | 6.20 | 4.46 | 3.05 (gather) |

(The other avg 8 rows: within a few % of the avg 1 ones; table.txt has them all.)

- **Averaging and accumulating** matter where the output is most of the work: the 500 ms pulsar's
  128 bins at 1024 channels (4.16x to 3.05x: its 118 empty bins no longer zeroed each subint), and
  the naive port.  Elsewhere they change little: the atomic adds they need cost about what the
  smaller output saves.
- **The worst case, 128 bins of a 1.5 ms pulsar at 1024 channels (32 µs windows), is 14x**: a bin
  (12 µs) is narrower than a window, so a channel has ~6 windows a bin a subint and the XMAC runs
  65,000 one-slot-block segments, each a block's set-up and a full store.  A 102.4 ms subint: 11.5x
  (padding 156% to 28%).  Bins narrower than a window sample rather than resolve the pulse; a
  configuration to warn about rather than optimise.
- **32 stations** (1.5 ms, 128 channels unless said): gate 0.33x, 16 scrunched 1.22x, 16 separate
  2.45x (gather), 128 separate 6.7x (sorted); 500 ms, 128 bins 2.32x (gather).  The XMAC is 4.8 ms
  there, so the binning's fixed costs weigh less.
- In real time (10 stations): the binned XMAC at 0.02-0.3 ms per ms of data in every case but the
  worst (0.8).

## The A100 (OzSTAR gina8, job 21374617, 2026-11-22)

`run-bins.sh` at commit fafcd29 (`results-local/gina8-bins-20261122-142203`), every run passing.
ms per subint; best / XMAC as above (the XMAC's own time varies by ~9% between runs of the same
shape here, 0.321-0.352 ms):

| P | ch | bins | avg / acc | XMAC | bins kernel | naive | stage | passes | sorted | gather | best / XMAC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.5 ms | 128 | gate | 1 / no | 0.352 | 0.021 | 2.81 | 0.27 | | | | 0.77 |
| | | 16 scrunched | 1 / no | 0.352 | 0.031 | 2.81 | 0.63 | | | | 1.78 |
| | | 16 | 1 / no | 0.352 | 0.030 | 4.03 | | 3.83 | 1.45 | 1.14 | 3.23 (gather) |
| | | 128 | 1 / no | 0.321 | 0.037 | 16.8 | | 26.4 | 2.28 | 2.11 | 6.57 (gather) |
| | 1024 | gate | 1 / no | 0.375 | 0.018 | 1.86 | 0.36 | | | | 0.96 |
| | | 16 scrunched | 1 / no | 0.376 | 0.023 | 1.86 | 0.62 | | | | 1.65 |
| | | 16 | 1 / no | 0.403 | 0.026 | 3.91 | | 5.35 | 2.02 | 1.61 | 4.00 (gather) |
| | | 128 | 1 / no | 0.375 | 0.036 | 20.8 | | 39.4 | 7.55 | 6.58 | 17.5 (gather) |
| 500 ms | 128 | gate | 1 / no | 0.321 | 0.019 | 2.65 | 0.42 | | | | 1.31 |
| | | 16 scrunched | 1 / no | 0.321 | 0.029 | 2.66 | 0.42 | | | | 1.31 |
| | | 16 | 1 / no | 0.352 | 0.032 | 3.78 | | 2.91 | 1.48 | 1.14 | 3.23 (gather) |
| | | 128 | 1 / no | 0.321 | 0.036 | 3.57 | | 18.7 | 1.59 | 1.24 | 3.88 (gather) |
| | 1024 | gate | 1 / no | 0.376 | 0.019 | 1.86 | 0.47 | | | | 1.26 |
| | | 16 scrunched | 1 / no | 0.375 | 0.029 | 1.86 | 0.47 | | | | 1.26 |
| | | 16 | 1 / no | 0.404 | 0.032 | 2.85 | | 3.45 | 1.64 | 1.19 | 2.95 (gather) |
| | | 128 | 1 / no | 0.376 | 0.036 | 2.95 | | 21.9 | 2.09 | 1.78 | 4.75 (gather) |
| | | 128 | 8 / yes | 0.375 | 0.036 | 2.93 | | 22.2 | 1.73 | 1.42 | 3.79 (gather) |

(The other avg 8 rows: within a few % of the avg 1 ones.)

- **The unbinned XMAC is 3.4-3.9x faster than on the 2070**; the binned versions mostly keep their
  ratios to it, with three changes.
- **Gather wins every separate-bin case on the A100**, the short runs too (1.5 ms, 128 channels,
  128 bins: 6.6x against the permuted copy's 7.1x; on the 2070 12.1x against 6.5x).  Its 4-byte
  loads cost less there (40 MB of L2 against 4 MB).  So the permuted copy, and its buffer, could go.
- **Weighting in the stage costs relatively more**: the gate 0.77-0.97x at 1.5 ms (0.46-0.70x on
  the 2070), 1.26-1.33x with nothing to skip (1.10-1.13x), 16 scrunched 1.65-1.80x (1.24-1.37x).
  As with the phase-centre shift, the A100's tensor cores are 16x its FP32 rate (the 2070's about
  4x), so the copy's extra instructions (classification, scaling, signs) weigh more.
- **A pass per bin is relatively worse** (57-108x for 128 bins: many small launches and a
  classification each), and **the naive port relatively better** (23-55x for 128 bins at 1.5 ms,
  8-11x at 500 ms: the A100's L2 takes its scattered adds better).
- **The worst case** (1.5 ms, 1024 channels, 128 bins: bins narrower than a window): 17.5-18x.
- **In real time** every binned case but that one is at most 0.08 ms per ms of data (2.1 ms a
  25.6 ms subint); that one 0.27.
- **The bins kernel**: 0.018-0.037 ms, 5-12% of the XMAC.
