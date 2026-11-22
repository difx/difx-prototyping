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
