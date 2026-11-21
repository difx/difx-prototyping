# Two-stage channelisation: implementation and results

_Split from `docs/two-stage.md` in the 2026-11-21 reorganisation; the analysis is in [analysis.md](analysis.md)._

## Implementation and first results (RTX 2070 SUPER, 150 W + 1500 MHz, 2026-11-21)

PFB first stage only (the WOLA option dropped).  `twostagebench` (`twostage.cuh`, `twostage1.cu`,
`twostage2.cu`; `run-twostage.sh`):
- **Stage 1** (cuFFTDx, L1 = 32, 64, 128 points; 1, 2, 4 bits).
  - **Decode once.**  A block of 256 threads takes FPB consecutive coarse windows of one
    (antenna, band).  It decodes the block's span of samples once into shared memory: the level,
    the alternating sign by absolute parity, and the fringe rotation as a function of the absolute
    sample.
  - **Fold.**  Each thread folds one element position of 8 windows, so each tap is loaded once for
    8 windows.  It writes padded FP16 rows.
  - **FFT and corrections.**  The L1-point FFT at 8 elements a thread reads those rows.  The
    fractional-sample phase at the coarse centre and the oversampling derotation are added in fixed
    point, one sincos per output.  The coarse series is stored FP16, time-contiguous per coarse
    channel.
- **Stage 2** (cuFFTDx, N2 = 128-8192): FFT, keep K bins (the in-band half for the DC and Nyquist
  coarse channels), FP16 in the staged XMAC's layout.
- **The delay model, in the kernel:** an integer shift per coarse window, taken at the centre of its
  P x L1 window, and a continuous fringe phase.  Nothing is uploaded or read per coarse window: the
  kernel takes a delay polynomial per (antenna, subint) and a fringe polynomial per (antenna, band)
  (`TwoStageModel`), plus the clock terms per (antenna, band).
  - **Per block (thread 0, FP64):** the delay at window n0's centre, split into an integer and a
    fraction, and the local rate; the fringe phase and rate at the block's first sample; the clock
    terms of the fractional-sample phase.
  - **Per window (FP32):** window n0 + k's delay is the block's fraction + rate x kH (+ d2 (kH)^2).
    A block spans at most ~3500 samples, so this is at most 0.035 samples at 10 us/s and FP32
    holds it to ~1e-9 samples.  Each thread that needs a window's shift or fraction (the fold, the
    correction) recomputes it, with explicit round-to-nearest FMAs so every thread gets the same
    bits.  Per-window arrays in shared memory instead (640 bytes a block) were slower where the
    kernel sits at Turing's 64 KB of shared memory an SM: 32 coarse channels, 4/3, 4 taps,
    0.427 ms/ms against 0.385 (by the arithmetic, 16.3 KB a block fits 4 blocks an SM and 17.0 KB
    only 3; not profiled).
  - **The limit:** a block's span has 16 samples of slack for shifts, so |rate| up to ~0.01 (100x
    a spacecraft's); beyond that the kernel refuses the block and the check fails.
  - **Before** (the "setup kernel" column): the host made an integer sample index, a fractional
    delay and fringe terms per coarse window, and a separate FP64 kernel turned them into a
    16-byte phase per (window, band) for stage 1 to read back: more bytes than the 2-bit data at
    16-96 samples a window.
- **The check:** a double-precision host reference of the whole chain (decode from the packed bytes,
  fold, DFT, corrections, stage-2 DFT) on 4 series per run, coarse channels 0 and Nc among them.
  The bench nulls every per-window array stage 1 used to read.  Also checked at rates of
  ±1e-4 and accelerations of ±0.5 s/s^2 (unphysical, to exercise the in-block quadratic term).
  All 18 configurations pass at about 4e-4 rms (FP16 rounding of the span, the folded rows, the
  coarse series and the spectra).
- **What got it there.**  The first version decoded each sample once per window it is in (P x O
  times) and was 6x slower.  Then: decoding once per block; moving the fold off the FFT's register
  layout (its half-warps spanned windows whose reads fell in the same banks); FP16 working buffers
  (FP32 allowed one 256-thread block an SM, 25% occupancy); and loading each tap once for 8 windows.
  4.5x in all.

6 antennas, 8 bands of 64 MHz, 2 bits, 16 stage-2 windows; ms per ms of data
(`results-local/ar313-adeller-twostage-20261121-120407`; with the setup kernel,
`results-local/ar313-adeller-twostage-20261121-002831`).
- **Slowdown** is the total over the single-stage end-to-end kernel at its largest size, 8192
  channels (FP16 spectra, the fastest variant, 0.282 ms/ms).
- **Against gpudifx today** is at the same channel count: 1.536 ms/ms at 49152 channels, 1.376 at
  65536.
- **Loss** is the zero-delay continuum loss: the exact model's a + b / K at this K, plus the
  average fractional-delay residual (0.013, 0.003, 0.001% at Nc = 16, 32, 64).
- **Alias** is the worst image and **ripple** the passband ripple (both for line work).

| Nc | O | taps | channels | stage 1 | stage 2 | total | total, setup kernel | slowdown | against gpudifx | loss | alias | ripple |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 4/3 | 4 | 49152 | 0.462 | 0.095 | 0.551 | 0.618 | 1.95 | 2.78x | 0.22% | −25 dB | 0.76 dB |
| 16 | 4/3 | 8 | 49152 | 0.506 | 0.094 | 0.591 | 0.662 | 2.09 | 2.60x | < 0.1% | −39 dB | 0.14 dB |
| 16 | 4/3 | 16 | 49152 | 0.597 | 0.093 | 0.685 | 0.753 | 2.43 | 2.24x | < 0.1% | −65 dB | 0.004 dB |
| 32 | 4/3 | 4 | 49152 | 0.385 | 0.091 | 0.468 | 0.500 | 1.66 | 3.28x | 0.22% | −25 dB | 0.76 dB |
| **32** | **4/3** | **8** | **49152** | **0.490** | **0.091** | **0.572** | **0.601** | **2.03** | **2.69x** | **< 0.1%** | **−39 dB** | **0.14 dB** |
| **32** | **4/3** | **16** | **49152** | **0.603** | **0.090** | **0.682** | **0.712** | **2.42** | **2.25x** | **< 0.1%** | **−65 dB** | **0.004 dB** |
| 64 | 4/3 | 4 | 49152 | 0.497 | 0.089 | 0.578 | 0.590 | 2.05 | 2.66x | 0.22% | −25 dB | 0.76 dB |
| 64 | 4/3 | 8 | 49152 | 0.570 | 0.089 | 0.649 | 0.662 | 2.30 | 2.37x | < 0.1% | −39 dB | 0.14 dB |
| 64 | 4/3 | 16 | 49152 | 0.863 | 0.089 | 0.937 | 0.954 | 3.32 | 1.64x | < 0.1% | −65 dB | 0.004 dB |
| 16 | 2 | 4 | 65536 | 0.630 | 0.178 | 0.803 | 0.903 | 2.85 | 1.71x | < 0.05% | −64 dB | 0.011 dB |
| 16 | 2 | 8 | 65536 | 0.692 | 0.178 | 0.870 | 0.971 | 3.08 | 1.58x | < 0.05% | −115 dB | 0 |
| 16 | 2 | 16 | 65536 | 0.828 | 0.178 | 1.005 | 1.103 | 3.56 | 1.37x | < 0.1% | −207 dB | 0 |
| **32** | **2** | **4** | **65536** | **0.523** | **0.133** | **0.651** | **0.699** | **2.31** | **2.11x** | **< 0.05%** | **−64 dB** | **0.011 dB** |
| 32 | 2 | 8 | 65536 | 0.603 | 0.133 | 0.732 | 0.777 | 2.60 | 1.88x | < 0.05% | −115 dB | 0 |
| 32 | 2 | 16 | 65536 | 0.834 | 0.133 | 0.966 | 1.004 | 3.43 | 1.42x | < 0.1% | −207 dB | 0 |
| 64 | 2 | 4 | 65536 | 0.674 | 0.131 | 0.804 | 0.819 | 2.85 | 1.71x | < 0.05% | −64 dB | 0.011 dB |
| 64 | 2 | 8 | 65536 | 0.782 | 0.131 | 0.912 | 0.929 | 3.23 | 1.51x | < 0.05% | −115 dB | 0 |
| 64 | 2 | 16 | 65536 | 0.992 | 0.130 | 1.122 | 1.143 | 3.98 | 1.23x | < 0.1% | −207 dB | 0 |

- **Cost.** Two stages cost 1.7-4x the single-stage kernel per ms of data, and are 1.2-3.3x faster
  than gpudifx today at the same channel count.
- **The in-kernel delay model** saved the whole setup (0.015-0.09 ms/ms, 2-11% of the total, most at
  16 coarse channels) and left stage 1's own time unchanged (within 1-2%).
- **The coarse channel count.** 32 coarse channels is the sweet spot.
  - At 16, stage 1 is up to 20% slower than at 32 (4 taps; the same at 16 taps) and the stage-2
    FFT (8192 points at O = 2) costs more.  With the setup gone, 16 is within 5% of 32 at 8 and 16
    taps at O = 4/3.
  - At 64, stage 1's blocks hold only 16 windows (L1 = 128), so the overlap between blocks is
    decoded up to 3x over, and 16 taps of 128 points make the fold heavy.
- **The oversampling** costs about 1.3-1.5x (O = 2 against 4/3: 1.5x the coarse windows and twice
  the coarse series), close to the window ratio.
- **Taps** cost 10-20% from 4 to 8, more to 16 (the fold is P x O multiply-adds per sample).
- **The sensitivity loss is negligible everywhere but one case.** 4 taps at O = 4/3 lose 0.22% and
  have −25 dB alias images and 0.76 dB of ripple: its 27 dB Kaiser stopband is too little for a 4/3
  transition.  Everything else is within the exact model's ±0.07% of zero; the edge term 0.13 / K
  is 0.002-0.017% at these K.
- **The choices** (bold):
  - 32 coarse channels, O = 4/3, 8 taps: 2.0x the single stage, < 0.1% loss, −39 dB images.
  - 16 taps if lines matter: 2.4x, −65 dB images.
  - O = 2, 4 taps for powers of two: 2.3x, −64 dB images, 65536 channels.
- **Not yet optimised:**
  - Stage 1's blocks overlap by P x L1 − H samples, decoded twice.  A block sliding along the
    series, keeping its tail, would remove that.
  - The A100 and RTX PRO 6000 are still to run.

## At about 8k channels: the like-for-like comparison

The single-stage kernel's largest size is 8192 channels, so the slowdown above compares 49152-65536
channels with 8192.  The same sweep at about 8192 channels (`NCHAN=8192 NWIN=128 ./run-twostage.sh`,
the same data per run; `results-local/ar313-adeller-twostage-8192-20261121-120452`; with the setup
kernel, `results-local/ar313-adeller-twostage-8192-20261121-104326`):
- **The channel counts.** O = 2 gives 8192 exactly at every Nc (N2 = 1024, 512, 256 at Nc = 16, 32,
  64).  O = 4/3 cannot: with power-of-two L1 and N2 the nearest is 6144 (N2 = 512, 256, 128),
  or 12288.  16 coarse channels with a 512-point stage 2 is 4096 channels at O = 2, 6144 at 4/3.
- **Loss** is the floor a (±0.07% model uncertainty; 0.22% for 4 taps at 4/3) plus the edge term
  b / K, b = 0.12-0.14 at O = 4/3 and about 0.09 at O = 2, plus the fractional-delay residual.  At
  these K the edge term is no longer negligible: it is larger than the fractional-delay residual,
  so 16 coarse channels lose less in all than 32.
- **gpudifx today** is 1.540 ms/ms at 6144 channels (a 12288-point cuFFT) and 1.088 at 8192.

| Nc | O | taps | N2 | channels | stage 1 | stage 2 | total | total, setup kernel | slowdown | against gpudifx | edge b/K | frac. delay |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 4/3 | 4 | 512 | 6144 | 0.462 | 0.094 | 0.551 | 0.627 | 1.95 | 2.79x | 0.034% | 0.013% |
| 16 | 4/3 | 8 | 512 | 6144 | 0.506 | 0.093 | 0.590 | 0.661 | 2.09 | 2.61x | 0.032% | 0.013% |
| 16 | 4/3 | 16 | 512 | 6144 | 0.602 | 0.093 | 0.684 | 0.752 | 2.43 | 2.25x | 0.036% | 0.013% |
| 32 | 4/3 | 4 | 256 | 6144 | 0.385 | 0.079 | 0.456 | 0.488 | 1.62 | 3.37x | 0.068% | 0.003% |
| **32** | **4/3** | **8** | **256** | **6144** | **0.490** | **0.079** | **0.560** | **0.590** | **1.98** | **2.75x** | **0.064%** | **0.003%** |
| 32 | 4/3 | 16 | 256 | 6144 | 0.603 | 0.079 | 0.671 | 0.700 | 2.38 | 2.29x | 0.071% | 0.003% |
| 64 | 4/3 | 4 | 128 | 6144 | 0.497 | 0.105 | 0.594 | 0.606 | 2.10 | 2.59x | 0.136% | 0.001% |
| 64 | 4/3 | 8 | 128 | 6144 | 0.570 | 0.105 | 0.665 | 0.678 | 2.36 | 2.31x | 0.128% | 0.001% |
| 64 | 4/3 | 16 | 128 | 6144 | 0.855 | 0.106 | 0.954 | 0.971 | 3.38 | 1.61x | 0.142% | 0.001% |
| 16 | 2 | 4 | 1024 | 8192 | 0.630 | 0.136 | 0.761 | 0.860 | 2.70 | 1.43x | 0.018% | 0.013% |
| 16 | 2 | 8 | 1024 | 8192 | 0.692 | 0.136 | 0.831 | 0.929 | 2.95 | 1.31x | 0.017% | 0.013% |
| 16 | 2 | 16 | 1024 | 8192 | 0.828 | 0.136 | 0.964 | 1.061 | 3.42 | 1.13x | 0.018% | 0.013% |
| **32** | **2** | **4** | **512** | **8192** | **0.528** | **0.133** | **0.651** | **0.699** | **2.31** | **1.67x** | **0.036%** | **0.003%** |
| 32 | 2 | 8 | 512 | 8192 | 0.600 | 0.133 | 0.733 | 0.777 | 2.60 | 1.49x | 0.034% | 0.003% |
| 32 | 2 | 16 | 512 | 8192 | 0.834 | 0.133 | 0.967 | 1.005 | 3.43 | 1.13x | 0.036% | 0.003% |
| 64 | 2 | 4 | 256 | 8192 | 0.675 | 0.111 | 0.785 | 0.799 | 2.78 | 1.39x | 0.073% | 0.001% |
| 64 | 2 | 8 | 256 | 8192 | 0.782 | 0.111 | 0.892 | 0.909 | 3.16 | 1.22x | 0.068% | 0.001% |
| 64 | 2 | 16 | 256 | 8192 | 0.992 | 0.111 | 1.103 | 1.124 | 3.91 | 0.99x | 0.073% | 0.001% |

- **The slowdown hardly changes with the channel count** (1.98 here against 2.03 at 49152 for
  Nc = 32, O = 4/3, 8 taps).  Stage 1 is 80-90% of the total, and its cost per sample does not
  depend on the final channel count; stage 2 is 0.08-0.14 ms/ms at either size.  The ~2x is the
  cost of the PFB itself, against a single stage that does one pass over the data.
- **16 coarse channels is not the fastest, but close.** With the setup in the kernel it is 5% behind
  32 at 8 taps and 2% at 16 (O = 4/3); at 4 taps its stage 1 is still 20% slower.  It also loses
  less (the edge term above), so at about 8k channels 16 x 8 taps at 4/3 (2.09x) is as good a
  choice as 32 x 8 (1.98x).
- **Choices at about 8k:** 32 (or 16) coarse channels, O = 4/3, 8 taps (6144 channels, 2.0x,
  < 0.1% loss); O = 2, 4 taps for 8192 exactly (2.3x).

## A100-SXM4-80GB (OzSTAR, gina301), about 64k channels

`fusebench-ozstar.sbatch` with `SCRIPTS=run-twostage.sh`, SLURM job 21315678, commit 39e0886, all 18
passing (`results-local/gina301-twostage-20261121-123215`).  The single-stage kernel at 8192
channels: 0.0951 ms/ms; gpudifx today 0.326 ms/ms at 49152 channels, 0.249 at 65536.

| Nc | O | taps | channels | stage 1 | stage 2 | total | slowdown | against gpudifx | 2070 slowdown |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 4/3 | 4 | 49152 | 0.1237 | 0.0284 | 0.1520 | 1.60 | 2.15x | 1.95 |
| 16 | 4/3 | 8 | 49152 | 0.1243 | 0.0258 | 0.1500 | 1.58 | 2.18x | 2.09 |
| 16 | 4/3 | 16 | 49152 | 0.1507 | 0.0258 | 0.1765 | 1.86 | 1.85x | 2.43 |
| 32 | 4/3 | 4 | 49152 | 0.1082 | 0.0254 | 0.1335 | 1.40 | 2.45x | 1.66 |
| 32 | 4/3 | 8 | 49152 | 0.1239 | 0.0254 | 0.1492 | 1.57 | 2.19x | 2.03 |
| 32 | 4/3 | 16 | 49152 | 0.1546 | 0.0254 | 0.1798 | 1.89 | 1.82x | 2.42 |
| 64 | 4/3 | 4 | 49152 | 0.1211 | 0.0312 | 0.1522 | 1.60 | 2.15x | 2.05 |
| 64 | 4/3 | 8 | 49152 | 0.1413 | 0.0312 | 0.1725 | 1.81 | 1.89x | 2.30 |
| 64 | 4/3 | 16 | 49152 | 0.1820 | 0.0312 | 0.2131 | 2.24 | 1.53x | 3.32 |
| 16 | 2 | 4 | 65536 | 0.1692 | 0.0610 | 0.2079 | 2.19 | 1.20x | 2.85 |
| 16 | 2 | 8 | 65536 | 0.1901 | 0.0609 | 0.2506 | 2.63 | 0.99x | 3.08 |
| 16 | 2 | 16 | 65536 | 0.2333 | 0.0608 | 0.2941 | 3.09 | 0.85x | 3.56 |
| 32 | 2 | 4 | 65536 | 0.1477 | 0.0362 | 0.1837 | 1.93 | 1.35x | 2.31 |
| 32 | 2 | 8 | 65536 | 0.1712 | 0.0362 | 0.2073 | 2.18 | 1.20x | 2.60 |
| 32 | 2 | 16 | 65536 | 0.2170 | 0.0362 | 0.2531 | 2.66 | 0.98x | 3.43 |
| 64 | 2 | 4 | 65536 | 0.1677 | 0.0364 | 0.2039 | 2.14 | 1.22x | 2.85 |
| 64 | 2 | 8 | 65536 | 0.1975 | 0.0364 | 0.2338 | 2.46 | 1.06x | 3.23 |
| 64 | 2 | 16 | 65536 | 0.2584 | 0.0364 | 0.2946 | 3.10 | 0.84x | 3.98 |

- **The slowdown is smaller on the A100:** 1.4-3.1x against 1.7-4x on the 2070.  32 coarse
  channels, O = 4/3: 1.40x at 4 taps, 1.57x at 8.
- **Against gpudifx today** the margin is smaller (0.84-2.45x): gpudifx's FP64 decode and cuFFT
  run well on the A100.  At O = 2 with 16 taps two stages are slower than gpudifx today; at
  O = 4/3, 8 taps, 2.2x faster.
- **16 and 32 coarse channels tie at 8 taps, O = 4/3** (0.150 and 0.149 ms/ms).  16's stage 1 costs
  the same at 4 and 8 taps (0.124), so the fold is not what limits it there (not profiled).  At O = 2,
  16's 8192-point stage 2 (0.061 against 0.036 ms/ms) puts it behind.
- **The choices hold:** 32 coarse channels, O = 4/3, 8 taps (1.6x the single stage); 16 taps for
  lines (1.9x); O = 2, 4 taps for powers of two (1.9x).

## A100-SXM4-80GB (OzSTAR, gina17), about 8k channels

SLURM job 21316333, commit 4ae0c67, all 18 passing
(`results-local/gina17-twostage-8192-20261121-124621`).  The single-stage kernel at 8192 channels:
0.0952 ms/ms; gpudifx today 0.221 ms/ms at 6144 channels, 0.1995 at 8192.

| Nc | O | taps | channels | stage 1 | stage 2 | total | slowdown | against gpudifx | 2070 slowdown |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 4/3 | 4 | 6144 | 0.1236 | 0.0274 | 0.1508 | 1.58 | 1.47x | 1.95 |
| 16 | 4/3 | 8 | 6144 | 0.1327 | 0.0265 | 0.1591 | 1.67 | 1.39x | 2.09 |
| 16 | 4/3 | 16 | 6144 | 0.1609 | 0.0265 | 0.1872 | 1.97 | 1.18x | 2.43 |
| 32 | 4/3 | 4 | 6144 | 0.1083 | 0.0214 | 0.1295 | 1.36 | 1.71x | 1.62 |
| 32 | 4/3 | 8 | 6144 | 0.1240 | 0.0214 | 0.1452 | 1.52 | 1.52x | 1.98 |
| 32 | 4/3 | 16 | 6144 | 0.1546 | 0.0214 | 0.1757 | 1.85 | 1.26x | 2.38 |
| 64 | 4/3 | 4 | 6144 | 0.1211 | 0.0242 | 0.1452 | 1.52 | 1.52x | 2.10 |
| 64 | 4/3 | 8 | 6144 | 0.1413 | 0.0243 | 0.1655 | 1.74 | 1.34x | 2.36 |
| 64 | 4/3 | 16 | 6144 | 0.1820 | 0.0242 | 0.2061 | 2.16 | 1.07x | 3.38 |
| 16 | 2 | 4 | 8192 | 0.1529 | 0.0467 | 0.1994 | 2.09 | 1.00x | 2.70 |
| 16 | 2 | 8 | 8192 | 0.1719 | 0.0467 | 0.2184 | 2.29 | 0.91x | 2.95 |
| 16 | 2 | 16 | 8192 | 0.2112 | 0.0467 | 0.2578 | 2.71 | 0.77x | 3.42 |
| 32 | 2 | 4 | 8192 | 0.1477 | 0.0349 | 0.1826 | 1.92 | 1.09x | 2.31 |
| 32 | 2 | 8 | 8192 | 0.1713 | 0.0349 | 0.2061 | 2.16 | 0.97x | 2.60 |
| 32 | 2 | 16 | 8192 | 0.2171 | 0.0349 | 0.2519 | 2.65 | 0.79x | 3.43 |
| 64 | 2 | 4 | 8192 | 0.1678 | 0.0301 | 0.1977 | 2.08 | 1.01x | 2.78 |
| 64 | 2 | 8 | 8192 | 0.1976 | 0.0301 | 0.2276 | 2.39 | 0.88x | 3.16 |
| 64 | 2 | 16 | 8192 | 0.2585 | 0.0301 | 0.2885 | 3.03 | 0.69x | 3.91 |

- **The slowdown at 8k is what it is at 64k** (32 coarse channels, O = 4/3, 8 taps: 1.52x against
  1.57x), as on the 2070.  Best: 1.36x (32, 4/3, 4 taps).
- **Against gpudifx today** two stages are 1.1-1.7x faster at 6144 channels (O = 4/3) and 0.7-1.1x at
  8192 (O = 2).  But 8192 channels is the single stage's range, where it is 2.1x faster than
  gpudifx today: two stages are only for what one cannot do.
- **16 coarse channels** is 10% behind 32 at 8 taps here (0.159 against 0.145 ms/ms; tied at 64k):
  its stage 1 is 7% slower (0.133 against 0.124) and its 512-point stage 2 costs more than 32's
  256-point one (0.027 against 0.021).
