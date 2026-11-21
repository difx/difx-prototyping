# two-stage: findings

Detail: [docs/analysis.md](docs/analysis.md) (the design and its losses), [docs/results.md](docs/results.md)
(the implementation and its timings), [docs/plan.md](docs/plan.md) (the plan as written).

## When two stages (analysis, 2026-11-20)

- **One stage whenever possible.**  Two only where one cannot: more channels per subband than one
  cuFFTDx FFT gives (8192 for real data; a limit per subband, not on the channel width), or a delay
  rate at which one long window loses (0.5% in the top channel at about 18k channels per subband at
  3 µs/s, 1.8k at 30 µs/s, 550 at 100 µs/s).  ([analysis.md](docs/analysis.md), "In short",
  "Delay accuracy")
- **A WOLA first stage leaves ghosts and loses 3-5% of continuum sensitivity** (a line lands as a
  sharp ghost about O coarse channels away; the loss does not depend on Nc).  An oversampled PFB
  removes both.  ([analysis.md](docs/analysis.md), "Is WOLA the right first stage?", "Leakage
  against frequency offset")
- **The PFB keeps the fractional-sample correction exact to well under 0.2%**: one integer shift per
  coarse window, the fraction corrected at each coarse centre, leaving a residual slope worth
  1 − sinc(1/(2 L1)) at worst.  ([analysis.md](docs/analysis.md), "Delay accuracy")

## The implementation (2026-11-21)

- **Built and checked:** a PFB of 16/32/64 coarse channels (O = 4/3 or 2, 4-16 taps) and a fine FFT
  of 128-8192 points, against a double-precision reference of the whole chain.  Stage 1 decodes
  each sample once a block (4.5x over the first version), folds 8 windows a thread, works in FP16.
  ([results.md](docs/results.md), "Implementation and first results")
- **The delay model in the kernel**: a polynomial per antenna and subint, evaluated in FP64 once per
  block, each window's shift and fraction in FP32; nothing per coarse window, 2-11% off the total.
  ([results.md](docs/results.md))
- **Cost against one stage, per ms of data: 1.4-3.1x on the A100, 1.6-4x on the 2070**, and the
  same at about 8k channels as at 64k: stage 1 (the PFB) is 80-90% of it, whatever the final
  channel count.  ([results.md](docs/results.md), all four tables)
- **The choices:** 32 coarse channels, O = 4/3, 8 taps (A100 1.5-1.6x one stage, 2070 2.0x; < 0.1%
  loss, −39 dB alias images); 16 taps where lines matter (−65 dB); O = 2, 4 taps for powers of two
  (1.9x, 2.3x).  4 taps at O = 4/3 lose 0.22% with −25 dB images.  ([results.md](docs/results.md))
- **Against gpudifx today at the same channel count: 0.7-2.5x faster on the A100, 1.0-3.4x on the
  2070** (on the A100 at O = 2 with 8-16 taps two stages are slower than gpudifx today; at 8192
  channels, though, one stage is the one to use).
