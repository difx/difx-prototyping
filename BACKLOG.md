# Backlog

One line per item; the reasoning is in each sub-project's `BACKLOG.md`, and what is done is in
its `FINDINGS.md`.

## Now

- [phase-centres](phase-centres/BACKLOG.md): phasecentrebench optimised (shift 2-3x at 100+ centres); next the A100 again.

## [phase-centres](phase-centres/BACKLOG.md)

- Done: mpifxcorr's implementation read (docs/mpifxcorr.md).
- Done: phasecentrebench (pieces, delays per antenna, the shift naive / SIMT / tensor), swept on the 2070.
- Done: the A100 sweep (shift / XMAC ≈ 0.5-0.65 x centres / windows per piece).
- Done: the optimisation round on the 2070 (shift 2-3x at 100+ centres, XMAC zeroing skipped); lower sideband; lag-window term checked.
- Dropped: the XMAC's pieces in one launch (0-1%); the shift in the XMAC's epilogue (atomics 2-260x the shift).
- Done: the A100 after the optimisation round (shift 1.1-2.1x; +45% over the XMAC at 100 centres).
- Next (optional): the A100's occupancy (3 blocks an SM by registers: 24 of 64 warps).
- Later: pieces in multiples of 16 windows (only for very short pieces); a NUFFT shift for centres ≫ channels averaged.

## [pulsar-binning](pulsar-binning/BACKLOG.md)

- Check mpifxcorr's and gpudifx's binning (bins, per-channel dispersion offsets, output layout).
- Prototype and compare: a pass per bin, weights in the stage, bin-major accumulators, bins at the station stage.
- Measure cost against bins (1-128), pulse period and DM, against gpudifx's or mpifxcorr's binned output.

## [two-stage](two-stage/BACKLOG.md)

- Next: stage 1's blocks decode their overlap twice: a block sliding along the series.
- Next: the whole chain with the staged XMAC.
- Next: the RTX PRO 6000.
- Later: both stages in one block; zoom bands by skipping coarse channels; history across subints; invalid data.

## [xmac](xmac/BACKLOG.md)

- Next: DiFX's flexibility in the map: bands straddling a frequency edge, baselines not correlated, channel averaging, FP16's range.
- Later: large N (100+ stations): bigger super-blocks, more output per stage load.
- Later: overlapping the station stage and the XMAC in chunks; accumulators on chip across chunks.

## [fft-fusion](fft-fusion/BACKLOG.md)

- Next: the lean decode's 32-frame window (2- and 4-bit, 128 bands, long windows).
- Next: the end-to-end kernel is arithmetic-bound on the A100: fewer instructions per sample.
- Later: the wider test matrix's open corners (channel counts not powers of two, single polarisation, 8 bits).
- Later: FFT lengths that are not powers of two (EHT: N x 125 points).
- Later: mixed-bandwidth correlation on the station side (per-station FFT lengths).
- Later: WOLA through a series buffer; phases by recurrence; complex data; phase cal; CODIF.
- Later: the effect inside mpifxcorr (overlap with transfers and the XMAC).

## Carry to DiFX

Things the prototypes point to that belong in difx itself, not here:
- **Report the channelisation's response at startup.** mpifxcorr (or gpudifx) should compute and
  log, for the configuration it is about to run, the expected continuum sensitivity loss, the
  passband ripple, the worst alias/ghost level, and the fractional-sample and delay-rate losses
  (worst channel and average, at the job's largest delay rate).  These follow from the window
  (WOLA, PFB taps, oversampling), the FFT sizes and the delay model, as `two-stage/two_stage.py`
  computes them; see `two-stage/docs/analysis.md`.
- **Select one or two stages** by the channel count per subband and the delay rate
  ([two-stage/FINDINGS.md](two-stage/FINDINGS.md)).
- **Decide how the tunable parameters are set** (to decide before the mpifxcorr implementation,
  not now).  Several kernel choices are compile-time today and their best values depend on the
  GPU and the job: fft-fusion's fused kernel is compiled per FFT length (cuFFTDx needs the size at
  compile time, and the build is slow); the XMAC's tile and super-block sizes; the phase-centre
  shift's centres per block and blocks per SM (2070 and A100 may differ); one stage or two.  The
  options:
  - *At run time*: compile the kernels when the job starts (NVRTC, with cuFFTDx's headers),
    cached per GPU and shape.  Fully flexible; a JIT toolchain at every site, and a first-run delay.
  - *At install time*: the user (or a benchmark suite run at install-difx time) picks the values
    for the site's GPU, written to a tuning file.  Simple at run time; re-tune when the GPU changes.
  - *Several builds, chosen per job*: compile a set of variants (FFT lengths, tile sizes, per GPU
    architecture) and select one from the job's inputs (channels, stations, centres, averaging) by
    heuristics.  No JIT; bigger binaries and builds, and the heuristics need measurements behind
    them (as the sweeps here provide).
  Mixing them is possible (e.g. variants built, with an install-time benchmark tuning the
  heuristics' thresholds).
