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
- Done: Nsight Compute on the A100: issue-bound (tensor pipe 9-10%); 4 blocks an SM 6-8% faster there; index fixes 2.7-7% (2070).
- Done: blocks an SM chosen for the GPU (4 where 4 fit, else 3; `--blocks` overrides).
- Done: the index fixes on the A100: 6-9%; with 4 blocks the shift 1.16-1.27x faster there (+40% over the XMAC at 100 centres).
- Later: pieces in multiples of 16 windows (only for very short pieces); a NUFFT shift for centres ≫ channels averaged.

## [pulsar-binning](pulsar-binning/BACKLOG.md)

- Done: mpifxcorr's binning read (docs/mpifxcorr.md); arbitrary breakpoints kept, phase in integer math.
- Done: binbench's bins (exact against mpifxcorr's rule) and naive port (3.8-188x the unbinned XMAC), on the 2070.
- Done: weights in the XMAC's stage: the gate and scrunched output in one pass, 0.46-1.35x the unbinned XMAC; a pass per bin 3-70x.
- Done: bins sorted per channel (channel-major spectra): 2.5-6.5x the unbinned XMAC (18x at 1024 channels x 128 bins, the output).
- Done: channel averaging and accumulation in the binned XMAC; run-bins.sh, the whole matrix on the 2070 (FINDINGS.md).
- Done: the A100 (gather wins every separate-bin case; the weighted stage relatively dearer); docs/approaches.md, the approaches with a diagram.
- Done: per-bin weights (exact); high DM in the sweep; gathering from window-major spectra (the station kernel need not change).
- Done: the A100 again: window-major spectra (gatherwm) beat a transpose in 19 of 24 runs, so the station kernel stays as it is.
- Done: the weights: the per-bin weights in two kernels (3-22% of the XMAC, from 12-28%); the weighted copy slimmed (0.99-1.16x with nothing to skip, from 1.04-1.37x; 2070).
- Done: the A100 with the new weights: 1.15-1.34x with nothing to skip (from 1.26-1.81x); bins + weights 8-22% of the XMAC (24-40%).
- Done: the classification with the bins (net 0.3-1.9% worse on the A100 as its own kernel); Nsight Compute: the weighted kernel's 15-25% is 28% more instructions, register pressure (ldmatrix addresses re-derived at 128 registers) and the list loop.
- Done: the binning against the whole chain (A100 station stage 3.2x the XMAC): gate and scrunched 0.91-1.11x the chain, 16 separate MSP bins 1.7-2.1x.
- Done: 32 separate bins on the A100: 2.0-2.07x the chain at DM 30, 2.95x at DM 300 (millisecond pulsar).
- Next: where 32 bins' time goes at high DM (Nsight Compute); the weighted kernel left as it is (within the 2x target); one sorted approach (gather from window-major).
- Investigate in mpifxcorr: the phase at the window's start; LSB dispersion offsets; the normalisation of binned and gated output.
- Later: coherent dedispersion.

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
- **Report the pulsar binning's smearing at startup**: the window length and the within-channel
  dispersion smearing against the narrowest bin (pulsar-binning/docs/mpifxcorr.md).  The user
  sets the resolution; DiFX says what it costs.
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
