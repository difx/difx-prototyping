# Backlog

One line per item; the reasoning is in each sub-project's `BACKLOG.md`, and what is done is in
its `FINDINGS.md`.

## Now

- [phase-centres](phase-centres/BACKLOG.md): a benchmark of the shift after the XMAC (direct, and as a matrix product).

## [phase-centres](phase-centres/BACKLOG.md)

- Done: mpifxcorr's implementation read (docs/mpifxcorr.md).
- Next: pieces from `maxNSBetweenUVShifts` (even split); delays per (antenna, centre); XMAC writing per piece.
- Next: the shift after the XMAC, direct and as a tensor-core matrix product; then in the XMAC's epilogue.
- Measure cost against centres (1-1000), windows per piece, averaging and stations, against a direct correlation.
- Later: a NUFFT shift for centres far more than the channels averaged.

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
