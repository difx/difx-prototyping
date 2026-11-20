# pulsar-binning

GPU pulsar gating and binning for gpudifx.  **Backlog: nothing built yet.**

## The problem

mpifxcorr can accumulate the visibilities into pulse-phase bins: each FFT window's data goes to the
bin of the pulse phase at its time, from the pulsar's ephemeris (polycos), per channel, because
dispersion delays each channel differently.  Gating is the one-bin case (on-pulse kept, the rest
dropped).  gpudifx does not have it yet.

## What makes it hard on the GPU

The tensor-core XMAC in `../fft-fusion` (the staged one, the path kept) sums 16 windows at a time
in registers, for blocks of channels.  With binning, the windows of one 16-window block can fall
in different bins, and which bin depends on the channel.  For a millisecond pulsar with 32 bins
and 8 µs windows, a bin lasts only a few windows, so bins change inside a block.

## Approaches to compare

- **A pass per bin**: the XMAC run once per bin, with the windows not in the bin set to zero (or
  skipped by the copy into the stage).  Simple, and it costs nbins x the XMAC.  Gating is then one
  pass.
- **Weights in the stage**: the copy into shared memory multiplies each (window, channel) value
  by its bin's on/off weight, so the tensor cores never see the other bins.  This is still a pass
  per bin, but the passes share the loads if one block holds several bins' accumulators.
- **Bin-major accumulators**: one accumulator set per bin, held in registers or shared memory, with
  each 16-window block split at the bin boundaries.  Tensor-core efficiency falls as the bins get
  shorter.
- **Bin boundaries at the station stage**: the station kernel tags or reorders windows by bin
  (per channel: hard).

## What to measure

- The cost against nbins (1 = gating, 8, 32, 128), pulse period and DM, with the visibilities
  checked against gpudifx's (or mpifxcorr's) binned output.
- Memory: nbins x the visibilities.
- Whether incoherent dedispersion (the per-channel bin offset) changes the answer.

Depends on `../fft-fusion`'s staged XMAC.  The input/output maps there ("DiFX's flexibility in the
tensor XMAC") are the natural place for the per-window, per-channel bin weights.
