# pulsar-binning: backlog

## Next

- **Check mpifxcorr's and gpudifx's binning first**: how the bins and their per-channel offsets
  (dispersion) are computed and applied, and the output layout.

## Approaches to compare (prototype and measure each)

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
