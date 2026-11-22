# pulsar-binning: findings

## mpifxcorr's binning (2026-11-22)

- **Each (FFT window, channel) goes wholly into one bin**, by the polyco phase at the window's start
  plus the channel's dispersion offset (fixed per subint); the binconfig's bin ends are arbitrary
  phases.  Separate bins are written each with its weight (its share of the samples); scrunched
  bins are summed with their weights.  gpudifx does not bin.  ([docs/mpifxcorr.md](docs/mpifxcorr.md))
- **Arbitrary breakpoints, kept**: the bin of each (window, channel) is computed once per subint for
  every baseline, in 32-bit integer turns, so its cost does not depend on the breakpoints; the
  precision is set by the window and the within-channel smearing, not by them.
- To investigate in mpifxcorr: the phase at the window's start (half a window's bias), the
  dispersion offsets of LSB bands, the normalisation of binned and gated output.  (BACKLOG.md)

## binbench (2026-11-22, RTX 2070 SUPER at 150 W / 1500 MHz)

- **The gate and scrunched output: one pass of the staged XMAC, weighted in its stage**
  (`xmacTensorStagedWeighted`): 0.33-0.70x the unbinned XMAC for the 1.5 ms gate (the blocks with no
  on-pulse window skipped), 1.04-1.37x with nothing to skip (16 bins on - off, negative weights
  included: the sign goes into the A operand).  ([docs/results.md](docs/results.md), steps 3, 6)
- **Separate bins: each channel's windows sorted by bin** (`xmacTensorStagedSorted`, a block per
  segment of a bin), from channel-major spectra ([station][band][channel][window]): 2.3-4.5x the
  unbinned XMAC for 16 bins and for the 500 ms pulsar (the XMAC gathering each slot's window), 6-7x
  for 128 bins of the 1.5 ms pulsar at 128 channels (a permuted copy), 14x at 1024 channels (bins
  narrower than a window).  ([docs/results.md](docs/results.md), steps 4, 6)
- **The data layout is the crux**: from the usual window-major spectra, sorting a channel's windows
  is a 4-byte gather each (dispersion keeps neighbouring channels apart): the permuted copy cost
  17-28 ms against 3-6 ms from channel-major.  So in pulsar mode the station kernel should write
  channel-major spectra.
- **One pass per bin** (one-hot weights): 3.3-70x; **the naive port** (a thread per output): 3.8-188x.
  Neither is the way for many bins.
- **The output** (nbins x the visibilities per subint) is a large part for many bins: averaging
  channels in the XMAC and accumulating over the integration (no zeroing each subint) help where it
  dominates (the 500 ms pulsar's empty bins: 4.2x to 3.1x).
- **The bins kernel**: 3-9% of the unbinned XMAC at 10 stations (a binary search over the ends).
- **Kept out of the unweighted XMAC**: every binned mode is a template mode of the staged kernel;
  the plain kernel's registers and speed are unchanged (xmacbench, 4-64 stations).
