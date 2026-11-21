# pulsar-binning

GPU pulsar gating and binning for gpudifx.  **Not started**: the approaches are in [BACKLOG.md](BACKLOG.md).

## The problem

mpifxcorr can accumulate the visibilities into pulse-phase bins: each FFT window's data goes to the
bin of the pulse phase at its time, from the pulsar's ephemeris (polycos), per channel, because
dispersion delays each channel differently.  Gating is the one-bin case (on-pulse kept, the rest
dropped).  gpudifx does not have it yet.

## What makes it hard on the GPU

The tensor-core XMAC in [`../xmac`](../xmac/README.md) (the staged one, the path kept) sums 16 windows at a time
in registers, for blocks of channels.  With binning, the windows of one 16-window block can fall
in different bins, and which bin depends on the channel.  For a millisecond pulsar with 32 bins
and 8 µs windows, a bin lasts only a few windows, so bins change inside a block.

Depends on [`../xmac`](../xmac/README.md)'s staged XMAC.  The input/output maps there ("DiFX's flexibility in the
tensor XMAC" in [its backlog](../xmac/BACKLOG.md)) are the natural place for the per-window, per-channel bin weights.
