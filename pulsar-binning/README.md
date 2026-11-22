# pulsar-binning

GPU pulsar gating and binning for gpudifx.  mpifxcorr's implementation is read
([docs/mpifxcorr.md](docs/mpifxcorr.md): arbitrary breakpoints kept); the plan and test cases are
in [BACKLOG.md](BACKLOG.md).

## binbench

```bash
make binbench            # in the repository root
pulsar-binning/binbench --period-ms 1.5 --nchan 128 --bins 128
pulsar-binning/binbench --period-ms 500 --nchan 1024 --bins gate
```

Simulates a dispersed pulsar in the station spectra, computes each (window, frequency, channel)'s
bin as mpifxcorr does (integer phase, arbitrary ends), and bins the XMAC: the naive port (a thread
per output), the staged XMAC weighted in its stage (`stage`: one pass, the gate and scrunched
output; `passes`: one a bin), bins sorted per channel (`sorted`: a permuted copy of channel-major
spectra; `gather`: the XMAC gathering them), timed against the unbinned staged XMAC and checked
against the host in double.  Bins: `gate` (2, scrunched with weights 0 / 1), `16` (unequal, the narrowest 1/1024; on - off
when `--scrunch`ed), `128`, `even:N`.  `--avg A` sums A channels into each output channel,
`--accumulate` adds into the output instead of zeroing it each subint.  `--help` lists the rest
(stations, channels, subint, period, DM, the phase convention).  `run-bins.sh` runs the whole test
matrix (results-local/<host>-bins-<stamp>/).  Results: [docs/results.md](docs/results.md); in short,
[FINDINGS.md](FINDINGS.md); the approaches side by side, with a diagram: [docs/approaches.md](docs/approaches.md).

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
