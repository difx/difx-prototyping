# phase-centres

GPU correlation to many phase centres at once (wide-field / multi-field VLBI) for gpudifx.
**Not started**: mpifxcorr's implementation is in [docs/mpifxcorr.md](docs/mpifxcorr.md), the GPU
design and plan in [BACKLOG.md](BACKLOG.md).

## The problem

DiFX can correlate one pointing to many phase centres in a single pass.  The visibilities are
formed at high time and frequency resolution for the pointing centre.  Then, for each phase
centre, they are shifted (the differential delay and rate to that centre, a phase per baseline,
channel and short time step) and averaged down to the output resolution.  The averaging can
then be coarse without smearing the target sources, so each centre's output is small.  mpifxcorr
shifts every `maxNSBetweenUVShifts` (each thread over its share of the subint), applies the
delay only (no rate within a piece), and averages in frequency
([docs/mpifxcorr.md](docs/mpifxcorr.md)).

## Why it follows the two-stage FFT

The pointing-centre correlation must keep fine channels (bandwidth smearing across the field) and
short time steps (time smearing).  Fine channels mean long FFTs, which is where the two-stage
channelisation in [`../two-stage`](../two-stage/README.md) should help: a coarse stage, then short FFTs.

## The XMAC's output cost: a concern if the pieces are short

fft-fusion's big matrix found that with very few windows a subint (20-23, at 64-256 stations) the
XMAC's per-subint output (zeroing, atomics, stores) outweighs its products ("The big matrix" in
[`../fft-fusion/docs/shapes-and-matrix.md`](../fft-fusion/docs/shapes-and-matrix.md)).  Counted per subint, multiple phase centres do not get there: at L band, 8k channels in a
32 MHz subband (4 kHz) with a 40 ms subint (time smearing comparable to, still a bit below, the
frequency smearing) is about 160 FFT windows a subint.  But the XMAC flushes once per shift piece, not once per subint,
so it is the windows per piece that count: with `maxNSBetweenUVShifts` at a few ms the pieces can
drop below 20 windows (BACKLOG, "Watch").

Depends on [`../xmac`](../xmac/README.md)'s staged XMAC and, for fine channels, [`../two-stage`](../two-stage/README.md).
