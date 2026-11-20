# phase-centres

GPU correlation to many phase centres at once (wide-field / multi-field VLBI) for gpudifx.
**Backlog: nothing built yet.**

## The problem

DiFX can correlate one pointing to many phase centres in a single pass.  The visibilities are
formed at high time and frequency resolution for the pointing centre.  Then, for each phase
centre, they are shifted (the differential delay and rate to that centre, a phase per baseline,
channel and short time step) and averaged down to the output resolution.  The averaging can
then be coarse without smearing the target sources, so each centre's output is small.  (To check
against mpifxcorr's implementation before designing: the time step, how the shift is applied, and
the per-centre cost there.)

## Why it follows the two-stage FFT

The pointing-centre correlation must keep fine channels (bandwidth smearing across the field) and
short time steps (time smearing).  Fine channels mean long FFTs, which is where the two-stage
channelisation in `../fft-fusion` (its open items) should help: a coarse stage, then short FFTs.

## Approaches to compare

- **Shift after the XMAC (as DiFX does)**: the XMAC writes high-resolution visibilities for each
  short time step, and a second kernel rotates and averages them once per centre.  This is
  memory-bound on reading the high-resolution visibilities once per centre (or once for all
  centres, if one kernel does several centres per read).
- **Shift in the XMAC's epilogue**: the staged XMAC holds a time step's sums in registers.  For
  each centre it could apply the phase and add into that centre's low-resolution accumulator
  directly, so the high-resolution visibilities never reach DRAM.  Its cost is centres x baselines
  x channels x time steps of complex multiply-adds, plus atomics or a reduction.  The question is
  whether that beats the DRAM traffic it saves.
- **Shift per station before the XMAC**: a fringe rotation per centre at the station stage
  multiplies the XMAC by the number of centres.  Only for a few centres.

## The XMAC's output cost: not a concern here

fft-fusion's big matrix found that with very few windows a subint (20-23, at 64-256 stations) the
XMAC's per-subint output (zeroing, atomics, stores) outweighs its products ("The big matrix" in
`../fft-fusion/README.md`).  Multiple phase centres do not get there: at L band, 8k channels in a
32 MHz subband (4 kHz) with a 40 ms subint (time smearing comparable to, still a bit below, the
frequency smearing) is about 160 FFT windows a subint before the shift to each centre.

## What to measure

- The cost against the number of centres (1, 10, 100, 1000), the channel count and the time step,
  for each approach, with the visibilities checked against a direct correlation at each centre
  (a correlation with that centre as the delay centre).
- Memory and DRAM traffic per centre.

Depends on `../fft-fusion`'s staged XMAC and, for fine channels, its two-stage channelisation.
