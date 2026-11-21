# phase-centres: backlog

## Next

- **Check mpifxcorr's implementation first**: the time step, how the shift is applied, and the
  per-centre cost there (see the README's "The problem").

## Approaches to compare (prototype and measure each)

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

## What to measure

- The cost against the number of centres (1, 10, 100, 1000), the channel count and the time step,
  for each approach, with the visibilities checked against a direct correlation at each centre
  (a correlation with that centre as the delay centre).
- Memory and DRAM traffic per centre.
