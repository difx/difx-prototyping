# two-stage: backlog

## Next

- **The decode overlap between stage-1 blocks.**  Blocks overlap by P x L1 − H samples, decoded
  twice; at 64 coarse channels (16 windows a block) up to 3x over.  A block sliding along the
  series, keeping its tail, would remove that.
- **The whole chain with the staged XMAC** (the stage-2 output is already in its layout).
- **The RTX PRO 6000** (the 2070 and the A100 are done).

## Later

- **Both stages in one block** for small N2 x Nc.
- **Zoom bands by skipping the coarse channels not needed**: stage 2 only where a zoom band is.
- **History across subints**: the PFB needs (P − 1) L1 samples before each subint, from the
  previous one (in mpifxcorr, from the datastream buffer).
- **Invalid data**: coarse samples whose input is invalid written as zeros, and a stage-2 window's
  data weight as the fraction of its coarse samples that were valid (the benchmark has all data
  valid; a per-(antenna, subint) flag for the delay model's validity).
- **Into DiFX**: the selection (one stage up to the cuFFTDx limit or the delay-rate limit, then
  two), and the startup report of the expected losses (../BACKLOG.md, "Carry to DiFX").
