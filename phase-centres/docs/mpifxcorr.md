# How mpifxcorr does multiple phase centres

Read from `difx` (branch `gpudifx`, 4ea9ef2d8), `mpifxcorr/src/core.cpp`: `Core::processdata`
(the shift cadence), `Core::uvshiftAndAverage` and `Core::uvshiftAndAverageBaselineFreq` (the
shift).  gpudifx does not support it: `GPUCore::completegpudata` says so, and the fused XMAC writes
the averaged visibilities straight to the results.

## The cadence: `maxNSBetweenUVShifts`

- v2d `maxNSBetweenUVShifts` (default 2e9 ns, i.e. off), per scan in the `.calc`/`.im`, read by
  `Model` as `maxnsbetweenxcavg` (`getMaxNSBetweenXCAvg`).  The autocorrelations have their own,
  `maxNSBetweenACAvg`.
- Each core thread gets a contiguous `numblocks` of the subint's FFT windows (1/Nthreads of it).
  It accumulates its fine visibilities (`threadcrosscorrs`: every baseline, band, pol product and
  channel, FP32) and every `maxxcblocks` windows calls `uvshiftAndAverage`, then zeroes them.
  - `maxxcblocks = floor(maxNS / window time)`, rounded **down** to a multiple of
    `numBufferedFFTs` (at least one batch).  Whatever is left at the end of the thread's share is
    shifted as a shorter last piece.  So the pieces are uneven, and the shift time is never below
    one buffered batch.
  - The shift is evaluated at the piece's midpoint (`nsoffset`); its length (`nswidth`) feeds the
    decorrelation estimate.
- So on the CPU the shift interval is min(maxNS, the thread's share of the subint), as you say.

## The shift (per piece, per baseline, band)

1. **Delays**, on the host, per baseline (so each antenna's are recomputed once per baseline it is
   in): `Model::calculateDelayInterpolator` at the piece's midpoint, a linear fit over 1 µs, for the
   pointing centre and each phase centre s, for both antennas.
   - Differential delay: Δτ_s = (τ2,s − τ2,0)(1 + τ̇2,0) − (τ1,s − τ1,0)(1 + τ̇1,0).  The (1 + rate)
     factor is the pointing centre's geometric rate.
   - Differential rate: Δτ̇_s = (τ̇2,s − τ̇2,0) − (τ̇1,s − τ̇1,0).  It is used only for the decorrelation
     estimate: **no rate is applied within a piece**.
2. **Rotator**: the phase of channel c is 2π Δτ_s (f_LO + c δf), the sky frequency (the stations
   were fringe-rotated to the pointing centre).  It is built in two levels to save sin/cos:
   - a fine rotator of `rotatestridelen` channels, including the band-edge term Δτ f_LO;
   - a coarse step rotator, one value per stride of `rotatestridelen` channels;
   - rotatestridelen + nchan/rotatestridelen sin/cos per (baseline, band, centre).
3. **Rotate**: per pol product (and pulsar bin), two complex multiplies per channel
   (`vectorMul_cf32` by the fine rotator, `vectorMulC_cf32_I` by the stride's coarse value), into
   a scratch buffer.
4. **Average and accumulate**: sum each group of `channelinc` channels (`vectorSum_cf32`, which
   handles output bands nested in wider ones) and add it into the slot's results, at that centre's
   offset.  It runs under a mutex per (band, baseline), because every thread adds into the same
   results.
   - Each centre's output is a full copy of the averaged spectra:
     centres x baselines x bands x pols x nchan/channelinc.
5. **Decorrelation estimate**, accumulated per (band, baseline, centre), and written to the output
   (`baselineshiftdecorr`) as Σ nswidth · d.  d is the product of:
   - time smearing: sinc(π Δτ̇_s · nswidth · f_LO) (the band's edge frequency, not its top);
   - delay: 1 − |Δτ_s| / (nchan/bw) (the lag window).
   - It logs an error when either goes negative.

## Cost per piece

Per (baseline, band, pol, centre): 2 complex multiplies and 1 add per fine channel.  The sin/cos
is per (baseline, band, centre), in the two-level rotator above.  The fine visibilities are
read once per centre (from cache: they are the thread's own).  The XMAC over the same piece does
one complex multiply-add per (baseline, band, pol, channel, window), so

> **shift / XMAC ≈ 2 × centres / windows per piece.**

With a few centres and long pieces the shift is free.  With 100s of centres it dominates whatever
the platform.  On the CPU, each thread's share sets the windows per piece (subint / Nthreads at
most).  On a GPU nothing forces the pieces to be short, only `maxNSBetweenUVShifts`.
