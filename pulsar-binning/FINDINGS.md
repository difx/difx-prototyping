# pulsar-binning: findings

## mpifxcorr's binning (2026-11-22)

- **Each (FFT window, channel) goes wholly into one bin**, by the polyco phase at the window's start
  plus the channel's dispersion offset (fixed per subint); the binconfig's bin ends are arbitrary
  phases.  Separate bins are written each with its weight (its share of the samples); scrunched
  bins are summed with their weights.  gpudifx does not bin.  ([docs/mpifxcorr.md](docs/mpifxcorr.md))
- **Arbitrary breakpoints, kept**: the bin of each (window, frequency, channel) is computed once per
  subint, in 32-bit integer turns, and is the same for every station and baseline (DiFX delays the
  data to the geocentre, where the polyco is evaluated); so its cost is tiny next to the XMAC's
  and does not depend on the breakpoints.  The precision is set by the window and the
  within-channel smearing, not by them.
- To investigate in mpifxcorr: the phase at the window's start (half a window's bias), the
  dispersion offsets of LSB bands, the normalisation of binned and gated output.  (BACKLOG.md)

## binbench (2026-11-22, RTX 2070 SUPER at 150 W / 1500 MHz)

The approaches side by side, with a diagram: [docs/approaches.md](docs/approaches.md).


- **The gate and scrunched output: one pass of the staged XMAC, weighted in its stage**
  (`xmacTensorStagedWeighted`): 0.32-0.64x the unbinned XMAC for the 1.5 ms gate (the blocks with no
  on-pulse window skipped), 0.99-1.16x with nothing to skip (16 bins on - off, negative weights
  included).  The copy is the plain one; each (window, channel)'s weight goes beside the stage in
  FP16 and multiplies the A operand in registers, sign included (the first version, scaling every
  value by sqrt|w| in the copy, was 1.04-1.37x).  ([docs/results.md](docs/results.md), steps 3, 6)
- **Separate bins: each channel's windows sorted by bin** (`xmacTensorStagedSorted`, a block per
  segment of a bin), from channel-major spectra ([station][band][channel][window]): 2.3-4.5x the
  unbinned XMAC for 16 bins and for the 500 ms pulsar (the XMAC gathering each slot's window), 6-7x
  for 128 bins of the 1.5 ms pulsar at 128 channels (a permuted copy), 14x at 1024 channels (bins
  narrower than a window).  ([docs/results.md](docs/results.md), steps 4, 6)
- **The data layout matters, but the station kernel need not change**: a permuted copy from the
  usual window-major spectra cost 17-28 ms against 3-6 ms from channel-major ones.  But gathering
  straight from window-major spectra in the XMAC (`gatherwm`) is as fast as from channel-major at
  1024 channels and low DM, and up to 2x slower at 128 channels or high DM; or the binning can keep
  its own transpose (about the XMAC's time).  On the A100 gatherwm beats the transpose in most
  cases (below), so window-major it is: the station kernel stays as it is.
- **One pass per bin** (one-hot weights): 3.3-70x; **the naive port** (a thread per output): 3.8-188x.
  Neither is the way for many bins.
- **The output** (nbins x the visibilities per subint) is a large part for many bins: averaging
  channels in the XMAC and accumulating over the integration (no zeroing each subint) help where it
  dominates (the 500 ms pulsar's empty bins: 4.2x to 3.1x).
- **The bins kernel**: 3-9% of the unbinned XMAC at 10 stations (a binary search over the ends).
- **The per-bin weights** (mpifxcorr's baseline weight per bin): a count of each bin's samples less a
  correction over the windows with a station weight below 1 (a matrix product): exact against
  mpifxcorr's sum.  The bins kernel counts as it goes (0.01-0.03 ms more) and one kernel finishes
  (0.004 ms with every weight 1): bins + weights 3-22% of the XMAC, from 12-28% with six kernels;
  with flagged windows 0.23-0.41 ms (from 0.42-1.05).
- **High DM** (300 at 1.5 ms: the bin changes at every channel at 1024 channels; 1000 at 500 ms): the
  sorted approaches are unchanged (each channel sorted on its own); the gate loses its skipping
  (1.17-1.31x instead of 0.46-0.70x); gathering from window-major spectra, a pass per bin and the
  naive port get worse.

## The A100 (2026-11-22, OzSTAR jobs 21374617 and 21375824)

- **The same picture, with gather winning every separate-bin case**: 3.0-4.8x the unbinned XMAC
  (6.6x for 128 bins of the 1.5 ms pulsar at 128 channels, where the permuted copy is 7.1x; on the
  2070 the copy won there).  One sorted approach, gathered, would do on both.
- **The weighted stage costs relatively more**: the gate 0.77-0.97x (1.5 ms), 1.26-1.33x with
  nothing to skip; scrunched 1.26-1.80x.  The A100 is issue-bound on the copy's extra instructions,
  as the phase-centre shift is.
- In real time every case but bins narrower than a window (0.27) is at most 0.08 ms per ms of data.
  ([docs/results.md](docs/results.md), "The A100")
- **Window-major spectra** (job 21375824): gathering from the station kernel's usual output beats a
  transpose plus the channel-major gather in 19 of 24 separate-bin runs, and at 1024 channels (DM 30)
  even the channel-major gather without its transpose.  It loses at 128 channels for the 1.5 ms
  pulsar's 128 bins (10.9x against 7.8x) and at high DM at 128 channels (up to 8.7x against 5.4x);
  worst case 0.14 ms per ms of data.  The permuted copy never wins on the A100.
- **High DM as on the 2070**: the sorted approaches unchanged; the 1.5 ms gate loses its skipping
  (0.77 to 1.70x at 128 channels).
- **The per-bin weights**: 0.06-0.12 ms a subint, 18-31% of the A100's XMAC (7-14% on the 2070): the
  count into the bins kernel matters more here.  ([docs/results.md](docs/results.md), "The A100 again")
- **The weights reworked** (job 21376562): the slimmed weighted copy 0.67-0.89x for the 1.5 ms gate
  (from 0.77-1.08x), 1.15-1.34x with nothing to skip (from 1.26-1.81x); bins + per-bin weights 8-22%
  of the XMAC, from 24-40%.  What is left in the weighted kernel (15-25%) is not the multiplies nor
  much the classification (2-3%; precomputed flags as their own kernel cost more than they save):
  Nsight Compute shows 28% more instructions at the same issue rate, from register pressure (the
  ldmatrix addresses re-derived from the thread index at the 128-register cap) and the list loop's
  bookkeeping.  ([docs/results.md](docs/results.md), "The weights" and "The A100: the classification ...")
- **Against the whole chain** (station + XMAC, job 21377096: the station stage is 3.2x the XMAC at 10
  stations on the A100): the gate and scrunched bins cost the chain 0.91-1.11x; 16 separate bins of a
  millisecond pulsar 1.7-1.8x (2.09x at 128 channels, DM 300); 128 separate bins 3.4-5.1x.
  ([docs/results.md](docs/results.md), "The binning against the whole chain")
- **32 separate bins** (gina17): the millisecond pulsar 2.00-2.07x the chain at DM 30, 2.95x at DM
  300 (window-major spectra: each channel its own slot list, a 4-byte load a slot); the 500 ms
  pulsar 1.41-1.68x (2.48x at 128 channels, DM 1000).  Channel-major spectra would give 1.73x at 128
  channels (before a transpose) but 2.38x at 1024.  ([docs/results.md](docs/results.md), "32 bins on the A100")
- **Kept out of the unweighted XMAC**: every binned mode is a template mode of the staged kernel;
  the plain kernel's registers and speed are unchanged (xmacbench, 4-64 stations).
