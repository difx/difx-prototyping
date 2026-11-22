# pulsar-binning: backlog

mpifxcorr's implementation is read: [docs/mpifxcorr.md](docs/mpifxcorr.md), with the decision
to keep arbitrary breakpoints.

## Done

- binbench steps 1-2 (below): the bins (integer phase, arbitrary ends: exact against mpifxcorr's
  rule; 3-9% of the unbinned XMAC at 10 stations) and the naive port (3.8-6.2x the XMAC with one
  sum, 96-188x for 128 separate bins at 1.5 ms), checked on the 2070 ([docs/results.md](docs/results.md)).
- Step 3: weights in the staged XMAC's stage (`xmacTensorStagedWeighted`, the unweighted kernel
  unchanged): the gate and scrunched output in one pass at 0.46-1.35x the unbinned XMAC; separate
  bins by a pass each 3.3-70x (not the way for many bins).
- Step 4: bins sorted per channel (`xmacTensorStagedSorted`, a block per segment of a bin), from
  channel-major spectra: with a permuted copy 4.6-13x, gathering in the XMAC 2.5-18x (best for long
  runs; the copy for short).  The output (nbins x the visibilities, zeroed and written each subint)
  is a large part.  Step 5 (splitting 16-window blocks at bin changes) not built: its in-loop store
  spilled, and gather covers long runs.
- The A100 (job 21374617): the same picture; gather wins every separate-bin case; the stage costs
  relatively more (docs/results.md, "The A100").
- The per-bin weights (count + a correction over flagged windows, exact against mpifxcorr's sum);
  high DM in the sweep; gathering from window-major spectra (`gatherwm`): the station kernel need
  not change (docs/results.md, "Per-bin weights, high DM ...").
- The weights (docs/results.md, "The weights"): the per-bin weights in two kernels (the count in
  the bins kernel; scrunched, a matrix-vector product): bins + weights 3-22% of the XMAC on the
  2070, from 12-28%.  The weighted copy slimmed (the weight on the A operand in registers): 0.99-1.16x
  the XMAC with nothing to skip, from 1.04-1.37x; the gate 0.40x (from 0.47x).  The A100 (job
  21376562): 1.15-1.34x with nothing to skip (from 1.26-1.81x), the gate 0.67x (0.77x); bins +
  weights 8-22% of the XMAC (24-40%).
- The A100 again (job 21375824): gathering from window-major spectra beats a transpose to
  channel-major in 19 of 24 separate-bin runs, so window-major is the layout; high DM as on the
  2070; the per-bin weights 18-31% of the A100's XMAC (docs/results.md, "The A100 again").
- Step 6 (part): channel averaging in the binned XMAC (a shared-memory sum over a block's channels)
  and accumulating over an integration; `run-bins.sh`, the whole matrix on the 2070: the gate
  0.45-1.13x, scrunched 1.04-1.37x, 16 separate 2.6-4.5x, 128 separate 2.7-6.5x (14x for bins
  narrower than a window).  [FINDINGS.md](FINDINGS.md).

## Next

- **One sorted approach: gather from window-major spectra** (gatherwm): on the A100 the fastest,
  counting the transpose the others need, in 19 of 24 separate-bin runs, and at most 1.6x the
  fastest in the rest (128 channels, short runs or high DM); the permuted copy never wins there.  Kept (2026-11-22): drop the
  permuted copy and the channel-major path in DiFX; the 128-channel millisecond pulsar below.
- **The weighted kernel's rest** (the A100, job 21376562: 1.15-1.25x the XMAC for an all-ones gate,
  no multiplies; 2-11% on the 2070): the classification is now made with the bins (`block_flags`,
  flag bytes per frequency, 16-window block and 4-channel quad; on the 2070 net -0.4% to +1.6%,
  docs/results.md, "The classification with the bins"): the A100 to measure it, and Nsight Compute
  there (the pb-* cases of ncu-ozstar.sbatch) to split what is left.  If the flags stay, fold
  `block_flags` into another kernel (0.014 ms alone on the 2070, mostly its launch and 16 dependent
  loads).
- **The per-bin weights' correction** on the tensor cores, if heavily flagged data matter (0.26 ms at
  1161 flagged windows, 128 bins, 10 stations; it grows with flagged windows x bins x baselines).
- **The millisecond pulsar at fewer channels, from window-major spectra** (kept: window-major; 128
  channels, 1.5 ms: 10.9x the XMAC for 128 bins, 5.7x for 16 at DM 300, against 7.8x and 4.4x with a
  transpose to channel-major).  To try: 16-byte loads of 4 channels where a quad's channels share a
  slot's window (most slots at low DM), 4-byte gathers only where they differ; cp.async for the
  gathers; ordering the blocks so those reading the same windows run together (L2); slots shared
  across a wider channel group; a transpose of just the windows a subint's short-run bins touch;
  or the transpose kept as a fallback chosen per job (fewer than ~256 channels with short runs or
  high DM).
- **The bins kernel**: a lookup table on the phase's top bits instead of the binary search.
- **Sorting over several subints** (or a longer XMAC interval in pulsar mode) where bins get few
  windows a subint (1024 channels, 128 bins at 1.5 ms: 11.5x at 102.4 ms against 14x).
- **The weighted kernel's spills** (KS 2: 40-320 bytes; the gate and scrunched output's 4-37% over
  the XMAC when nothing is skipped).
- **In DiFX**: the sorted XMAC gathering from the station kernel's usual window-major output (no
  change there); the bins per subint without the device-to-host sync the prototype's permute uses.

## The test cases

Every parameter variable; these are the defaults and the sweep.

| case | bins | output |
|---|---|---|
| gate | 2 (on-pulse 10%) | scrunched, weights 1 (on), 0 (off) |
| 16 | 16 of unequal widths, the narrowest 1/1024, finest round the pulse | separate; and scrunched (weighted) |
| 128 | 128 even | separate |

- **Pulsars**: P = 1.5 ms and P = 500 ms, DM 30 (variable), a polyco-like phase model (phase,
  frequency, its derivative per subint).
- **Channelisation**: 128 and 1024 channels a band for each pulsar (4 µs and 32 µs windows;
  6400 and 800 a 25.6 ms subint), 4 dual-pol 32 MHz bands, 10 stations (also 32).  The frequency
  resolution is the user's choice; the smearing it implies is reported, not changed.
- **Checks**: a simulated pulsar (a pulsed signal common to the stations, with its profile and
  dispersion, plus noise), every version against a double-precision host reference on sampled
  outputs, and the bins' signal against the profile; the per-bin weights against mpifxcorr's
  rule.

## The plan: `binbench` (pulsar-binning/binbench.cu)

1. **The bins**: a kernel computing the bin index per (window, channel, frequency) per subint
   (uint8 / uint16), in integer phase (a 32-bit fraction of a turn: the subint's phase, the
   window's step and the channel's dispersion offset added as integers, the wrap free), arbitrary
   breakpoints by a 1024-cell table on the top bits plus one comparison.  mpifxcorr's convention
   by default (the window's start; its channel frequencies), an option for the window's middle.
   Its cost against the XMAC's: 3-9% at 10 stations (the binary search); a lookup table to try.
2. **The reference**: a naive port of mpifxcorr (a thread per (baseline, product, channel),
   looping over windows into its bins), checked against the host in double.
3. **Weights in the stage** (the staged XMAC, `common/xmac.cu`, behind a template flag so the
   default path is unchanged): the copy into shared memory multiplies each (window, channel) by
   a weight from its bin.  The tile serves both sides of the product (X1 X2^*), so a weight s
   there enters as s²: the stage carries √|w|, and a negative weight's sign goes into the A
   operand's registers (an XOR of the sign bits per window, next to the sign flip the kernel
   already makes there for the imaginary part).  So one pass for any weights.
   - Scrunched output (and the gate) in one pass, at the XMAC's cost (+ the weights' loads).
   - Separate bins as one 0/1 pass per bin, skipping the 16-window chunks with none of the bin
     in the block's channels.  Expected fine for few bins, poor for 128 at 1.5 ms: a bin is in
     ~30% of an 8-channel block's chunks at 128 channels, nearly all at 1024.
4. **Bin-sorted windows** (separate bins, many bins): per channel, the windows sorted by bin
   (a counting sort; offsets per (channel, bin), padded to 16).  The XMAC gathers each channel's
   16 windows of one bin and writes its accumulators to that bin at the bin's end.  One pass;
   the cost is the gather (4-byte loads per channel instead of rows of channels) and the padding
   (1.5 ms, 128 bins: ~50 windows per (channel, bin) a subint at 128 channels, +15-28%; ~6 at 1024,
   +150%: there a longer subint, or bins sorted over several subints, would help).  Measure; if
   the gather is too slow, a separate pass writing the spectra bin-sorted per channel (a
   permutation of the station spectra), and later the station kernel writing that order directly.
5. **Splitting the 16-window chunks at bin boundaries** (the alternative to 4 when runs are long:
   the 500 ms pulsar): masked MMAs per bin present in a chunk, accumulators written at each bin
   change.  ~1x for long runs, ~nbins-per-chunk x for short.
6. **The output side**: per-bin weights; channel averaging per bin; for the 500 ms pulsar the
   sparse bins (only those touched in a subint written or copied; the device-to-host copy of
   all 128 bins x 1024 channels would be ~30 GB/s).  (Phase centres with binning: not needed on the
   GPU, one or the other.)
7. **Sweep** on the 2070, then the A100 (OzSTAR), each version against the unbinned XMAC:
   cases x periods x stations, and DM 0 (no channel dependence) against DM 30.

## What to decide from it

- Which approach per regime (few bins / many; short runs / long), and whether one covers all.
- Whether the station kernel should write a bin-friendly order in pulsar mode.
- Window start or middle (mpifxcorr compatibility against accuracy).
- From the 2070 (above): the stage for the gate and scrunched output; sorting for separate bins,
  gathering for long runs (16 bins, slow pulsars), a permuted copy for short ones (many bins of a
  millisecond pulsar); channel-major spectra from the station kernel in pulsar mode.

## To investigate in mpifxcorr

- **The phase at the window's start** (`getBins`: offset i x the window length), not its middle:
  a bias of half a window (1/750 of a 1.5 ms period at 4 µs windows, 1/94 at 32 µs).  A bug?
- **Lower sideband**: `Configuration` gives the polyco the band edge minus (nchan - 1) channels,
  so the dispersion offsets ascend with the channel index; if an LSB band's channel 0 is its
  edge (highest frequency), they are reversed across the band.
- **Normalisation of binned and gated output**: the weights count samples, so neither a bin's or
  gate's amplitude nor a weighted scrunch's noise comes out right (a weighted sum Σ w_b V_b has
  noise ∝ √(Σ w_b² n_b); its weight is Σ over w_b > 0 of w_b n_b).  Work out the normalisation
  that makes the noise correct, for mpifxcorr and the GPU alike.

## Later

- **Coherent dedispersion** before the channelisation (per station, in the station kernel or
  before it): removes the within-channel smearing that limits the bins' precision for
  millisecond pulsars (~1/80 of a 1.5 ms period at DM 30, 1.4 GHz, at best, incoherently).
