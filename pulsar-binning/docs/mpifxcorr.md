# How mpifxcorr bins, and what it means for the GPU

Read from `difx` (branch `gpudifx`, 4ea9ef2d8), `mpifxcorr/src/polyco.cpp` (`Polyco`) and
`core.cpp` (`Core::processdata`, `Core::uvshiftAndAverage`), `configuration.cpp` (the binconfig).
gpudifx does not bin: `GPUCore` sets up the polyco for each subint, as `Core` does, but nothing on
the GPU uses it.

## The bins

- The binconfig: `NUM PULSAR BINS`, `SCRUNCH OUTPUT`, then per bin `BIN PHASE END` and
  `BIN WEIGHT`.  Bin k ends at phase end[k] and starts at end[k-1]; bin 0 wraps, from end[N-1]
  to end[0] through phase 0 (`getBins`: phase < end[0] or >= end[N-1] is bin 0).  The ends must
  ascend (`setFrequencyValues` checks the minimum width).  Any doubles in [0, 1].
- Width of bin k: end[k] - end[k-1] (mod 1).  Weight: any double, negative allowed (to subtract
  an off-pulse estimate when scrunching).

## The phase of each (window, channel)

- Per subint (`setTime`): the polyco's frequency f at the subint's middle (`calculateDMPhaseOffsets(
  calclengthmins / 2)`), and per channel the dispersion phase offset f × d_c, with
  d_c = -DM K / ν_c² + DM K / ν_ref², K = 1/0.000241 s MHz², ν_c = band edge + j bw / nchan
  (channel j's centre in DiFX's convention, the first channel centred on the band edge), ν_ref the
  polyco's reference frequency.
  - For a lower-sideband band `Configuration` passes the edge minus (nchan - 1) bw / nchan, so
    channel 0 gets the **lowest** frequency and the offsets ascend with j.  If the band's data
    have channel 0 at the band edge (the highest frequency), the offsets are reversed across the
    band: a possible bug, to investigate (BACKLOG.md).  Fixed for the subint; it warns
  if f changes enough over the subint to move a phase by more than 1% of the narrowest bin.
- Per FFT window i (`getBins`, every window of every batch): the polyco phase at the window's
  **start** (offset i x the window length; the polynomial in double), plus each channel's offset;
  bin = the bin holding frac(phase).  So every (window, channel) goes wholly into one bin: there is
  no splitting of a window across a boundary.
- Bins depend on (window, channel, frequency) only: the same for every station and baseline
  (the data are already aligned to the geocentre).

## Accumulation and output

- **Separate bins** (`SCRUNCH OUTPUT FALSE`): each (window, channel) product goes to its bin's
  fine visibilities, `threadcrosscorrs[baseline][bin][product][channel]`, and its weight
  w1 w2 / nchan to `baselineweight[freq][bin][baseline][product]`: so each bin's weight is its
  share of the (window, channel) samples.  Each bin is then shifted (phase centres), averaged in
  frequency and written as its own output.
- **Scrunched** (`SCRUNCH OUTPUT TRUE`): per-bin accumulators as above (`pulsaraccumspace`), then
  at the shift each bin is scaled by its weight and they are summed into one output.  Its weight
  is Σ (w1 w2 / nchan) x weight[bin] over the samples, **positive weights only** (negative ones
  are left out of the weight, by design).
- Phase centres and binning combine: the shift runs per bin.
- **Normalisation**: the weights count samples, so a gated or binned visibility's amplitude, and
  a scrunched one's noise, are not normalised so that the noise comes out right (a weighted sum
  Σ w_b V_b has noise ∝ √(Σ w_b² n_b), its weight Σ_{w_b > 0} w_b n_b).  To fix (BACKLOG.md).

## Arbitrary breakpoints or 1/2^N?

**Keep arbitrary breakpoints (mpifxcorr's binconfig, unchanged).**  The reasons:

1. **The cost does not depend on it.**  The bin index is computed once per (window, channel,
   frequency) per subint and shared by every baseline and product: nwin x nchan x nfreq values,
   against the XMAC's nwin x nchan x nfreq x nbaseline x 4 multiply-accumulates.  Every GPU
   approach below takes a bin index (or a weight) per (window, channel), however it was computed.
   - 1/2^N would simplify the phase-to-bin step (the top N bits of the phase).  Integer phase gets
     most of that anyway: the phase as a 32-bit unsigned fraction of a turn (2^-32 turns; the wrap
     modulo 1 is the integer overflow), per (window, channel) the subint's phase + the window's
     step + the channel's dispersion offset, in integer adds.  The breakpoints converted to the
     same units; the bin from a 1024-cell table on the top 10 bits, with one comparison in the cells
     that hold a breakpoint (or a 7-step binary search over 128).  For 1/2^N bins the table is the
     answer.
2. **The precision is set elsewhere.**  A (window, channel) is one sample in time, so a phase is
   known to the window length T_w = nchan / bw at best, and within a channel dispersion smears by
   8.3 µs x DM x Δν(MHz) / ν(GHz)³.  At 1.4 GHz, DM 30, a 1.5 ms pulsar: 0.1 MHz channels (the
   best balance) give ~10 µs of each, about 1/80 of the period; 1/1024 (1.5 µs) is out of reach
   for incoherent dedispersion.  For a 500 ms pulsar 1/1024 (490 µs) is easy, and arbitrary
   breakpoints lose nothing there either.
3. **Compatibility**: existing binconfig files, and the same output as mpifxcorr's for checking.

What 1/2^N would buy is the "accumulate 2^N fine bins, combine afterwards" design other
correlators use, which costs 2^N x the visibility memory; with the bins known before the job,
there is no need for it.

To decide for the GPU (not a question of the breakpoints): mpifxcorr takes the phase at the
window's **start**, a bias of T_w / 2 (2 µs for 4 µs windows, 1/750 of a 1.5 ms period; 16 µs at
1024 channels, 1/94): a possible bug, to investigate.  The prototype matches mpifxcorr by default,
with an option for the window's middle.

## Why the GPU XMAC finds it hard

The staged tensor XMAC (`common/xmac.cu`) sums **16 consecutive windows of a channel** in one MMA
(K = windows), for blocks of 8-16 channels, from spectra stored [station][window][band][channel].
With binning, the 16 windows must all be in the same bin, for that channel.

- **Bin runs are short for millisecond pulsars.**  Windows per bin (32 MHz bands):

  | | 128 channels (4 µs windows) | 1024 channels (32 µs) |
  |---|---|---|
  | P = 1.5 ms: a period | 375 | 47 |
  | a 1/128 bin | 2.9 | 0.37 |
  | a 1/16 bin | 23 | 2.9 |
  | the 10% gate | 37 | 4.7 |
  | P = 500 ms: a 1/128 bin | 977 | 122 |

  A 25.6 ms subint is 17 periods of the 1.5 ms pulsar, but 5% of the 500 ms one's (~7 of 128
  bins).  A bin narrower than a window gets the windows whose start falls in it.
- **Dispersion makes the bins differ by channel.**  Across a 32 MHz band at 1.4 GHz, DM 30, the
  delay changes by 2.9 ms (two periods of a 1.5 ms pulsar).  Adjacent channels differ by 23 µs
  (5.7 windows) at 128 channels, 2.9 µs (0.09 windows) at 1024.  So a block's channels see
  different bins in the same window, and a window order sorted by bin is different for every
  channel (at 1024 channels, only near the boundaries).
- **Output volume**: separate bins multiply the output by nbins.  At 1024 channels, 10 stations,
  4 dual-pol bands, 128 bins: 755 MB per subint before averaging (8 GB at 32 stations); 94 MB
  at 128 channels.  The
  slow pulsar touches ~7 bins a subint, so the output is sparse per subint but full over an
  integration.
