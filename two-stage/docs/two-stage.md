# Two-stage channelisation: analysis and implementation plan

**Status: plan, nothing implemented yet.**  The numbers come from `two_stage.py` and the plots
from `plot_two_stage.py` (`python3 two_stage.py`, `python3 plot_two_stage.py`, from `fft-fusion/`).

## In short

- **When (the direction, 2026-11-20).**  One stage whenever possible.  Two stages, with a moderate
  number of PFB coarse channels, only where one stage cannot do it:
  - **The resolution needs more channels per subband than one cuFFTDx FFT gives.**  The fused
    kernel is built up to 16384 points, so 8192 channels per real-sampled subband.  This is a
    limit on channels per subband, not on the channel width: recording narrower subbands reaches
    finer channels within it.
  - **The delay rate makes the fractional-sample correction lose.**  One stage loses 0.5% in its
    top channel at about 18k channels per subband at Earth-rotation rates (3 µs/s), so it is fine
    up to the cuFFTDx limit.  But it loses that at 1.8k channels per subband at 30 µs/s and 550 at
    100 µs/s (spacecraft VLBI).
  - **The size of the PFB:** 16 points (8 coarse channels of real data) already keeps the
    fractional-sample loss under 0.2% in the worst fine channel (Delay accuracy).  In practice the
    stage-2 FFT sets the number: Nc of at least O × (channels per subband) / 16384.

- **Why.** The fractional-sample correction uses one delay per FFT window (its middle), in
  fft-fusion and in mpifxcorr.
  Across a window channel j's phase then drifts by 2π j × (delay rate), whatever the bandwidth.
  At an Earth-rotation delay rate (3 µs/s) that costs 6% of the top channel's coherence at 65536
  channels per subband, and nearly all of it at 2^18 and beyond; at spacecraft rates, from a few
  thousand channels per subband.  A coarse first stage with an integer shift per coarse window
  leaves only the fractional part within a coarse channel, under 0.2% loss with a 16-point PFB,
  whatever the rate.  It also replaces one long FFT (the fused kernel stops at 16384 points) with
  short ones.
- **Should the first stage be WOLA and the second not?**  The second stage needs no overlap.  It
  runs on each coarse channel's continuous series, and back-to-back windows use every coarse sample
  once, so they lose no sensitivity.  A taper there only buys fine-channel leakage, the same trade
  as `--wola` today.  **A WOLA first stage works, with a catch.**  Its response droops across the
  coarse channel and its skirts are broad, so the fine channels near a coarse channel's edges
  alias in the neighbouring coarse channels' signal: −1 dB (12.5% taper) to −18 dB (50%) at the
  edge, even oversampled.  That costs 3-5% of continuum sensitivity, more at a residual delay.
  A line leaks no more power into other coarse channels than one WOLA FFT at the coarse
  resolution would, but it lands as a sharp ghost about O coarse channels away (up to −5.7 dB with
  a 12.5% taper) instead of broadening the neighbouring channels.
- **What removes the ghosts: an oversampled polyphase filter bank (PFB) as the first stage.**
  - Oversampled by O = 4/3 with 16 taps, or O = 2 with 4-8 taps.
  - Keep the central 1/O of each coarse channel's fine bins.
  - The alias images fall to −44 dB (8 taps, O = 4/3) and below −63 dB (the others).
  - The fine channels are flat across the coarse channel.
  - Continuum sensitivity within 0.15 / K of ideal, K fine channels a coarse channel (0.3% at
    K = 48), with the same leakage as one rectangular FFT.
  - A Vorbis WOLA second stage gives the same leakage and sensitivity as `--wola` today.
  - Two stages cost 2.2-3x the FFT arithmetic of one long FFT, plus a round trip of the coarse
    series through DRAM.
- **Against fewer channels in one stage** (the question that decides WOLA or PFB): with a WOLA
  first stage a line leaks exactly the same power into each other coarse channel as one WOLA FFT
  at the coarse resolution.  But it lands as a sharp ghost about O coarse channels from the line,
  not as a broadened neighbour; elsewhere the leakage is far lower.  A PFB removes the ghosts.
- **Decisions** at the end: the second stage is rectangular, two stages are an option for about
  2k-4k channels and up, DiFX's conventions are as assumed, and channel counts need not be powers
  of two.  Still open: WOLA or PFB for the first stage (both kept as options), and the oversampling.

## The design

Real-sampled band of B MHz (2B samples/s); the final channel count nchan; Nc coarse channels.

| stage | what | per |
|---|---|---|
| 1 (coarse kernel) | decode, integer delay shift, fringe rotation (per sample, as now); WOLA window (P = 1) or PFB weights on P × L1 samples, folded to L1 = 2 Nc; real FFT of L1 points; per coarse channel the oversampling derotation and the delay's phase at the channel's centre | coarse window: every H samples, O = L1 / H |
| DRAM | the coarse series, FP16, `[ant][band][coarse channel][coarse sample]` | |
| 2 (fine kernel) | N2-point FFT of N2 consecutive coarse samples; keep the central K = N2 / O bins; optionally the residual delay phase's window mean per fine channel; scale | stage-2 window = the XMAC's window |
| DRAM | FP16 spectra in the staged XMAC's order (`halfIndex`), as the end-to-end kernel writes them now | |
| XMAC | the staged XMAC, unchanged | |

- **The channel grid.**  An even-stacked stage 1 (a plain real FFT, coarse centres at c × B / Nc,
  c = 0 … Nc) with the fine bins k = −K/2 … K/2 − 1 of each coarse channel gives fine channels at
  j × B / nchan, j = 0 … nchan − 1, which is one long FFT's bin grid (and, I believe, DiFX's).  The DC and
  Nyquist coarse channels contribute only their in-band halves; nchan = Nc × K.
- **Channel counts.**  O = L1 / H = N2 / K must hold with H, N2 and K whole numbers.  With nchan
  and Nc both powers of two, K is one too, so O must be a power of two: O = 2 (or 1, which
  aliases).  O = 4/3 makes K, and so nchan, a multiple of 3 (3072, 12288, 49152, …); O = 8/7 a
  multiple of 7.
- **Where the delay is corrected.**
  - **Stage 1:** an integer-sample shift per coarse window, and the fractional part δ at each coarse
    channel's centre f_c.  Coarse channel c's sample n is multiplied by e^{−2πi f_c δ(t_n)}.
  - **The residual:** fine channel u coarse channels from the centre (|u| ≤ ½) is left a phase
    2π u δ / L1, with |δ| ≤ ½ sample.  It is bounded whatever the delay rate or the channel count
    (Delay accuracy).
  - **Stage 2 (optional):** can remove the residual's mean over each window, per fine channel.
  - **The fringe rotation** (the LO term) stays per sample, before stage 1.
- **A PFB does not affect the correction's correctness.**  It is a linear, time-invariant filter,
  the same at every station, so a delay is a phase e^{−2πi f δ} after it as before it, and the
  phase applied is exact at each channel's centre.
  - **Checked by simulation:** a two-stage PFB on a delayed signal leaves exactly 2π u δ / L1
    (to a few mrad) and full coherence for a static delay.
  - **The one rule:** a coarse sample's time is the centre of the whole P × L1-sample PFB window,
    its group delay, not the start of the window or of the L1-point fold.  That matters between
    stations only if they use different PFBs.

## Is WOLA the right first stage?

The two stages together make each fine channel's response the product of the stage-1 window's
response H1(f) and the stage-2 window's.  The stage-2 response repeats every 1/H in frequency (the
coarse sample rate), so whatever H1 passes at f ± m/H lands in the same fine channel.
- **The oversampling.**  Oversampling by O and keeping only the central 1/O of each coarse
  channel's fine bins puts the nearest image O − 0.5 coarse channels from the centre.
- **The window.**  H1 must be down by then.

![Stage 1: a coarse channel's response](../plots/two-stage-coarse.png)

A short-tapered WOLA window has the response of a rectangular window with slightly lower sidelobes:
it is still above −20 dB a whole coarse channel out.  A PFB (a windowed sinc P × L1 samples long,
folded into the FFT) is flat to the kept edge and falls to its stopband by O − 0.5.

![Across one coarse channel](../plots/two-stage-fine.png)

What that does to the fine channels, across one coarse channel:
- **WOLA:** the gain droops by 1.7-3 dB at the edges, and the edge channels take −1 to −18 dB of
  their power from the images.
- **A critically sampled PFB (O = 1):** flat inside, but at its edges two frequencies one coarse
  channel apart land in the same fine channel equally (−6 dB each).  Oversampling is what removes
  that, not taps.
- **The oversampled PFBs:** flat, with images at −90 dB (16 taps, O = 4/3) or below.

![A narrow line near a coarse channel's edge](../plots/two-stage-line.png)

A line near a coarse channel's edge shows the difference.  With a WOLA first stage it appears
again O coarse channels away on each side: at −11.5 / −12.9 dB with the 12.5% taper, and
−21 / −33 dB with the 50% taper (those images are at ±2 coarse channels, beyond the plot).  With
the oversampled PFB the images are at −100 dB or below (16 taps, O = 4/3; −68 dB with 4 taps at
O = 2), and the fine channels see the line as one rectangular FFT would.

## Leakage against frequency offset: two stages against fewer channels

The question that matters for a WOLA first stage: is a line's leakage, at a given frequency offset,
worse than with one WOLA FFT at the coarse resolution, the smaller channel count one would have
used before?

![Leakage against frequency offset](../plots/two-stage-offset.png)

The setup: 64 coarse channels, Vorbis 12.5% WOLA windows, O = 8/7 (56 fine channels a coarse
channel), rectangular stage 2.  For two stages, each offset is shown for the worst and the median
of a coarse channel's fine channels, each normalised to its own peak.
- **Within one coarse channel of the line,** two stages are 10-40 dB better.  One coarse WOLA
  channel is still in its main lobe there.
- **Further out,** a two-stage fine channel is generally 15-35 dB below the coarse response's
  sidelobes (−30 to −55 dB, against −13 to −20 dB).
- **Except at the alias ghosts.**  These are narrow, one fine channel wide, at ±O and ±2O coarse
  channels from the line.  For the worst-placed fine channels they reach −3 dB and −12 dB; for the
  median channel, about −25 dB.

How large a ghost is depends on where the line falls in its coarse channel.  Two stages: the
largest response more than half a coarse channel from the line.  One stage at 64 channels: the
line's response in that same coarse channel (`python3 two_stage.py ghost`):

| line's position from its coarse centre | WOLA stage 1: ghost, offset (coarse channel) | one stage, 64 channels, the same coarse channel | PFB 32 taps stage 1: largest beyond half a coarse channel |
|---|---|---|---|
| 0.0 | −19.4 dB at −1.14 (−1) | −19.4 dB | −105 dB |
| 0.1 | −13.1 dB at +1.15 (+1) | −13.1 dB | −34 dB (stage-2 sidelobe, own coarse channel) |
| 0.2 | −9.0 dB at +1.14 (+1) | −9.0 dB | −40 dB |
| 0.3 | −5.7 dB at +1.15 (+1) | −5.7 dB | −38 dB |
| 0.4 | −11.5 dB at +1.14 (+2) | −11.5 dB | −30 dB |
| 0.45 | −11.0 dB at +1.14 (+2) | −11.0 dB | −32 dB |
| 0.5 | −10.6 dB at −1.14 (−1) | −10.6 dB | −121 dB |

**So the answer is no more leakage, but in a different place.**
- **The power is identical.**  A line puts exactly the same power into each other coarse channel
  as one WOLA FFT at the coarse resolution does: the two columns agree to 0.1 dB.  It has to, since
  that power is the stage-1 window's response to the line, seen from the other coarse channel.
- **What changes is where it lands.**  One stage spreads it over the whole of that coarse channel,
  a line broadened into its neighbours at about the right frequency.  Two stages fold it into one
  fine channel about O coarse channels from the line: a sharp spurious line at the wrong frequency.
  For a line 0.3 coarse channels from its coarse centre, the −5.7 dB that one stage shows in the
  adjacent channel (0.7 coarse channels away) appears in two stages as a −5.7 dB ghost 1.15 coarse
  channels away.
- **Everywhere else two stages are much cleaner,** since the fine channels resolve what one coarse
  channel would have smeared.
- **The ghosts are the ~5% continuum loss.**  The same folded power, as noise, is the continuum
  sensitivity loss in the table above (0.951 with the 12.5% WOLA stage 1).
- **For continuum, or lines well inside the band of interest,** a WOLA first stage is no worse
  than the coarse channel count was.  For line work, a strong line anywhere can put a ghost
  several dB down one coarse channel away.  A PFB first stage removes the ghosts (everything
  beyond half a coarse channel at −30 dB or below, the rectangular stage 2's own sidelobes), for P
  multiply-adds a sample.

## Sensitivity

The continuum sensitivity is exact, not estimated.
- **The map.** The script builds the whole linear map from a circular stream of input samples to
  every kept fine channel and stage-2 window.  The summed cross-power's SNR for white noise is then
  tr(G) / ‖G‖_F, with G the map times its conjugate transpose.  That includes the correlations
  between fine channels that aliasing and oversampling create.
- **At a residual delay,** each channel keeps |(A D A^H)_jj| of its correlation, D the delay; the
  phase slope is the fringe fit's.
- **The check:** the one-stage rows reproduce the README's window-design table.

All values relative to the ideal:

| design | O | SNR at delay 0 | at W/64 | at W/16 | edge gain | alias, worst / median | leakage beyond 5 / 20 (median) |
|---|---|---|---|---|---|---|---|
| one stage, rectangular (mpifxcorr now) | 1 | 1.000 | 0.984 | 0.938 | 0 dB | — | −16.9 / −23.0 dB |
| one stage, Vorbis WOLA 12.5% (`--wola`) | 8/7 | 1.000 | 0.997 | 0.957 | 0 dB | — | −23.4 / −59.2 dB |
| two stage: rectangular stage 1 | 1 | 1.000 | 0.627 | 0.938 | −3.9 dB | +1.7 / −6.3 dB | −6.9 / −7.1 dB |
| two stage: WOLA 12.5% | 8/7 | 0.951 | 0.700 | 0.892 | −3.1 dB | −0.8 / −8.7 dB | −8.7 / −9.1 dB |
| two stage: WOLA 25% | 4/3 | 0.930 | 0.780 | 0.873 | −2.5 dB | −4.2 / −11.8 dB | −10.9 / −11.7 dB |
| two stage: WOLA 50% | 2 | 0.973 | 0.942 | 0.919 | −1.7 dB | −18.3 / −19.4 dB | −15.5 / −18.7 dB |
| two stage: PFB 8 taps | 1 | 0.968 | 0.893 | 0.908 | −6.0 dB | 0.0 / −41.9 dB | −17.3 / −24.2 dB |
| two stage: PFB 8 taps (46 dB) | 4/3 | 0.994 | 0.982 | 0.935 | −0.1 dB | −44.5 / −51.3 dB | −17.2 / −25.1 dB |
| **two stage: PFB 16 taps (85 dB)** | **4/3** | **0.994** | **0.981** | **0.934** | **0.0 dB** | **−74.5 / −93.2 dB** | **−17.2 / −25.0 dB** |
| two stage: PFB 32 taps (74 dB) | 8/7 | 0.995 | 0.979 | 0.934 | 0.0 dB | −58.4 / −86.5 dB | −17.1 / −24.5 dB |
| **two stage: PFB 4 taps (65 dB)** | **2** | **0.994** | **0.984** | **0.935** | **0.0 dB** | **−63.4 / −70.2 dB** | **−17.3 / −25.8 dB** |
| two stage: PFB 8 taps (123 dB) | 2 | 0.993 | 0.984 | 0.933 | 0.0 dB | −109 / −126 dB | −17.2 / −25.6 dB |
| **PFB 16 taps, O = 4/3, stage 2 Vorbis WOLA 12.5%** | **4/3** | **0.999** | **0.996** | **0.958** | **0.0 dB** | **−74.3 / −93.2 dB** | **−23.3 / −58.2 dB** |

How to read the table:
- **Model size.** 64 coarse channels.  The response figures use 64 fine bins a coarse channel
  before trimming; the sensitivities use 32 (K = 24, 28 or 16 kept).
- **Delays.** W is the stage-2 window, which is the XMAC's window.
- **The PFB's dB figure** is the stopband Kaiser's formula gives for its taps and transition.
- **The PFBs' zero-delay loss is an edge effect.** The rectangular stage-2 window's sidelobes reach
  across the coarse channels' edges, so the loss falls as 1 / K: 0.10-0.15 / K, measured at K = 6,
  12 and 24.  That is under 0.3% at K = 48 and negligible at the K of thousands of channels.
- **The WOLA and critically sampled designs' losses level off instead** (3-5%): that is their
  aliasing, which no N2 removes.

### What the continuum loss is, and what it is measured against

The metric:
- **What is summed.** The visibilities summed with equal weight over every channel and window, the
  continuum as channel averaging forms it.
- **The inputs.** White noise with a weak common signal.
- **The reference.** The ideal SNR from the same samples.

One stage scores exactly 1.000 at zero delay at any channel count.  A rectangular DFT is unitary,
and Vorbis WOLA weights add to one in power, so the channel width does not matter.

Two stages lose for two reasons when the first stage is WOLA:
- **Two frequencies share one output sample (most of the loss).**  A fine channel near a coarse
  edge contains its own frequency and, almost as strongly, an alias frequency O coarse channels
  away.
  - Both carry correlated signal, but the two components' noise also multiplies each other in
    X Y*.  That adds noise and no signal.
  - The two components are worth two independent samples, but they come out as one.
  - Separate channels give a noise variance of a² + b²; one channel holding both gives (a + b)².
  - This loss cannot be undone by weighting the channels.
- **The band is weighted unevenly (1.4 of the 4.9%).**  Each frequency's total weight over all the
  fine channels it appears in (its own and its alias) ranges 0.62-1.12 across a coarse channel.
  Summing such a band loses mean / rms = 0.986, the frequency-domain counterpart of a Tukey taper's
  uneven time weights.

**The loss does not depend on the number of coarse channels.**
- **Coarse channels:** with a WOLA first stage it is 0.9506 at 32, 64 and 128 coarse channels of
  real data (L1 = 64, 128, 256; 12.5% taper, O = 8/7), and 0.9694 for a 50% taper at O = 2.  The
  window is a fixed fraction of L1, so its response, in coarse channels, has the same shape at
  any L1.
- **Fine channels:** at 12.5% it does not depend on them either (0.9506 at K = 14 and 28).
- **The levers are the stage-1 filter's shape and the oversampling:** 0.951 at 12.5% (O = 8/7),
  0.930 at 25% (O = 4/3), 0.973 at 50% (O = 2), and 0.995 or better with a PFB.

(One stage loses nothing at zero delay with Vorbis tapers: 1.0000 against rectangular windows at
64, 256 and 1024 points, as the tapers are power-complementary; a Tukey 12.5% taper loses 0.65%.)

A PFB keeps each fine channel to one frequency range, so neither happens.  What is left is the
rectangular stage 2's sidelobes reaching across the coarse edges, 0.15 / K.

Against which single stage?  At a residual delay τ (input samples; the coarse window is 64
samples, the fine resolution 64 × 28 = 1792 channels):

| channelisation | τ = 0 | 1 | 4 | 16 |
|---|---|---|---|---|
| one stage, 64 channels, rectangular | 1.000 | 0.984 | 0.938 | 0.750 |
| one stage, 64 channels, Vorbis WOLA 12.5% | 1.000 | 0.997 | 0.957 | 0.743 |
| one stage, 1792 channels, rectangular | 1.000 | 0.999 | 0.998 | 0.991 |
| one stage, 1792 channels, Vorbis WOLA 12.5% | 1.000 | 1.000 | 1.000 | 0.999 |
| two stage, WOLA 12.5% stage 1, 64 × 28 | 0.951 | 0.948 | 0.918 | 0.774 |
| two stage, PFB 32 taps stage 1, 64 × 28 | 0.995 | 0.995 | 0.994 | 0.989 |

- **One stage, fewer channels:** at zero delay the same as more channels (1.000).  At a residual
  delay it loses more, because its window is shorter: about τ / L of the correlation for an
  L-sample rectangular window.
- **Two stages, WOLA first stage:** about 5% below one stage at the same final width.  It inherits
  the coarse window's sensitivity to residual delay, because the aliased and coarse-edge
  components decorrelate like the 64-sample window.  So it is close to one stage at the coarse
  resolution: 5% worse at small delays, slightly better (0.774 against 0.743) at 16 samples.
- **Two stages, PFB first stage:** behaves like one stage at the final resolution (0.989 against
  0.991 at 16 samples), less 0.15 / K.

![Continuum sensitivity against residual delay](../plots/two-stage-sensitivity.png)

At a residual delay the oversampled PFBs follow one rectangular FFT of the same resolution: the
composite fine-channel filter is as long as the stage-2 window, so short stage-1 windows do not
make it more delay-sensitive.  The aliasing designs do worse and oscillate: an alias image's phase
turns with the delay and comes back into phase at multiples of the stage-1 hop.  With a Vorbis
stage 2 the PFB matches `--wola`.

## Delay accuracy

![The fractional-sample correction against the delay rate](../plots/two-stage-delayrate.png)

**One stage.**  The fractional-sample correction uses one delay per window, its middle (fft-fusion
and mpifxcorr).
- **The drift:** channel j (frequency j B / nchan, window nchan / B long) drifts in phase by
  2π j × (delay rate) across the window, and keeps sinc(j × rate) of its amplitude.
- **Channels, not bandwidth:** the loss depends on the channel count per subband, not on the
  bandwidth.

Coherence kept, top channel / band average:

| channels per subband | 3 µs/s (Earth rotation) | 30 µs/s | 100 µs/s |
|---|---|---|---|
| 1024 | 1.0000 / 1.0000 | 0.9985 / 0.9995 | 0.9829 / 0.9943 |
| 4096 | 0.9998 / 0.9999 | 0.9754 / 0.9918 | 0.7461 / 0.9125 |
| 16384 | 0.9960 / 0.9987 | 0.6474 / 0.8766 | 0.1762 / 0.4238 |
| 65536 | 0.9376 / 0.9790 | 0.0172 / 0.3699 | 0.0479 / 0.1477 |
| 262144 | 0.2516 / 0.7169 | 0.0167 / 0.1280 | 0.0076 / 0.0476 |
| 1048576 | 0.0447 / 0.2583 | 0.0100 / 0.0408 | 0.0013 / 0.0146 |

The loss reaches 0.5% (top channel / band average) at these channel counts per subband:

| delay rate | channels per subband |
|---|---|
| 3 µs/s | 18400 / 31900 |
| 30 µs/s | 1840 / 3190 |
| 100 µs/s | 550 / 960 |

**Two stages.**  An integer shift per coarse window, and the fractional part corrected at each
coarse channel's centre, leave fine channel u (|u| ≤ ½ coarse channel) a phase 2π u δ / L1, with
|δ| ≤ ½ sample.  As δ sweeps a sample (the delay drifting), the channel keeps sinc(u / L1).  That
holds whatever the delay rate or the channel count, so the PFB can be small:

| stage-1 FFT points L1 (L1 / 2 coarse channels of real data) | loss, worst fine channel / average |
|---|---|
| 8 | 0.64% / 0.21% |
| 16 | 0.16% / 0.05% |
| 32 | 0.04% / 0.01% |
| 64 | 0.01% / 0.003% |

- **The bound holds:** a simulated two-stage PFB (16 taps, O = 4/3; L1 = 8, 16, 32) on a delayed
  signal leaves exactly the predicted phase.
- **Stage 2's optional correction** (the residual's mean per window and fine channel) reduces it
  further when δ changes slowly.
- **What sets the coarse channel count in practice** is the stage-2 FFT, N2 = O × (channels per
  subband) / Nc, which must fit cuFFTDx (16384 as built here).  65536 channels per subband at
  O = 4/3 need Nc ≥ 6 (so 8), and 2^20 need Nc ≥ 86 (so 128).

## Cost

Floating-point operations per input sample:
- **One FFT of N points:** about 2.5 log2 N.
- **Two stages:** about 2.5 O log2(N O) for the two FFTs, plus 2 P O for the PFB's weights.

| channels | one stage | PFB 16, O = 4/3 | PFB 4, O = 2 | PFB 8, O = 2 |
|---|---|---|---|---|
| 8192 | 35 | 91 | 91 | 107 |
| 65536 | 42.5 | 101 | 106 | 122 |
| 1048576 | 52.5 | 114 | 126 | 142 |

- **Arithmetic** is about 2-3x the FFT work of one FFT.  The FFT is not the whole kernel: on the
  A100 the end-to-end kernel is bound by the FMA pipe and the sincos calls, not the FFT alone
  (README.md), so the measured ratio will be smaller.
- **DRAM:** the coarse series, FP16, written and read once: 8 × O bytes per final channel value,
  against 8 bytes (FP16 spectra written, read by the XMAC) now: 2.3x the traffic at O = 4/3, 3x at
  O = 2.  At 10 stations, 8 bands of 64 MHz, that is 55 GB/s in real time, 3% of an A100.
- **Shared memory:** a stage-1 block decoding G consecutive coarse windows needs (G − 1) H + P L1
  samples, each decoded once and used by P × O windows.
- **A later option:** where N2 × Nc complex values fit in shared memory (nchan up to about
  8-16k at FP16), both stages fit in one block with no DRAM round trip.  For those channel counts
  the existing single-stage kernel is already accurate, so this is an optimisation, not the
  first target.

## Implementation and first results (RTX 2070 SUPER, 150 W + 1500 MHz, 2026-11-21)

PFB first stage only (the WOLA option dropped).  `twostagebench` (`twostage.cuh`, `twostage1.cu`,
`twostage2.cu`; `run-twostage.sh`):
- **Stage 1** (cuFFTDx, L1 = 32, 64, 128 points; 1, 2, 4 bits).
  - **Decode once.**  A block of 256 threads takes FPB consecutive coarse windows of one
    (antenna, band).  It decodes the block's span of samples once into shared memory: the level,
    the alternating sign by absolute parity, and the fringe rotation as a function of the absolute
    sample.
  - **Fold.**  Each thread folds one element position of 8 windows, so each tap is loaded once for
    8 windows.  It writes padded FP16 rows.
  - **FFT and corrections.**  The L1-point FFT at 8 elements a thread reads those rows.  The
    fractional-sample phase at the coarse centre and the oversampling derotation are added in fixed
    point, one sincos per output.  The coarse series is stored FP16, time-contiguous per coarse
    channel.
- **Stage 2** (cuFFTDx, N2 = 128-8192): FFT, keep K bins (the in-band half for the DC and Nyquist
  coarse channels), FP16 in the staged XMAC's layout.
- **The delay model, in the kernel:** an integer shift per coarse window, taken at the centre of its
  P x L1 window, and a continuous fringe phase.  Nothing is uploaded or read per coarse window: the
  kernel takes a delay polynomial per (antenna, subint) and a fringe polynomial per (antenna, band)
  (`TwoStageModel`), plus the clock terms per (antenna, band).
  - **Per block (thread 0, FP64):** the delay at window n0's centre, split into an integer and a
    fraction, and the local rate; the fringe phase and rate at the block's first sample; the clock
    terms of the fractional-sample phase.
  - **Per window (FP32):** window n0 + k's delay is the block's fraction + rate x kH (+ d2 (kH)^2).
    A block spans at most ~3500 samples, so this is at most 0.035 samples at 10 us/s and FP32
    holds it to ~1e-9 samples.  Each thread that needs a window's shift or fraction (the fold, the
    correction) recomputes it, with explicit round-to-nearest FMAs so every thread gets the same
    bits.  Per-window arrays in shared memory instead (640 bytes a block) were slower where the
    kernel sits at Turing's 64 KB of shared memory an SM: 32 coarse channels, 4/3, 4 taps,
    0.427 ms/ms against 0.385 (by the arithmetic, 16.3 KB a block fits 4 blocks an SM and 17.0 KB
    only 3; not profiled).
  - **The limit:** a block's span has 16 samples of slack for shifts, so |rate| up to ~0.01 (100x
    a spacecraft's); beyond that the kernel refuses the block and the check fails.
  - **Before** (the "setup kernel" column): the host made an integer sample index, a fractional
    delay and fringe terms per coarse window, and a separate FP64 kernel turned them into a
    16-byte phase per (window, band) for stage 1 to read back: more bytes than the 2-bit data at
    16-96 samples a window.
- **The check:** a double-precision host reference of the whole chain (decode from the packed bytes,
  fold, DFT, corrections, stage-2 DFT) on 4 series per run, coarse channels 0 and Nc among them.
  The bench nulls every per-window array stage 1 used to read.  Also checked at rates of
  ±1e-4 and accelerations of ±0.5 s/s^2 (unphysical, to exercise the in-block quadratic term).
  All 18 configurations pass at about 4e-4 rms (FP16 rounding of the span, the folded rows, the
  coarse series and the spectra).
- **What got it there.**  The first version decoded each sample once per window it is in (P x O
  times) and was 6x slower.  Then: decoding once per block; moving the fold off the FFT's register
  layout (its half-warps spanned windows whose reads fell in the same banks); FP16 working buffers
  (FP32 allowed one 256-thread block an SM, 25% occupancy); and loading each tap once for 8 windows.
  4.5x in all.

6 antennas, 8 bands of 64 MHz, 2 bits, 16 stage-2 windows; ms per ms of data
(`results-local/ar313-adeller-twostage-20261121-120407`; with the setup kernel,
`results-local/ar313-adeller-twostage-20261121-002831`).
- **Slowdown** is the total over the single-stage end-to-end kernel at its largest size, 8192
  channels (FP16 spectra, the fastest variant, 0.282 ms/ms).
- **Against gpudifx today** is at the same channel count: 1.536 ms/ms at 49152 channels, 1.376 at
  65536.
- **Loss** is the zero-delay continuum loss: the exact model's a + b / K at this K, plus the
  average fractional-delay residual (0.013, 0.003, 0.001% at Nc = 16, 32, 64).
- **Alias** is the worst image and **ripple** the passband ripple (both for line work).

| Nc | O | taps | channels | stage 1 | stage 2 | total | total, setup kernel | slowdown | against gpudifx | loss | alias | ripple |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 4/3 | 4 | 49152 | 0.462 | 0.095 | 0.551 | 0.618 | 1.95 | 2.78x | 0.22% | −25 dB | 0.76 dB |
| 16 | 4/3 | 8 | 49152 | 0.506 | 0.094 | 0.591 | 0.662 | 2.09 | 2.60x | < 0.1% | −39 dB | 0.14 dB |
| 16 | 4/3 | 16 | 49152 | 0.597 | 0.093 | 0.685 | 0.753 | 2.43 | 2.24x | < 0.1% | −65 dB | 0.004 dB |
| 32 | 4/3 | 4 | 49152 | 0.385 | 0.091 | 0.468 | 0.500 | 1.66 | 3.28x | 0.22% | −25 dB | 0.76 dB |
| **32** | **4/3** | **8** | **49152** | **0.490** | **0.091** | **0.572** | **0.601** | **2.03** | **2.69x** | **< 0.1%** | **−39 dB** | **0.14 dB** |
| **32** | **4/3** | **16** | **49152** | **0.603** | **0.090** | **0.682** | **0.712** | **2.42** | **2.25x** | **< 0.1%** | **−65 dB** | **0.004 dB** |
| 64 | 4/3 | 4 | 49152 | 0.497 | 0.089 | 0.578 | 0.590 | 2.05 | 2.66x | 0.22% | −25 dB | 0.76 dB |
| 64 | 4/3 | 8 | 49152 | 0.570 | 0.089 | 0.649 | 0.662 | 2.30 | 2.37x | < 0.1% | −39 dB | 0.14 dB |
| 64 | 4/3 | 16 | 49152 | 0.863 | 0.089 | 0.937 | 0.954 | 3.32 | 1.64x | < 0.1% | −65 dB | 0.004 dB |
| 16 | 2 | 4 | 65536 | 0.630 | 0.178 | 0.803 | 0.903 | 2.85 | 1.71x | < 0.05% | −64 dB | 0.011 dB |
| 16 | 2 | 8 | 65536 | 0.692 | 0.178 | 0.870 | 0.971 | 3.08 | 1.58x | < 0.05% | −115 dB | 0 |
| 16 | 2 | 16 | 65536 | 0.828 | 0.178 | 1.005 | 1.103 | 3.56 | 1.37x | < 0.1% | −207 dB | 0 |
| **32** | **2** | **4** | **65536** | **0.523** | **0.133** | **0.651** | **0.699** | **2.31** | **2.11x** | **< 0.05%** | **−64 dB** | **0.011 dB** |
| 32 | 2 | 8 | 65536 | 0.603 | 0.133 | 0.732 | 0.777 | 2.60 | 1.88x | < 0.05% | −115 dB | 0 |
| 32 | 2 | 16 | 65536 | 0.834 | 0.133 | 0.966 | 1.004 | 3.43 | 1.42x | < 0.1% | −207 dB | 0 |
| 64 | 2 | 4 | 65536 | 0.674 | 0.131 | 0.804 | 0.819 | 2.85 | 1.71x | < 0.05% | −64 dB | 0.011 dB |
| 64 | 2 | 8 | 65536 | 0.782 | 0.131 | 0.912 | 0.929 | 3.23 | 1.51x | < 0.05% | −115 dB | 0 |
| 64 | 2 | 16 | 65536 | 0.992 | 0.130 | 1.122 | 1.143 | 3.98 | 1.23x | < 0.1% | −207 dB | 0 |

- **Cost.** Two stages cost 1.7-4x the single-stage kernel per ms of data, and are 1.2-3.3x faster
  than gpudifx today at the same channel count.
- **The in-kernel delay model** saved the whole setup (0.015-0.09 ms/ms, 2-11% of the total, most at
  16 coarse channels) and left stage 1's own time unchanged (within 1-2%).
- **The coarse channel count.** 32 coarse channels is the sweet spot.
  - At 16, stage 1 is up to 20% slower than at 32 (4 taps; the same at 16 taps) and the stage-2
    FFT (8192 points at O = 2) costs more.  With the setup gone, 16 is within 5% of 32 at 8 and 16
    taps at O = 4/3.
  - At 64, stage 1's blocks hold only 16 windows (L1 = 128), so the overlap between blocks is
    decoded up to 3x over, and 16 taps of 128 points make the fold heavy.
- **The oversampling** costs about 1.3-1.5x (O = 2 against 4/3: 1.5x the coarse windows and twice
  the coarse series), close to the window ratio.
- **Taps** cost 10-20% from 4 to 8, more to 16 (the fold is P x O multiply-adds per sample).
- **The sensitivity loss is negligible everywhere but one case.** 4 taps at O = 4/3 lose 0.22% and
  have −25 dB alias images and 0.76 dB of ripple: its 27 dB Kaiser stopband is too little for a 4/3
  transition.  Everything else is within the exact model's ±0.07% of zero; the edge term 0.13 / K
  is 0.002-0.017% at these K.
- **The choices** (bold):
  - 32 coarse channels, O = 4/3, 8 taps: 2.0x the single stage, < 0.1% loss, −39 dB images.
  - 16 taps if lines matter: 2.4x, −65 dB images.
  - O = 2, 4 taps for powers of two: 2.3x, −64 dB images, 65536 channels.
- **Not yet optimised:**
  - Stage 1's blocks overlap by P x L1 − H samples, decoded twice.  A block sliding along the
    series, keeping its tail, would remove that.
  - The A100 and RTX PRO 6000 are still to run.

### At about 8k channels: the like-for-like comparison

The single-stage kernel's largest size is 8192 channels, so the slowdown above compares 49152-65536
channels with 8192.  The same sweep at about 8192 channels (`NCHAN=8192 NWIN=128 ./run-twostage.sh`,
the same data per run; `results-local/ar313-adeller-twostage-8192-20261121-120452`; with the setup
kernel, `results-local/ar313-adeller-twostage-8192-20261121-104326`):
- **The channel counts.** O = 2 gives 8192 exactly at every Nc (N2 = 1024, 512, 256 at Nc = 16, 32,
  64).  O = 4/3 cannot: with power-of-two L1 and N2 the nearest is 6144 (N2 = 512, 256, 128),
  or 12288.  16 coarse channels with a 512-point stage 2 is 4096 channels at O = 2, 6144 at 4/3.
- **Loss** is the floor a (±0.07% model uncertainty; 0.22% for 4 taps at 4/3) plus the edge term
  b / K, b = 0.12-0.14 at O = 4/3 and about 0.09 at O = 2, plus the fractional-delay residual.  At
  these K the edge term is no longer negligible: it is larger than the fractional-delay residual,
  so 16 coarse channels lose less in all than 32.
- **gpudifx today** is 1.540 ms/ms at 6144 channels (a 12288-point cuFFT) and 1.088 at 8192.

| Nc | O | taps | N2 | channels | stage 1 | stage 2 | total | total, setup kernel | slowdown | against gpudifx | edge b/K | frac. delay |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 4/3 | 4 | 512 | 6144 | 0.462 | 0.094 | 0.551 | 0.627 | 1.95 | 2.79x | 0.034% | 0.013% |
| 16 | 4/3 | 8 | 512 | 6144 | 0.506 | 0.093 | 0.590 | 0.661 | 2.09 | 2.61x | 0.032% | 0.013% |
| 16 | 4/3 | 16 | 512 | 6144 | 0.602 | 0.093 | 0.684 | 0.752 | 2.43 | 2.25x | 0.036% | 0.013% |
| 32 | 4/3 | 4 | 256 | 6144 | 0.385 | 0.079 | 0.456 | 0.488 | 1.62 | 3.37x | 0.068% | 0.003% |
| **32** | **4/3** | **8** | **256** | **6144** | **0.490** | **0.079** | **0.560** | **0.590** | **1.98** | **2.75x** | **0.064%** | **0.003%** |
| 32 | 4/3 | 16 | 256 | 6144 | 0.603 | 0.079 | 0.671 | 0.700 | 2.38 | 2.29x | 0.071% | 0.003% |
| 64 | 4/3 | 4 | 128 | 6144 | 0.497 | 0.105 | 0.594 | 0.606 | 2.10 | 2.59x | 0.136% | 0.001% |
| 64 | 4/3 | 8 | 128 | 6144 | 0.570 | 0.105 | 0.665 | 0.678 | 2.36 | 2.31x | 0.128% | 0.001% |
| 64 | 4/3 | 16 | 128 | 6144 | 0.855 | 0.106 | 0.954 | 0.971 | 3.38 | 1.61x | 0.142% | 0.001% |
| 16 | 2 | 4 | 1024 | 8192 | 0.630 | 0.136 | 0.761 | 0.860 | 2.70 | 1.43x | 0.018% | 0.013% |
| 16 | 2 | 8 | 1024 | 8192 | 0.692 | 0.136 | 0.831 | 0.929 | 2.95 | 1.31x | 0.017% | 0.013% |
| 16 | 2 | 16 | 1024 | 8192 | 0.828 | 0.136 | 0.964 | 1.061 | 3.42 | 1.13x | 0.018% | 0.013% |
| **32** | **2** | **4** | **512** | **8192** | **0.528** | **0.133** | **0.651** | **0.699** | **2.31** | **1.67x** | **0.036%** | **0.003%** |
| 32 | 2 | 8 | 512 | 8192 | 0.600 | 0.133 | 0.733 | 0.777 | 2.60 | 1.49x | 0.034% | 0.003% |
| 32 | 2 | 16 | 512 | 8192 | 0.834 | 0.133 | 0.967 | 1.005 | 3.43 | 1.13x | 0.036% | 0.003% |
| 64 | 2 | 4 | 256 | 8192 | 0.675 | 0.111 | 0.785 | 0.799 | 2.78 | 1.39x | 0.073% | 0.001% |
| 64 | 2 | 8 | 256 | 8192 | 0.782 | 0.111 | 0.892 | 0.909 | 3.16 | 1.22x | 0.068% | 0.001% |
| 64 | 2 | 16 | 256 | 8192 | 0.992 | 0.111 | 1.103 | 1.124 | 3.91 | 0.99x | 0.073% | 0.001% |

- **The slowdown hardly changes with the channel count** (1.98 here against 2.03 at 49152 for
  Nc = 32, O = 4/3, 8 taps).  Stage 1 is 80-90% of the total, and its cost per sample does not
  depend on the final channel count; stage 2 is 0.08-0.14 ms/ms at either size.  The ~2x is the
  cost of the PFB itself, against a single stage that does one pass over the data.
- **16 coarse channels is not the fastest, but close.** With the setup in the kernel it is 5% behind
  32 at 8 taps and 2% at 16 (O = 4/3); at 4 taps its stage 1 is still 20% slower.  It also loses
  less (the edge term above), so at about 8k channels 16 x 8 taps at 4/3 (2.09x) is as good a
  choice as 32 x 8 (1.98x).
- **Choices at about 8k:** 32 (or 16) coarse channels, O = 4/3, 8 taps (6144 channels, 2.0x,
  < 0.1% loss); O = 2, 4 taps for 8192 exactly (2.3x).

### A100-SXM4-80GB (OzSTAR, gina301), about 64k channels

`fusebench-ozstar.sbatch` with `SCRIPTS=run-twostage.sh`, SLURM job 21315678, commit 39e0886, all 18
passing (`results-local/gina301-twostage-20261121-123215`).  The single-stage kernel at 8192
channels: 0.0951 ms/ms; gpudifx today 0.326 ms/ms at 49152 channels, 0.249 at 65536.

| Nc | O | taps | channels | stage 1 | stage 2 | total | slowdown | against gpudifx | 2070 slowdown |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 4/3 | 4 | 49152 | 0.1237 | 0.0284 | 0.1520 | 1.60 | 2.15x | 1.95 |
| 16 | 4/3 | 8 | 49152 | 0.1243 | 0.0258 | 0.1500 | 1.58 | 2.18x | 2.09 |
| 16 | 4/3 | 16 | 49152 | 0.1507 | 0.0258 | 0.1765 | 1.86 | 1.85x | 2.43 |
| 32 | 4/3 | 4 | 49152 | 0.1082 | 0.0254 | 0.1335 | 1.40 | 2.45x | 1.66 |
| 32 | 4/3 | 8 | 49152 | 0.1239 | 0.0254 | 0.1492 | 1.57 | 2.19x | 2.03 |
| 32 | 4/3 | 16 | 49152 | 0.1546 | 0.0254 | 0.1798 | 1.89 | 1.82x | 2.42 |
| 64 | 4/3 | 4 | 49152 | 0.1211 | 0.0312 | 0.1522 | 1.60 | 2.15x | 2.05 |
| 64 | 4/3 | 8 | 49152 | 0.1413 | 0.0312 | 0.1725 | 1.81 | 1.89x | 2.30 |
| 64 | 4/3 | 16 | 49152 | 0.1820 | 0.0312 | 0.2131 | 2.24 | 1.53x | 3.32 |
| 16 | 2 | 4 | 65536 | 0.1692 | 0.0610 | 0.2079 | 2.19 | 1.20x | 2.85 |
| 16 | 2 | 8 | 65536 | 0.1901 | 0.0609 | 0.2506 | 2.63 | 0.99x | 3.08 |
| 16 | 2 | 16 | 65536 | 0.2333 | 0.0608 | 0.2941 | 3.09 | 0.85x | 3.56 |
| 32 | 2 | 4 | 65536 | 0.1477 | 0.0362 | 0.1837 | 1.93 | 1.35x | 2.31 |
| 32 | 2 | 8 | 65536 | 0.1712 | 0.0362 | 0.2073 | 2.18 | 1.20x | 2.60 |
| 32 | 2 | 16 | 65536 | 0.2170 | 0.0362 | 0.2531 | 2.66 | 0.98x | 3.43 |
| 64 | 2 | 4 | 65536 | 0.1677 | 0.0364 | 0.2039 | 2.14 | 1.22x | 2.85 |
| 64 | 2 | 8 | 65536 | 0.1975 | 0.0364 | 0.2338 | 2.46 | 1.06x | 3.23 |
| 64 | 2 | 16 | 65536 | 0.2584 | 0.0364 | 0.2946 | 3.10 | 0.84x | 3.98 |

- **The slowdown is smaller on the A100:** 1.4-3.1x against 1.7-4x on the 2070.  32 coarse
  channels, O = 4/3: 1.40x at 4 taps, 1.57x at 8.
- **Against gpudifx today** the margin is smaller (0.84-2.45x): gpudifx's FP64 decode and cuFFT
  run well on the A100.  At O = 2 with 16 taps two stages are slower than gpudifx today; at
  O = 4/3, 8 taps, 2.2x faster.
- **16 and 32 coarse channels tie at 8 taps, O = 4/3** (0.150 and 0.149 ms/ms).  16's stage 1 costs
  the same at 4 and 8 taps (0.124), so the fold is not what limits it there (not profiled).  At O = 2,
  16's 8192-point stage 2 (0.061 against 0.036 ms/ms) puts it behind.
- **The choices hold:** 32 coarse channels, O = 4/3, 8 taps (1.6x the single stage); 16 taps for
  lines (1.9x); O = 2, 4 taps for powers of two (1.9x).

### A100-SXM4-80GB (OzSTAR, gina17), about 8k channels

SLURM job 21316333, commit 4ae0c67, all 18 passing
(`results-local/gina17-twostage-8192-20261121-124621`).  The single-stage kernel at 8192 channels:
0.0952 ms/ms; gpudifx today 0.221 ms/ms at 6144 channels, 0.1995 at 8192.

| Nc | O | taps | channels | stage 1 | stage 2 | total | slowdown | against gpudifx | 2070 slowdown |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 4/3 | 4 | 6144 | 0.1236 | 0.0274 | 0.1508 | 1.58 | 1.47x | 1.95 |
| 16 | 4/3 | 8 | 6144 | 0.1327 | 0.0265 | 0.1591 | 1.67 | 1.39x | 2.09 |
| 16 | 4/3 | 16 | 6144 | 0.1609 | 0.0265 | 0.1872 | 1.97 | 1.18x | 2.43 |
| 32 | 4/3 | 4 | 6144 | 0.1083 | 0.0214 | 0.1295 | 1.36 | 1.71x | 1.62 |
| 32 | 4/3 | 8 | 6144 | 0.1240 | 0.0214 | 0.1452 | 1.52 | 1.52x | 1.98 |
| 32 | 4/3 | 16 | 6144 | 0.1546 | 0.0214 | 0.1757 | 1.85 | 1.26x | 2.38 |
| 64 | 4/3 | 4 | 6144 | 0.1211 | 0.0242 | 0.1452 | 1.52 | 1.52x | 2.10 |
| 64 | 4/3 | 8 | 6144 | 0.1413 | 0.0243 | 0.1655 | 1.74 | 1.34x | 2.36 |
| 64 | 4/3 | 16 | 6144 | 0.1820 | 0.0242 | 0.2061 | 2.16 | 1.07x | 3.38 |
| 16 | 2 | 4 | 8192 | 0.1529 | 0.0467 | 0.1994 | 2.09 | 1.00x | 2.70 |
| 16 | 2 | 8 | 8192 | 0.1719 | 0.0467 | 0.2184 | 2.29 | 0.91x | 2.95 |
| 16 | 2 | 16 | 8192 | 0.2112 | 0.0467 | 0.2578 | 2.71 | 0.77x | 3.42 |
| 32 | 2 | 4 | 8192 | 0.1477 | 0.0349 | 0.1826 | 1.92 | 1.09x | 2.31 |
| 32 | 2 | 8 | 8192 | 0.1713 | 0.0349 | 0.2061 | 2.16 | 0.97x | 2.60 |
| 32 | 2 | 16 | 8192 | 0.2171 | 0.0349 | 0.2519 | 2.65 | 0.79x | 3.43 |
| 64 | 2 | 4 | 8192 | 0.1678 | 0.0301 | 0.1977 | 2.08 | 1.01x | 2.78 |
| 64 | 2 | 8 | 8192 | 0.1976 | 0.0301 | 0.2276 | 2.39 | 0.88x | 3.16 |
| 64 | 2 | 16 | 8192 | 0.2585 | 0.0301 | 0.2885 | 3.03 | 0.69x | 3.91 |

- **The slowdown at 8k is what it is at 64k** (32 coarse channels, O = 4/3, 8 taps: 1.52x against
  1.57x), as on the 2070.  Best: 1.36x (32, 4/3, 4 taps).
- **Against gpudifx today** two stages are 1.1-1.7x faster at 6144 channels (O = 4/3) and 0.7-1.1x at
  8192 (O = 2).  But 8192 channels is the single stage's range, where it is 2.1x faster than
  gpudifx today: two stages are only for what one cannot do.
- **16 coarse channels** is 10% behind 32 at 8 taps here (0.159 against 0.145 ms/ms; tied at 64k):
  its stage 1 is 7% slower (0.133 against 0.124) and its 512-point stage 2 costs more than 32's
  256-point one (0.027 against 0.021).

## Implementation plan

1. **Python reference.**  A numpy model of the whole two-stage chain from a sample series (both
   stations, delays and rates), checked against `two_stage.py`'s predicted responses (a tone
   sweep: gain, alias levels) and against one long FFT where they should agree.  It is the GPU
   kernels' reference for small cases.
2. **Stage-1 kernel** (`coarse_inst.cu`, one instantiation per L1 and P, cuFFTDx; P = 1 is a WOLA
   window, P > 1 a PFB, so both first stages are the same kernel with a different weight table).
   - **Block:** G consecutive coarse windows of one (antenna, band).
   - **Input:** the packed data decoded once into shared memory with the lean decode, with an
     integer shift per coarse window.
   - **Per sample:** fringe rotation (as now).
   - **Per coarse window:** the P-tap fold to L1 samples, then a real FFT of L1 points.
   - **Per coarse channel:** derotation × delay phase at the channel's centre, then a scale.
   - **Output:** FP16, time fastest per coarse channel.
   - **History:** the PFB needs (P − 1) L1 samples before each subint, from the previous one (in
     mpifxcorr, from the datastream buffer).
   - **Invalid data:** coarse samples whose input is invalid are written as zeros.
3. **Stage-2 kernel.**
   - **Per (antenna, band, coarse channel, stage-2 window):** an N2-point complex FFT (cuFFTDx,
     batched in the block), keeping K bins.
   - **Per fine channel (optional):** the window mean of the residual delay phase.
   - **Correction:** none is needed.  The PFB's passband ripple is under 0.001 dB at 16 taps
     (O = 4/3) and 0.011 dB at 4 taps (O = 2); only the 8-tap O = 4/3 design reaches 0.09 dB.
   - **Output:** FP16 into the staged XMAC's layout, unchanged downstream; a stage-2 window's data
     weight is the fraction of its coarse samples that were valid.
4. **fusebench `--two-stage Nc,P,O`**, an option beside the single-stage kernel (selected for about
   2k-4k channels and up, as DiFX would select it).
   - **Timing:** against the single-stage kernel up to 8192 channels and cuFFT beyond.
   - **Checks:** a tone sweep against `two_stage.py`; a double-precision reference of the same
     chain on a few spectra; the whole chain through the staged XMAC against gpudifx's XMAC (as
     now); and a large synthetic delay rate, two stages against one.
5. **Benchmarks.** nchan 4k-1M, Nc 16/32/64, (P, O) in {(16, 4/3), (4, 2), (8, 2)}, on the 2070,
   the A100 and the RTX PRO 6000.
6. **Later.**
   - Both stages in one block for small N2 × Nc.
   - Zoom bands by skipping the coarse channels not needed: stage 2 runs only where a zoom band is.
   - The phase-centres work (`../../phase-centres`), which wants exactly these fine channels.

## Decisions

Answered (2026-11-20):
- **The second stage is rectangular.**
- **DiFX's channels are the long FFT's bins,** with DC in channel 0.  The even-stacked design above
  gives exactly that grid.
- **mpifxcorr also corrects the fractional sample once per FFT window,** so the delay-rate
  analysis applies to it as to fft-fusion.
- **Two stages are an option, not the one path.**  One stage whenever possible.  Two stages, with
  a moderate number of PFB coarse channels, when the resolution needs more channels per subband
  than one cuFFTDx FFT gives, or when the delay rate makes one stage's fractional-sample
  correction lose (spacecraft VLBI); see In short.

- **DiFX can use channel counts that are not powers of two,** so any oversampling will do
  (O = 8/7: multiples of 7; O = 4/3: multiples of 3).

Still open:
1. **The first stage: WOLA or a PFB,** both kept as options for now (one kernel, different weight
   tables).  WOLA gives the ghosts and the ~5% continuum loss above; a PFB costs P multiply-adds a
   sample more and has neither.
2. **The oversampling, which sets the channel counts.**  O = L1 / H = N2 / K must be whole-number
   ratios.
   - The 12.5% WOLA (O = 8/7) makes channel counts multiples of 7 (3584 = 64 × 56, 7168, …).
   - O = 4/3 makes them multiples of 3.
   - Powers of two need O = 2: a 50% WOLA taper, or a 4-8-tap PFB, at 2x the stage-1 FFTs and the
     coarse series' traffic.
