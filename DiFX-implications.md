# What this work means for DiFX

A quick-glance summary (2026-11-24) of how fast each prototyped change should run once it is in
GPU-DiFX, against gpudifx today, and how many GPUs some real configurations would need. The
details are in each sub-project's `FINDINGS.md`.

**In one line:** the new continuum chain is **8-9x faster than gpudifx today** on an RTX PRO 6000
(6x on an A100, 8x on the 2070). That speed-up pays for features gpudifx cannot do at all:
- pulsar gating or 16 bins, and 100 phase centres, each cost **10-35% of gpudifx's continuum time**;
- 1000 phase centres or 32 bins cost about 20-55%;
- only 128 bins comes close to gpudifx's continuum time.

## How to read the numbers

- **Times are GPU kernel time per unit of data** (ms per ms of data) for the station stage plus the
  cross-multiply-accumulate (XMAC), with every check passing.
  - A load of 1.0 means one GPU is exactly real time.
  - Host-to-device copies, the CPU side of mpifxcorr, autocorrelations and output are not
    included (see "Caveats").
- **GPU count:** ⌈load / 0.7⌉. Kernels may use at most 70% of each GPU's time; the rest is for
  copies, the rest of mpifxcorr, and margin.
- **The GPU is the RTX PRO 6000 Blackwell Server Edition** (OzSTAR gina18, 450 W). The continuum
  numbers were measured on it.
- **The pulsar, phase-centre and two-stage numbers were measured on the A100 and the 2070.** Each
  was measured as a factor over the XMAC or the station stage. The A100 and RTX PRO 6000 columns
  combine that factor with each GPU's own station-to-XMAC balance (S/X):
  - A100: S/X is 3.2 at 128-1024 channels, about 2.8 at 8k channels.
  - RTX PRO 6000: S/X is 1.6. Its XMAC is a larger share of the chain, so the same binned or
    phase-centre XMAC costs relatively more there.
  - The RTX PRO 6000 column is therefore the cautious one. Its 128 MB L2 may well help the
    high-DM binning, which suffers from DRAM re-reads, but that is unmeasured.
- **Reference configuration:** 10 stations x 4 Gbps, 2-bit, dual polarisation, i.e. 16 x 64 MHz
  bands per station. 128 channels a band (0.5 MHz) unless stated; 1024 channels a band costs
  about 25% more.

## Continuum: the new chain against gpudifx today

10 stations x 4 Gbps, 2-bit; GPU load and the GPUs needed for real time:

| GPU | gpudifx today, 128 / 1024 ch | new chain, 128 / 1024 ch | speed-up | GPUs, today → new |
|---|---|---|---|---|
| RTX 2070 SUPER (capped at 150 W, clock locked at 1500 MHz) | 5.2 / 5.1 | 0.70 / 0.84 | 7.5x / 6.0x | 8 → 2 |
| A100-SXM4-80GB | 1.18 / 1.14 | 0.20 / 0.24 | 5.9x / 4.7x | 2 → 1 |
| **RTX PRO 6000 Blackwell** | **0.86 / 0.91** | **0.095 / 0.12** | **9.1x / 7.6x** | **2 → 1** (load 10-12%) |

What each part contributes (10 stations, 128 channels):

| change | what it replaces | speed-up of that stage: 2070 / A100 / RTX PRO 6000 |
|---|---|---|
| **Fused station kernel** ([fft-fusion](fft-fusion/FINDINGS.md)): VDIF decode, fringe rotation, FFT and fractional-sample correction in one cuFFTDx kernel from the packed data, writing FP16 spectra | gpudifx's decode + fringe kernel (FP64), cuFFT, rotation kernel (FP64) | 7.3x / 4.1x / 10.3x |
| **Staged tensor-core XMAC** ([xmac](xmac/FINDINGS.md)): FP16 spectra rearranged in shared memory, mma.sync; any station band layout through a map | gpudifx's FP32 XMAC | 9.0x / 12x / 8.1x |
| **The two together** | | 7.8x / 5.9x / 9.1x |

The gain grows with the number of stations, because the XMAC's share grows: 13-18x at 32-64
stations on the RTX PRO 6000. Above about 64-100 stations the staged XMAC trails a tensor XMAC
fed by a station kernel that writes the tensor layout (xmac/BACKLOG.md, "Large N"). Precision:
108 shapes (1/2/4 bits x 4-256 stations x 32-4096 channels) gave no wrong results.

## Each effect, against the new continuum chain and against gpudifx today

10 stations x 4 Gbps, 2-bit, RTX PRO 6000. "x new chain" is the run time over the new continuum
chain (A100 measured; RTX PRO 6000 estimated, as above). "x gpudifx" is the run time over
gpudifx today's continuum run of the same data.

| effect | gpudifx today | x new chain: A100 / RTX PRO 6000 | x gpudifx | load | GPUs |
|---|---|---|---|---|---|
| Continuum (the new chain) | yes | 1 / 1 | 0.11-0.13 | 0.10-0.12 | 1 |
| **Pulsar gate** (1.5 or 500 ms pulsar) | no (CPU mpifxcorr only) | 0.91-1.10 / 0.85-1.16 | 0.09-0.15 | 0.08-0.14 | 1 |
| **16 bins, scrunched** (summed on - off) | no | 1.06-1.11 / 1.10-1.18 | 0.12-0.15 | 0.10-0.14 | 1 |
| **16 separate bins**, millisecond pulsar, DM 30-300 | no | 1.71-2.09 / 2.1-2.8 | 0.24-0.36 | 0.20-0.33 | 1 |
| 16 separate bins, 500 ms pulsar | no | 1.33-1.63 / 1.5-2.0 | 0.17-0.26 | 0.15-0.24 | 1 |
| **32 separate bins**, millisecond pulsar, DM 30 | no | 2.00-2.07 / 2.6-2.7 | 0.29-0.36 | 0.25-0.33 | 1 |
| 32 separate bins, millisecond pulsar, DM 300 | no | 2.95 / 4.1 | 0.46-0.55 | 0.39-0.50 | 1 |
| 128 separate bins, millisecond pulsar | no | 3.4-5.1 / 4.9-7.7 | 0.53-1.0 | 0.46-0.92 | 1-2 |
| **Phase centres: 10** (8k channels, 100 windows a piece) | no (CPU mpifxcorr only) | 1.04 / 1.06 | about 0.13 | 0.13 | 1 |
| **Phase centres: 100** | no | 1.10 / 1.15 | about 0.15 | 0.14 | 1 |
| Phase centres: 1000 | no | 1.7 / 2.0 | about 0.27 | 0.24 | 1 |
| **Two-stage channelisation** (PFB then FFT; above 8192 channels a subband, or high delay rates) | long cuFFTs only | 1.4 / 1.3-1.4, against one stage | about 0.3-0.7 at the same resolution (station stage 0.7-2.5x faster than gpudifx's on the A100) | about 0.16 | 1 |
| WOLA windows (12.5% Vorbis taper; optional, for channel isolation) | no | 1.18-1.27 / 1.14-1.22 | 0.13-0.16 | 0.11-0.15 | 1 |

Notes on the effects:
- **Pulsar binning** ([pulsar-binning](pulsar-binning/FINDINGS.md)) follows mpifxcorr's rule
  exactly: each (FFT window, channel) goes into one bin, and arbitrary bin edges are allowed.
  - The gate and scrunched bins weight the XMAC's input in a single pass. A gate skips the blocks
    with no on-pulse window.
  - Separate bins sort each channel's windows by bin. The XMAC then gathers them straight from the
    station kernel's usual output, so the station kernel does not change.
  - The target was a gate, or up to 16-32 bins of a millisecond pulsar at DM of hundreds, within
    2x of continuum at 10 stations. That is met on the A100, except 32 bins at DM 300 (2.95x).
  - High DM costs more because one 32-byte DRAM sector feeds several bins. Two fixes were tried
    and removed: running the XMAC blocks in time order, and having the station kernel write the
    spectra in bin order.
- **Phase centres** ([phase-centres](phase-centres/FINDINGS.md)): the shift to each centre is a
  tensor-core matrix product on the XMAC's fine visibilities, once per piece of the subint.
  - Its cost over the XMAC is about 0.3-0.45 x centres / windows per piece.
  - Short pieces make the XMAC itself dearer: 1.4x at 50 windows a piece, 2.3x at 25, 5x at 10
    (A100).
- **Two-stage channelisation** ([two-stage](two-stage/FINDINGS.md)) is needed only beyond one
  FFT's 8192 channels a subband, or at high delay rates.
  - It costs 1.5-1.6x the one-stage station kernel, at under 0.1% continuum loss and −39 dB alias
    images (32 coarse channels, 4/3 oversampled, 8 taps).
  - Its cost is the same at 8k as at 64k channels. Its RTX PRO 6000 run is still to do.

## Continuum for VGOS-like and EHT-like arrays (RTX PRO 6000, 2-bit)

Interpolated in station count from the measured 4- and 10-station runs, and scaled linearly in
data rate:
- The station stage costs in proportion to the number of stations, the XMAC to the number of
  baselines.
- "Long FFTs" adds the station stage's measured +56% at 8192-point FFTs (A100). That applies if
  EHT's 2 GHz bands are channelised directly at 0.5 MHz.
- The input column is the data every GPU together must take in over PCIe. One PCIe 5 x16 link
  carries about 40-50 GB/s in practice.

| array | stations x rate | gpudifx today: load / GPUs | new chain: load / GPUs | speed-up | input to the GPUs |
|---|---|---|---|---|---|
| **VGOS** (planned maximum) | 4 x 32 Gbps | 2.3 / **4** | 0.28 (long FFTs 0.43) / **1** | 8.1x | 16 GB/s |
| EHT today | 7 x 64 Gbps | 8.8 / 13 | 1.0-1.5 / 2-3 | 8.5x | 56 GB/s |
| **EHT** (ngEHT, one band, 2 x 8 GHz) | 7 x 128 Gbps | 17.5 / **26** | 2.1-3.1 / **3-5** | 8.5x | 112 GB/s |
| ngEHT, 230 + 345 GHz together | 7 x 256 Gbps | 35 / 51 | 4.1-6.2 / 6-9 | 8.5x | 224 GB/s |

- **VGOS at its planned 32 Gbps fits one RTX PRO 6000 with room to spare**, where gpudifx today
  needs four.
- **For EHT, the GPU count is set about equally by compute and by PCIe.**
  - 112 GB/s of station data needs at least 3 GPUs just to take it in. At 3-5 GPUs, input will
    matter as much as the kernels.
  - The network into the correlator nodes, about 0.9 Tbps for ngEHT, is a separate limit,
    outside this work.
- gpudifx today's counts use 256-point FFTs. Longer FFTs would raise them too.

## Caveats

- **Kernel time only.** Not included:
  - the host-to-device copies: 10 x 4 Gbps is 5 GB/s, about 0.1 of a PCIe 5 link, which is as
    long as the new chain's own kernels, so copies must overlap the compute;
  - mpifxcorr's CPU side and the MPI data distribution;
  - autocorrelations (not computed in these runs);
  - output, and the features still to add: phase cal, invalid data, bands straddling an edge,
    baselines left out.
- **Extrapolations.** The effects on the RTX PRO 6000 are estimates from A100 ratios and its
  measured S/X, not measurements. The two-stage and binning runs are still to do on it.
- **The 2070 is power-capped and clock-locked.** It shows the trend; its absolute times are not
  comparable with other eras.
- **Known limits:**
  - The lean decode's window may span at most 32 VDIF frames (2- and 4-bit, 128 bands, 4096
    channels).
  - FFT lengths that are not powers of two (EHT's N x 125) are not yet supported.
  - The staged XMAC trails above about 64-100 stations.
- **Port order this suggests:**
  1. the fused station kernel and the staged XMAC: the 8-9x, and every later feature depends on
     them;
  2. the pulsar gate and scrunched bins: cheap, a single pass;
  3. separate bins and phase centres;
  4. two-stage, where the channel counts require it.
