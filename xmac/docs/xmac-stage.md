# The XMAC stage on its own

_Moved from `fft-fusion/README.md` in the 2026-11-21 reorganisation, unchanged but for the heading levels. `results-local/` paths are in `fft-fusion/` (runs of `run-xmac.sh` and `run-staged.sh`: `xmac/`). Section names in quotes refer to these docs: see [the findings index](../FINDINGS.md)._

## The XMAC stage: RTX 2070 SUPER, 215 W (before 2026-11-18)

`xmacbench` cross-multiplies station spectra in the layout the fused kernel writes
(`[ant][win][band][fftchannels]`, FP32, bands 2f and 2f+1 the two polarisations of frequency f)
for every baseline and all four polarisation products, summed over the subint, and checks a
sample of channels against a double-precision sum.  The spectra are a common signal (correlation
0.1 between stations) plus each station's own noise.  Three versions:

- **gpudifx**: `gpu_fuse_xmac_and_average`, verbatim, launched as GPUCore launches it: a thread per
  (baseline, product, channel), each reading both stations' FP32 values for every window from
  global memory, so each station's spectra are read twice per baseline it is in.
- **tiled**: FP32 on the CUDA cores, a thread per (channel, 4 x 4 block of receiver pairs), each
  loaded value used 4 times from registers.
- **tensor**: FP16 in, FP32 sums, WMMA 16 x 16 x 16: per channel, the receivers (station,
  polarisation) as rows and 16 windows as columns, V += X X^H as four real products.  It needs its
  own layout (per channel, blocks of 16 windows, receivers on the rows, real and imaginary planes);
  here a transpose kernel makes it from the fused kernel's output, timed separately.  The rows are
  not padded to 16: a tile that runs past the last receiver reads the next plane's rows, whose
  products are discarded (padding 20 receivers to 32 made the XMAC 64% slower).

The new versions take invalid windows to be zero, which the station processing would write (it
knows which windows are invalid), rather than reading flags per sample; gpudifx's kernel skips
them, which gives the same sums.  All pass the check: FP32 versions within 1-4e-7 rms of the
double-precision sums, the tensor version within 4-7e-5 (FP16 rounding of the input).

`run-xmac.sh` (`results-local/ar313-adeller-xmac-20261117-134059`): 4 frequencies x 2
polarisations, 128 channels; ms per ms of data (below 1: faster than real time; 10000 windows,
4000 from 24 stations up, to fit the 2070's 8 GB).  For context, the station processing that
comes before it (2-bit data, no window, 128 channels; `run-bench.sh`,
`results-local/ar313-adeller-20261117-123426`, measured at 10 stations and scaled by the
number of stations): gpudifx today (its decode + fringe kernel and fractional-sample
correction, both with FP64 phases, and cuFFT: 31.0 ms per 20 ms at 10 stations; the 2070's FP64
runs at 1/32 of its FP32 rate), the end-to-end kernel (4.06 ms) and the 2 kernels with an FP16
buffer (7.34 ms); with WOLA the last two cost 15% more.  Scaling them by the number of stations
is fair: each station is processed on its own (its own launches, no data shared between
stations, each launch enough to fill the GPU), and measured with 4 to 32 stations (fusebench
`--nant`, 10000 windows up to 12 stations, 2500 to 32) the cost per station is the same within
3% for all three; only 2 stations cost 13-20% more per station (the fixed work of a subint), and
quarter-length subints 1-4% more per ms of data.

| stations | station: gpudifx | station: end-to-end kernel | station: 2 kernels, FP16 | XMAC: gpudifx | XMAC: tiled | XMAC: tensor | transpose | XMAC: tensor + transpose |
|---|---|---|---|---|---|---|---|---|
| 4 | 0.621 | 0.081 | 0.147 | 0.118 | 0.040 | 0.021 | 0.081 | 0.103 |
| 6 | 0.931 | 0.122 | 0.220 | 0.265 | 0.064 | 0.031 | 0.123 | 0.154 |
| 10 | 1.55 | 0.203 | 0.367 | 0.700 | 0.159 | 0.055 | 0.196 | 0.250 |
| 16 | 2.48 | 0.325 | 0.587 | 2.78 | 0.610 | 0.105 | 0.280 | 0.380 |
| 24 | 3.72 | 0.487 | 0.880 | 6.20 | 0.970 | 0.186 | 0.420 | 0.610 |
| 32 | 4.97 | 0.649 | 1.17 | 12.0 | 2.87 | 0.410 | 0.550 | 0.960 |
| 48 | 7.45 | 0.974 | 1.76 | 30.8 | 7.30 | 0.790 | 0.830 | 1.62 |
| 64 | 9.93 | 1.30 | 2.35 | 57.8 | 14.7 | 1.14 | 1.11 | 2.25 |

(1024 channels, 1000 windows: 10 stations 0.65 / 0.194 / 0.057; 32 stations 11.7 / 3.18 / 0.21.)

- **gpudifx's XMAC is the most expensive stage once the station processing is fast.**  At 10
  stations it costs 0.70 ms per ms of data, 3.5 times the end-to-end kernel (0.20), though less
  than gpudifx's own station processing on this card (1.55, FP64-bound); from 16 stations up it
  costs more than that too.  It runs at 40-65 G complex multiply-accumulates per second whatever
  the number of stations, so its cost grows as the number of baselines, and it stops keeping up
  with real time at about 12 stations.
- **Register tiling alone gains 3-6x** (150-300 Gcmac/s): 0.16 at 10 stations.
- **The tensor cores gain another 2-13x**, more the more stations (0.055 at 10 stations, real
  time up to about 55 stations on the 2070), but only if the station kernel writes their layout:
  the transpose costs more than the tensor XMAC itself (it reads the FP32 spectra and writes the
  FP16 copy).  Written directly by the fused kernel, the layout would also halve the fused
  kernel's output (FP16).
  The fused kernel writing it, and the two stages together: see "The whole chain" ([whole-chain.md](whole-chain.md)).

## The XMAC stage: A100-SXM4-80GB (OzSTAR, gina9)

`fusebench-ozstar.sbatch`, SLURM job 21055513, commit 8e7cb4b, all runs passing (`run-bench.sh` 38,
`run-xmac.sh` 10, and the channel sweep); 10000 windows at every station count.  As for the 2070,
ms per ms of data, 128 channels, with the station processing for context (2-bit, no window, from
`run-bench.sh` in the same job at 10 stations, scaled by the number of stations: gpudifx 6.18 ms
per 20 ms, the end-to-end kernel 1.50, the 2 kernels with an FP16 buffer 2.87; the scaling was
measured on the 2070):

| stations | station: gpudifx | station: end-to-end kernel | station: 2 kernels, FP16 | XMAC: gpudifx | XMAC: tiled | XMAC: tensor | transpose | XMAC: tensor + transpose |
|---|---|---|---|---|---|---|---|---|
| 4 | 0.124 | 0.0301 | 0.0573 | 0.0427 | 0.0152 | 0.0065 | 0.0171 | 0.0234 |
| 6 | 0.185 | 0.0451 | 0.0860 | 0.100 | 0.0225 | 0.0087 | 0.0265 | 0.0353 |
| 10 | 0.309 | 0.0751 | 0.143 | 0.281 | 0.0506 | 0.0172 | 0.0432 | 0.0600 |
| 16 | 0.495 | 0.120 | 0.229 | 0.708 | 0.121 | 0.0231 | 0.0656 | 0.0885 |
| 24 | 0.742 | 0.180 | 0.344 | 1.52 | 0.263 | 0.0391 | 0.0983 | 0.136 |
| 32 | 0.989 | 0.240 | 0.459 | 3.06 | 0.439 | 0.0579 | 0.131 | 0.187 |
| 48 | 1.48 | 0.361 | 0.688 | 6.69 | 1.24 | 0.108 | 0.196 | 0.303 |
| 64 | 1.98 | 0.481 | 0.917 | 12.5 | 2.53 | 0.183 | 0.260 | 0.441 |

(1024 channels, 1000 windows: 10 stations 0.269 / 0.052 / 0.019; 32 stations 2.50 / 0.573 / 0.064.)

- **gpudifx's XMAC** runs at 140-200 G complex multiply-accumulates per second (about 3 times the
  2070): at 10 stations it costs nearly as much as gpudifx's own station processing and 3.7
  times the end-to-end kernel, from 12 stations up more than both, and it keeps up with real time
  to about 19 stations.
- **The tiled FP32 kernel** runs at 0.4-1.2 T per second (about half the A100's FP32 peak): it
  costs as much as the end-to-end station kernel at 16 stations and keeps up to about 45.
- **The tensor kernel** runs at up to 11 T per second at 64 stations (about 30% of the A100's
  FP16 tensor-core peak), 2-14 times the tiled kernel, more the more stations: 0.017 at 10
  stations, a quarter of the end-to-end station kernel, and 0.18 at 64, still well under the
  station processing (0.48).
  With the tensor XMAC, the station processing is the larger cost at every station count here.
- **The transpose costs 1.4-3.0 times the tensor XMAC**; even with it the tensor version beats
  the tiled one from about 12 stations up.  Writing the tensor layout from the station kernel
  remains the step that decides it.
