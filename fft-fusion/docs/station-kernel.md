# The station kernel: fused against unfused, and from the packed data

_Moved from `fft-fusion/README.md` in the 2026-11-21 reorganisation, unchanged but for the heading levels. `results-local/` paths are in `fft-fusion/` (runs of `run-xmac.sh` and `run-staged.sh`: `xmac/`). Section names in quotes refer to these docs: see [the findings index](../FINDINGS.md)._

`run-bench.sh`: 10 antennas, 8 bands, real data, 64 MHz bands, no autocorrelations. Median of
20 subints, ms per subint for all 10 antennas. Fused times are for the best FFTs-per-block
setting, with cuFFTDx's suggested setting in brackets where it differs. "vs mpifxcorr" is
relative to the first row: FP32 input, cuFFT, then mpifxcorr's rotation kernel. WOLA is 12.5%
tapers with hop 0.875 FFT lengths; "WOLA, hop 0.75" is the same window with consecutive windows
sharing 25% (`--overlap 0.25`), for comparison. The 2070 was timed with Tukey tapers, `--wola`'s
shape at the time, and the A100 with Vorbis tapers. The taper shape does not change the cost
(the window is a table of weights; Tukey, sine and Vorbis time the same).

## RTX 2070 SUPER at the 150 W cap (2026-11-18)

All three scripts at 150 W (`system.txt`), commit 5fa7179 plus the uncommitted staged tensor XMAC
and FP16-spectra output: `run-bench.sh` (`results-local/ar313-adeller-20261118-134417`, 38 runs),
`run-nchan.sh` (`results-local/ar313-adeller-nchan-20261118-134700`, 18 runs) and `run-xmac.sh`
(`results-local/ar313-adeller-xmac-20261118-134902`), every check passing.  The 48- and
64-station XMAC rows were rerun after `xmacbench` was changed to skip the staged XMAC where it
does not fit (it had exited before timing anything; `NOTE.txt` there).

**What the cap costs**, against the last 215 W runs of the same scripts
(`ar313-adeller-20261117-123426`, `ar313-adeller-nchan-20261117-123703`,
`ar313-adeller-xmac-20261117-171357`; the card varies by about 5% between runs):
- **The station kernels that compute hardest slow down most.** The end-to-end kernel from packed
  data is 9-14% slower at both shapes (128 channels: 4.06 -> 4.54 ms; 1024: 3.97 -> 4.43; with
  WOLA 4.69 -> 5.36 and 4.60 -> 5.18), and the fused kernel from FP16 input 1-5% at 128 channels
  but 14-17% at 1024.  The 2-kernel pipeline with an FP16 buffer is 1-10% slower.
- **Memory-bound kernels barely move**: the fused kernel from FP32 input 0-5%, cuFFT and the
  unfused pipelines (gpudifx's included) 1-4%, the 2-kernel FP32 pipeline 1-4%.
- **Channel sweep**: the end-to-end kernel is 3-12% slower (most at 256-2048 channels), and still
  0.24-0.49 ms per ms of data.
- **XMAC alone**: the tensor XMAC is 18% slower at 1024 channels and 32 stations (the load tests
  found +21%), but within 2% at 128 channels at every station count.  The transpose is 4-8%
  slower.  gpudifx's XMAC and the tiled one are within run-to-run noise (up to ±15% at 4-10
  stations, within 5% from 16 up).
- **The whole chain**: the end-to-end paths are 10-16% slower up to 16 stations (3-10% at 32,
  where the XMAC dominates), gpudifx today 3-7%, so the gain narrows a little.  At 128 channels and 10 stations the end-to-end kernel writing the layout,
  plus the tensor XMAC, is 6.9x gpudifx today (7.5x at 215 W).  Which chain is fastest at each
  shape does not change among the versions measured at 215 W.

XMAC (`table.txt`), ms per ms of data, 128 channels (10000 windows, 4000 from 24 stations up),
with the station processing for context, scaled by the number of stations as in the 215 W table
("The XMAC stage" ([xmac-stage.md](../../xmac/docs/xmac-stage.md))): `run-bench.sh` `c128-packed` at 10 stations, gpudifx 31.4 ms, the
end-to-end kernel 4.54, the 2 kernels with an FP16 buffer 7.38 per 20 ms.  "Staged" is the tensor
XMAC from FP16 spectra in the station kernel's own order, which it reshuffles itself, in its first
version: superseded by the version split by receiver below.

| stations | station: gpudifx | station: end-to-end kernel | station: 2 kernels, FP16 | XMAC: gpudifx | XMAC: tiled | XMAC: tensor | transpose | XMAC: tensor + transpose | XMAC: staged, first version |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.628 | 0.091 | 0.148 | 0.103 | 0.040 | 0.021 | 0.075 | 0.100 | 0.041 |
| 6 | 0.943 | 0.136 | 0.221 | 0.278 | 0.066 | 0.031 | 0.125 | 0.162 | 0.048 |
| 10 | 1.57 | 0.227 | 0.369 | 0.799 | 0.179 | 0.056 | 0.206 | 0.273 | 0.088 |
| 16 | 2.51 | 0.363 | 0.590 | 2.82 | 0.633 | 0.105 | 0.297 | 0.407 | 0.126 |
| 24 | 3.77 | 0.545 | 0.885 | 6.49 | 0.869 | 0.185 | 0.447 | 0.661 | 0.367 |
| 32 | 5.03 | 0.726 | 1.18 | 12.3 | 2.89 | 0.406 | 0.593 | 1.02 | 0.658 |
| 48 | 7.54 | 1.09 | 1.77 | 31.0 | 7.29 | 0.800 | 0.860 | 1.69 | does not fit |
| 64 | 10.1 | 1.45 | 2.36 | 58.0 | 14.7 | 1.10 | 1.15 | 2.34 | does not fit |

(1024 channels, 1000 windows, gpudifx / tiled / tensor / staged: 10 stations 0.744 / 0.200 /
0.058 / 0.129; 32 stations 11.8 / 3.20 / 0.257 / 1.31.)

Whole chain (`chain.txt`), ms per ms of data, the fastest variant of each kind; 10000 windows
(2500 at 16 and 32 stations), 1000 at 1024 channels:

| channels | stations | gpudifx today | end-to-end + gpudifx XMAC | end-to-end + tiled | end-to-end + transpose + tensor | end-to-end writing the layout + tensor | end-to-end writing FP16 spectra + staged, first version |
|---|---|---|---|---|---|---|---|
| 128 | 4 | 0.753 | 0.211 | 0.136 | 0.196 | **0.134** | 0.136 |
| 128 | 10 | 2.41 | 1.05 | 0.423 | 0.510 | 0.349 | **0.332** |
| 128 | 16 | 4.56 | 2.42 | 0.769 | 0.815 | 0.569 | **0.558** |
| 128 | 32 | 16.5 | 11.9 | 3.28 | 1.73 | **1.29** | 1.52 |
| 1024 | 10 | 2.29 | 1.05 | 0.486 | 0.556 | 0.512 | **0.409** |

**The staged tensor XMAC on the 2070, first version** (the 2026-11-17 run that would have
measured it was cut off by the reset).  Its stage held 8 channels of *every* receiver, about 770
bytes per receiver: it did not fit in sm_75's 64 KB a block above 40 stations, and alone it cost
1.2-2.0x the tensor XMAC from its layout at 128 channels and 5x at 1024 channels and 32 stations.
Even so it already made the fastest chain at 10 stations (128 and 1024 channels), because writing
FP16 spectra costs the end-to-end kernel no more than FP32 ones (10 stations: 4.70 against 4.81 ms
at 128 channels, 4.45 against 4.67 at 1024), while writing the tensor layout costs 5.68 and 7.10
ms.  Nsight Compute (`results-local/ncu-ar313-adeller-20261118-160023-staged`) showed why it
fell behind from 24 stations up: above 32 receivers its items no longer fitted one block, and
the 2-3 blocks of a channel group each staged every receiver again, two of them only half of
each 32-byte sector, so it read 3.1x the FP16 input from DRAM at 32 stations (1.15x at 16); and
at 37 KB and more only one block fitted on an SM.  At 1024 channels the same bytes moved at half
the bandwidth (36% of peak, 30 cycles a load waiting against 10).

## RTX 2070 SUPER, 215 W (before 2026-11-18)

CUDA 13.2, cuFFT 12.2, cuFFTDx 1.5.0. 448 GB/s memory bandwidth, FP64 at 1/32 rate.

**128 channels** (256-point FFT), 20 ms subint (10000 FFT lengths):

| FFT input | windows | unfused: prep + cuFFT + rotation | unfused | fused | fused vs unfused | fused vs mpifxcorr |
|---|---|---|---|---|---|---|
| FP32, mpifxcorr's FP64 rotation | 10000 | 0 + 8.46 + 9.03 | 17.45 | 6.38 | 2.74 | **2.7** |
| FP32 | 10000 | 0 + 8.46 + 4.41 | 12.87 | 6.38 | 2.02 | **2.7** |
| FP16 | 10000 | 6.50 + 8.46 + 4.41 | 19.37 | 4.56 (4.70) | 4.24 | **3.8** |
| FP32, WOLA | 11428 | 9.09 + 9.67 + 5.03 | 23.79 | 7.33 (7.58) | 3.25 | **2.4** |
| FP16, WOLA | 11428 | 7.40 + 9.66 + 5.03 | 22.09 | 5.77 (6.12) | 3.83 | **3.0** |
| FP16, WOLA, hop 0.75 | 13333 | 8.30 + 11.27 + 5.87 | 25.43 | (6.94) | 3.67 | **2.5** |

**1024 channels** (2048-point FFT), 16 ms subint (1000 FFT lengths):

| FFT input | windows | unfused: prep + cuFFT + rotation | unfused | fused | fused vs unfused | fused vs mpifxcorr |
|---|---|---|---|---|---|---|
| FP32, mpifxcorr's FP64 rotation | 1000 | 0 + 6.78 + 7.22 | 14.00 | 5.01 | 2.79 | **2.8** |
| FP32 | 1000 | 0 + 6.78 + 3.41 | 10.19 | 5.01 | 2.03 | **2.8** |
| FP16 | 1000 | 5.20 + 6.78 + 3.41 | 15.39 | 3.53 | 4.36 | **4.0** |
| FP32, WOLA | 1142 | 7.26 + 7.74 + 3.89 | 18.89 | 6.23 | 3.03 | **2.2** |
| FP16, WOLA | 1142 | 5.84 + 7.74 + 3.89 | 17.47 | 4.72 (4.79) | 3.70 | **3.0** |
| FP16, WOLA, hop 0.75 | 1333 | 6.56 + 9.03 + 4.54 | 20.12 | (5.26) | 3.82 | **2.7** |

(The hop 0.75 rows were timed at the suggested setting only.)

### What the numbers say

1. **Fused with FP32 input is 2.7-2.8x faster than mpifxcorr's FFT + rotation today.** That
   splits into two parts. Replacing the per-element FP64 phases with fixed point is worth
   1.35x on this card: mpifxcorr's rotation kernel is FP64-bound on the 2070 (84% of the FP64
   pipe, 32% of DRAM, Nsight Compute), because Turing runs FP64 at 1/32 rate. Fusion is worth
   another 2.0x. With fixed-point phases both pipelines are memory-bound, and fusion halves the
   traffic: cuFFT reads and writes every sample, and the rotation reads and writes half of them
   again. The fused kernel reads the input once and writes only the half that is used.
2. **FP16 input makes the fused kernel faster still: 3.8-4.0x vs mpifxcorr.** It reads half the
   bytes. The FP32 output is unchanged. Unfused, FP16 is slower than FP32, because cuFFT needs
   FP32 input and converting it is a full extra pass. A cuFFT load callback could avoid that
   pass; it is not tried here.
3. **WOLA costs 26-34% over the fused kernel without it**: 1.143x more windows, plus the series
   reads and the weights. Fused with FP16 input it is still 3.0x faster than mpifxcorr's FFT +
   rotation without any windowing. Unfused, it would cost 1.25-1.4x mpifxcorr's time.
   Overlapping by 25% of an FFT length between consecutive windows (hop 0.75) costs 10-13%
   more than overlapping just the tapers.
4. **FFTs per block matter.** One per block is bad for small FFTs (256 points). Above 16 KB of
   shared memory per block, the larger settings lose. cuFFTDx's suggestion is within about 6% of
   the best everywhere here.

## A100-SXM4-80GB (OzSTAR, gina7)

`fusebench-ozstar.sbatch`, SLURM job 21008444. CUDA 12.8, cuFFTDx 1.5.0. 2039 GB/s memory
bandwidth (4.6x the 2070), FP64 at 1/2 rate. Every check passed.

**128 channels** (256-point FFT), 20 ms subint (10000 FFT lengths):

| FFT input | windows | unfused: prep + cuFFT + rotation | unfused | fused | fused vs unfused | fused vs mpifxcorr |
|---|---|---|---|---|---|---|
| FP32, mpifxcorr's FP64 rotation | 10000 | 0 + 1.97 + 1.16 | 3.13 | 1.64 | 1.91 | **1.9** |
| FP32 | 10000 | 0 + 1.96 + 1.13 | 3.09 | 1.64 | 1.89 | **1.9** |
| FP16 | 10000 | 1.59 + 1.96 + 1.13 | 4.69 | 1.40 | 3.36 | **2.2** |
| FP32, WOLA | 11428 | 2.27 + 2.23 + 1.28 | 5.77 | 2.23 | 2.59 | **1.4** |
| FP16, WOLA | 11428 | 1.97 + 2.23 + 1.28 | 5.48 | 1.85 | 2.96 | **1.7** |
| FP16, WOLA, hop 0.75 | 13333 | 2.24 + 2.58 + 1.49 | 6.31 | (2.08) | 3.03 | **1.5** |

**1024 channels** (2048-point FFT), 16 ms subint (1000 FFT lengths):

| FFT input | windows | unfused: prep + cuFFT + rotation | unfused | fused | fused vs unfused | fused vs mpifxcorr |
|---|---|---|---|---|---|---|
| FP32, mpifxcorr's FP64 rotation | 1000 | 0 + 1.62 + 1.06 | 2.68 | 1.35 | 1.98 | **2.0** |
| FP32 | 1000 | 0 + 1.62 + 0.90 | 2.52 | 1.35 | 1.86 | **2.0** |
| FP16 | 1000 | 1.28 + 1.62 + 0.90 | 3.80 | 1.26 | 3.03 | **2.1** |
| FP32, WOLA | 1142 | 1.81 + 1.83 + 1.02 | 4.65 | 1.83 | 2.54 | **1.5** |
| FP16, WOLA | 1142 | 1.56 + 1.83 + 1.02 | 4.40 | 1.55 | 2.84 | **1.7** |
| FP16, WOLA, hop 0.75 | 1333 | 1.78 + 2.12 + 1.18 | 5.07 | (1.76) | 2.88 | **1.5** |

### What changes on the A100

1. **Fused is 1.9-2.0x faster than mpifxcorr's FFT + rotation with FP32 input, 2.1-2.2x with
   FP16** (2070: 2.7-2.8x and 3.8-4.0x). The A100 runs mpifxcorr's FFT + rotation 5.2-5.6x
   faster than the 2070, but the fused kernel only 2.8-3.9x faster.
2. **Fixed-point phases gain little here; the gain is fusion.** With FP64 at half rate,
   mpifxcorr's rotation kernel takes about as long as the fixed-point one: 1.16 vs 1.13 ms at
   128 channels (2%), 1.06 vs 0.90 ms at 1024 (17%). Fixed-point phases still matter on cards
   with low-rate FP64, such as the 2070.
3. **With FP32 input the fused kernel is limited by memory bandwidth; with FP16 input, on the
   A100, it is not.** It must read each window's `fftchannels` complex input samples and write
   `nchan` FP32 complex channels: 2.46 GB per subint at 128 channels with FP32 input, 1.64 GB
   with FP16. With FP32 input that is 1.45-1.50 TB/s, 71-74% of the A100's 2039 GB/s, as fast
   as the fixed-point rotation kernel, a plain streaming kernel (1.45 TB/s, 71%); on the 2070
   86-88%, against the streaming kernel's 84-86% (Nsight Compute agrees: 89% DRAM throughput).
   With FP16 input it is 1.04-1.17 TB/s, 51-58%: FP16 moves 33% fewer bytes but saves only
   7-15% of the time on the A100 (29-30% on the 2070, 80-83% of its bandwidth). So only the
   FP16 path has room on the A100, up to about 1.3-1.4x. (Corrected: the first version of this
   paragraph, in 501bc92, counted the FP32 input at 4 bytes a sample instead of 8, and
   concluded wrongly that the FP32 kernel reached only 38-49%.)
4. **WOLA costs 23-36% over the fused kernel without it** (2070: 26-34%), and hop 0.75 another
   13-14% (2070: 10-13%). Fused with FP16 input, WOLA is still 1.7x faster than mpifxcorr's
   FFT + rotation without windowing; unfused it would cost 1.6-1.8x mpifxcorr's time.
5. **cuFFTDx's suggested FFTs per block was the best setting in every case.**
6. For this array (10 antennas, 8 x 64 MHz bands), the fused FP16 WOLA kernel runs at
   10.3-10.8x real time, mpifxcorr's FFT + rotation at 6.0-6.4x. That is this stage alone,
   without unpacking, fringe rotation, the cross-multiply or transfers.

## From the packed data: RTX 2070 SUPER, 215 W (before 2026-11-18)

`run-bench.sh`, the `packed-*` cases: 2-bit, 8-band VDIF (8000-byte payloads), odd bands lower
sideband, an integer delay per antenna, a fringe phase slope and offset per window and band.
Best variant of each fused kernel (FFTs per block, elements per thread), ms per subint for all
10 antennas, all checks passing. The FFT stage of the 2-kernel rows is the better of cuFFTDx's
and half its elements per thread where both were timed (the WOLA columns). gpudifx's decode
kernel cannot apply a window, so the WOLA columns have no gpudifx rows.

| pipeline | 128 ch | 128 ch, WOLA | 1024 ch | 1024 ch, WOLA |
|---|---|---|---|---|
| gpudifx today: decode + fringe (FP64), cuFFT, rotation (FP64) | 13.52 + 8.45 + 9.26 = 31.18 | | 9.50 + 6.77 + 7.41 = 23.65 | |
| unfused, fixed-point phases | 21.15 | 24.10 | 16.06 | 18.18 |
| **end-to-end fused kernel** | **6.71** (4.6x) | **7.88** | **6.20** (3.8x) | **7.19** |
| 2 kernels: decode + fringe -> FP16, fused FFT | 3.78 + 4.72 = 8.46 | 4.33 + 5.21 = 9.54 | 2.69 + 3.56 = 6.32 | 3.05 + 4.92 = 7.94 |
| 2 kernels: decode + fringe -> FP32, fused FFT | 4.90 + 6.38 = 11.28 | 5.55 + 7.31 = 12.87 | 4.55 + 5.00 = 9.54 | 5.14 + 5.95 = 11.08 |
| 2 kernels: gpudifx's decode (FP64) -> FP32, fused FFT | 13.52 + 6.38 = 19.98 | | 9.50 + 5.00 = 14.61 | |
| end-to-end, tiled decode | 8.69 | | 6.29 | |

1. **On the 2070 the end-to-end kernel is the fastest: 4.6x gpudifx today at 128 channels,
   3.8x at 1024.** Against two kernels with an FP16 buffer between them it takes 21% less time
   at 128 channels (17% with WOLA), and 2% less at 1024 channels (9% with WOLA). WOLA costs the
   end-to-end kernel 16-17% (1.14x the windows). The A100 has 4.6x the memory bandwidth and
   about 2x the arithmetic, which makes the trip through memory relatively cheaper there: it
   needs measuring (`run-bench.sh`) before choosing. The first kernel of the pair was written
   for this comparison (one block per antenna and window, 8 samples per thread, unrolled, as in
   the fused kernel). At 128 channels it is still far from the write bandwidth (4.33 ms for
   0.94 GB with WOLA), so it may have more in it.
2. **Decode is half the end-to-end kernel**, not the FFT. Taking pieces out of it (128
   channels, no window, cuFFTDx's elements per thread): FFT + correction + store 3.02 ms; with
   fringe rotation 3.89; with the decode 7.58. Per sample the decode finds its frame, checks
   that the frame is valid and reads its bits, and that is long latency for little arithmetic
   (Nsight Compute: 49% of issue slots used, DRAM 22% busy, the main stalls instruction-cache
   misses and load latency).
3. **The tiled decode does not pay here.** Decoding the block's windows into shared memory with
   band on the lane (gpudifx's tiling) is slower at 128 channels (8.80 vs 6.75 ms) and level at
   1024. Its point in gpudifx is coalescing the stores; in the fused kernel the FFT's own layout
   already keeps the stores coalesced, and each window decodes only its own samples (a few
   bytes per sample time, read straight from the payload).
4. **cuFFTDx's suggested elements per thread is not the best.** Half of it (8 instead of 16 for
   both the 256- and 2048-point FFT on this card) takes 7-10% off the end-to-end kernel and 16%
   off the fused FFT of an FP16 buffer (WOLA, 128 channels: 6.15 -> 5.13 ms), against cuFFTDx's
   suggested setting. The variants are registered for both, and `--ept sweep` times them. The
   earlier tables for the FP32 and FP16 buffers were taken with cuFFTDx's setting only.
5. Other changes tried on the end-to-end kernel (128 channels):
   - kept: 32-bit sample numbers in the decode (gpudifx: 64-bit, whose division is a long
     software routine): -4%; 1- and 2-bit levels by arithmetic, not from `__constant__`
     memory: -2%
   - not kept: finding each window's frame once and stepping per sample: +12% (slower);
     one launch for all antennas: no change
   - not done yet: the bit width as a template parameter: -3 to -4% in a quick test, worth it
     in production code

## From the packed data: A100-SXM4-80GB (OzSTAR, gina10)

`fusebench-ozstar.sbatch`, SLURM job 21016518, commit 2247eaa; the same runs as on the 2070, all
checks passing. ms per subint.

| pipeline | 128 ch | 128 ch, WOLA | 1024 ch | 1024 ch, WOLA |
|---|---|---|---|---|
| gpudifx today: decode + fringe (FP64), cuFFT, rotation (FP64) | 3.09 + 1.96 + 1.15 = 6.19 | | 2.17 + 1.62 + 1.08 = 4.84 | |
| unfused, fixed-point phases | 5.99 | 6.82 | 4.77 | 5.42 |
| **end-to-end fused kernel** | **2.85** (2.2x) | **3.24** | 2.48 (2.0x) | **2.83** |
| 2 kernels: decode + fringe -> FP16, fused FFT | 1.65 + 1.39 = 3.02 | 1.88 + 1.73 = 3.60 | **1.18 + 1.25 = 2.43** | 1.34 + 1.63 = 2.97 |
| 2 kernels: decode + fringe -> FP32, fused FFT | 1.66 + 1.63 = 3.31 | 1.89 + 2.11 = 4.01 | 1.17 + 1.35 = 2.52 | 1.33 + 1.80 = 3.13 |
| 2 kernels: gpudifx's decode (FP64) -> FP32, fused FFT | 3.09 + 1.63 = 4.74 | | 2.17 + 1.35 = 3.52 | |
| end-to-end, tiled decode | 3.02 | | 2.48 | |

1. **The end-to-end kernel's lead shrinks on the A100, as expected.** Against two kernels with
   an FP16 buffer it takes 6% less time at 128 channels (10% with WOLA) and 5% less at 1024
   channels with WOLA; without a window at 1024 channels the two kernels are 2% faster. Against
   gpudifx today both are about 2x (2.2x at 128 channels, 2.0x at 1024); on the 2070 4.6x and
   3.8x, where gpudifx's FP64 phases cost far more.
2. **Decode + fringe rotation is not limited by bandwidth on the A100 either.** Writing FP16 or
   FP32 takes the same time (1.65 vs 1.66 ms at 128 channels), so the FP32 buffer costs only
   what its larger read costs the FFT kernel. The FP16 buffer still wins.
3. **Half of cuFFTDx's elements per thread** takes 7-10% off the end-to-end kernel and 8-10%
   off the fused FFT with WOLA (not with FP16 input at 1024 channels, where cuFFTDx's setting is
   best), as on the 2070. The tiled decode does best at 1 FFT per block, and there only matches
   the direct decode.
4. **For this array the end-to-end kernel with WOLA runs at 5.7-6.2x real time**, this stage
   alone.

What Nsight Compute shows (`ncu-ozstar.sbatch`, job 21016519; per antenna; ncu runs the GPU at
its base clock):
- **Occupancy is limited by registers.** The fused kernels use 64-72 registers a thread, which
  leaves 36-42% of the A100's warp slots active (81-98% on the 2070, whose SMs hold half as
  many threads). The main stall is waiting for loads (long scoreboard), with too few warps to
  hide it. Halving the elements per thread cuts registers to 48-56 and raises occupancy to
  51-61%, which is where its gain comes from. Next: cap the registers (`__launch_bounds__`
  with a minimum of blocks per SM) and measure.
- **The end-to-end kernel uses 3-10% of the memory bandwidth** and 49-68% of its issue slots,
  and executes 2.6-3.4x the instructions of the same kernel reading an FP32 buffer: decode and
  fringe rotation are about two-thirds of its instructions. Making it faster means fewer
  instructions per sample in the decode.
- **cuFFTDx's shared-memory exchange has bank conflicts** in every fused variant: 1.0-1.2M
  excess wavefronts per launch at 256 points, 4.1-4.7M at 2048 (the same on the 2070).
- The 128-channel profiles of this run (2000 windows, 33 MB of input per antenna) fit in the
  A100's 40 MB L2 cache, which understates their DRAM traffic; the job now profiles 10000.
