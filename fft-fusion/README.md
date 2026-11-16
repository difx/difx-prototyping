# fft-fusion: cuFFT + fractional-sample correction vs one fused cuFFTDx kernel

A standalone benchmark for the first step of `docs/gpu-window-pipeline-design.md` in difx
(gpudifx branch). That step fuses the FFT and the fractional-sample correction into one kernel
with cuFFTDx, with room to pull the pre-FFT steps in too. It times both ways of doing one
subint's FFTs and rotations, on synthetic data already in GPU memory, and checks that they
agree:

- **unfused**, mpifxcorr's structure: if the FFT input needs any preparation, a kernel writes it
  out as an FP32 buffer. Then one batched cuFFT plan per antenna, then the rotation kernel per
  antenna. `--rotator double` runs mpifxcorr's own `gpu_resultsrotatorMultiply`, copied
  verbatim into `unfused.cu` (the commit it was copied from is noted there). Otherwise the
  unfused pipeline uses the same fixed-point rotation as the fused kernel.
- **fused**: one cuFFTDx kernel per antenna. It loads the FFT input (FP32 or FP16, optionally
  windowed and overlapped) into registers as FP32, does an FP32 FFT in shared memory, rotates
  each channel, and writes only the channels the correlator uses, in FP32.

Directions this follows (gpudifx, 2026-11-16):
- **Autocorrelations will move to the XMAC.** They are off by default (`--autocorr none`).
  `--autocorr band|full` restores mpifxcorr's in-rotation accumulation for comparison.
- **No per-element FP64 phases.** The fused kernel always uses the fixed-point rotator: each
  spectrum's phase slope and offset are computed once in double, and each channel's phase in
  64-bit fixed point.
- **FP16 out of decode and fringe rotation** (`--input fp16`). The FFT takes FP16 in and gives
  FP32 out.
- **A windowed overlap-add front end** (`--wola`): Tukey window with 12.5% / 75% / 12.5%
  taper / flat / taper and 25% overlap. See "Window design" below.

## Build and run

cuFFTDx is part of NVIDIA's header-only MathDx package. It is not in the CUDA toolkit. Get it
from https://developer.nvidia.com/cufftdx-downloads, or unpack the `nvidia-mathdx` wheel from
PyPI (`pip download --no-deps nvidia-mathdx`, then unzip it). `MATHDX` is the directory that
holds `include/cufftdx.hpp` and `external/cutlass` (in the wheel: `nvidia/mathdx`). Tested with
MathDx 25.6.0 (cuFFTDx 1.5.0) and CUDA 13.2. MathDx is not redistributed here (NVIDIA licence).

```bash
make -j8 MATHDX=<mathdx dir> CUDA_HOME=/usr/local/cuda-13.2          # RTX 2070 (sm_75)
make -j8 MATHDX=<mathdx dir> SM=80                                   # A100
./fusebench --help
./fusebench --nchan 128 --nwin 10000 --input fp16 --wola --fpb sweep  # one case
./run-bench.sh                                                       # the standard set below
python3 window_response.py 256                                       # window design numbers
```

The fused kernel is compiled once per FFT length (`SIZES` in the Makefile, default 64 to
16384). That takes about a minute with `-j8`. `fusebench` exits non-zero if a check fails.
`run-bench.sh` takes about 30 s and writes its logs to `results-local/<host>-<stamp>/`
(git-ignored).

The main options:
- shape: `--nant --nband --nchan`, and `--nwin`, the subint length in FFT lengths;
- `--complex` for complex sampling (real data use an FFT of 2 × nchan and keep the lower half);
- `--input fp32|fp16`;
- `--layout window|series`: one buffer per FFT window (mpifxcorr's layout), or one continuous
  series per antenna and band, which overlapping windows need;
- the window: `--taper F --overlap F --window tukey|sine`, or `--wola`;
- `--rotator`, `--autocorr`, `--fpb N|sweep` (FFTs per block of the fused kernel, default
  cuFFTDx's suggestion), `--batched` (one launch for all antennas), `--invalid F` (flag a
  fraction of the windows).

### Checks

Each run checks:
- that the fused and unfused spectra agree (relative rms < 1e-5). They are bit-identical when
  both use the same rotator;
- that both agree with a double-precision DFT and rotation of the *exact* input on 16 random
  spectra, and that the fused result is at least as close as the unfused one. "Exact" means the
  generated noise before it is rounded to the storage precision, times the window. FP32 input
  gives 2-3e-7 rms. FP16 input gives 2e-4 rms, which is FP16's own rounding (about -74 dB),
  far below the quantisation noise of 2- or 4-bit data;
- that the autocorrelations agree (relative rms < 1e-4), when they are on.

## Code layout: adding the pre-FFT steps

| file | what |
|---|---|
| `common.cuh` | `Geometry` (the `[ant][window][band][fftchannels]` layout of mpifxcorr's buffers), error macros |
| `stages.cuh` | the stages either side of the FFT: `PreLoad<T, SERIES>` (FP32/FP16 storage, per-window or series layout, optional window), `withPre()` (the one place stage types are instantiated); `FracRotateFixed` |
| `unfused.cu` | the copied production kernel, cuFFT plans as `GPUMode` makes them, the kernel that materialises a pre-FFT stage, the generic rotation kernel, the fixed-point phase precompute |
| `fused_inst.cu` | the fused kernel, compiled per FFT length; each length registers its FFTs-per-block variants at start-up |
| `fusebench.cu` | data, window, checks, timing, options |
| `window_response.py` | sensitivity and spectral response of window designs |

Both pipelines are built from the same stage objects. A pre-FFT stage is a struct with
`operator()(geometry, ant, window, band, sample)` that returns the FP32 FFT input, plus
`sourceIndex()` and `weight()`, which the check uses to rebuild the exact input. The fused
kernel calls it while loading the FFT input. The unfused pipeline runs it as its own kernel
into an FP32 staging buffer, then runs cuFFT on that buffer. To try decode + fringe rotation in
the kernel, write a stage struct that reads the packed samples, and add it to `withPre()`.

A note for that step: the series layout assumes the stage before the FFT produces one
continuous series per band. Each window then starts at its own sample (`window * hop`, and in
mpifxcorr plus its integer delay shift). Fringe rotation must then be a continuous function of
time, not a phase line fitted per window, because a sample in an overlap belongs to two
windows.

## Results: RTX 2070 SUPER, 2026-11-16

CUDA 13.2, cuFFT 12.2, cuFFTDx 1.5.0. 10 antennas, 8 bands, real data, 64 MHz bands, no
autocorrelations. Median of 20 subints, ms per subint for all 10 antennas. Fused times are for
the best FFTs-per-block setting, with cuFFTDx's suggested setting in brackets where it differs.
"vs mpifxcorr" is relative to the first row: FP32 input, cuFFT, then mpifxcorr's rotation
kernel.

**128 channels** (256-point FFT), 20 ms subint (10000 FFT lengths):

| FFT input | windows | unfused: prep + cuFFT + rotation | unfused | fused | fused vs unfused | fused vs mpifxcorr |
|---|---|---|---|---|---|---|
| FP32, mpifxcorr's FP64 rotation | 10000 | 0 + 8.47 + 8.98 | 17.45 | 6.38 | 2.74 | **2.7** |
| FP32 | 10000 | 0 + 8.46 + 4.41 | 12.87 | 6.38 | 2.02 | **2.7** |
| FP16 | 10000 | 6.50 + 8.46 + 4.41 | 19.36 | 4.57 (4.69) | 4.24 | **3.8** |
| FP32, WOLA 25% | 13333 | 9.98 + 11.28 + 5.87 | 27.12 | 8.38 (8.65) | 3.24 | **2.1** |
| FP16, WOLA 25% | 13333 | 8.30 + 11.28 + 5.87 | 25.45 | 6.38 (6.94) | 3.99 | **2.7** |
| FP16, WOLA 12.5% overlap | 11428 | 7.40 + 9.67 + 5.03 | 22.09 | (6.12) | 3.61 | **2.9** |

**1024 channels** (2048-point FFT), 16 ms subint (1000 FFT lengths):

| FFT input | windows | unfused: prep + cuFFT + rotation | unfused | fused | fused vs unfused | fused vs mpifxcorr |
|---|---|---|---|---|---|---|
| FP32, mpifxcorr's FP64 rotation | 1000 | 0 + 6.78 + 7.17 | 13.95 | 5.01 | 2.78 | **2.8** |
| FP32 | 1000 | 0 + 6.78 + 3.41 | 10.19 | 5.01 | 2.03 | **2.8** |
| FP16 | 1000 | 5.20 + 6.78 + 3.41 | 15.39 | 3.53 | 4.36 | **4.0** |
| FP32, WOLA 25% | 1333 | 7.97 + 9.03 + 4.54 | 21.54 | 7.08 | 3.04 | **2.0** |
| FP16, WOLA 25% | 1333 | 6.56 + 9.03 + 4.54 | 20.12 | 5.08 (5.27) | 3.96 | **2.7** |
| FP16, WOLA 12.5% overlap | 1142 | 5.85 + 7.74 + 3.89 | 17.47 | (4.79) | 3.65 | **2.9** |

(The fused-vs-unfused figures for the 12.5% overlap rows use the suggested setting, the only
one timed.)

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
3. **WOLA with 25% overlap costs 40-45% over the fused kernel without it**: 1.33x more windows,
   plus the series reads and the weights. Fused, with FP16 input, it is still 2.7x faster than
   mpifxcorr's FFT + rotation without any windowing. Unfused, it would cost 1.4-1.6x
   mpifxcorr's time. With 12.5% overlap it is about 10% cheaper than with 25%.
4. **FFTs per block matter.** One per block is bad for small FFTs (256 points). Above 16 KB of
   shared memory per block, the larger settings lose. cuFFTDx's suggestion is within 10% of the
   best everywhere here.

### Window design

`window_response.py` gives these for a 256-point FFT (continuum sensitivity relative to
rectangular windows without overlap; white noise; all channels). The spectral figures do not
depend on the FFT length.

| design | FFTs | sensitivity, delay 0 / 4 / 16 samples | leakage beyond 5 / 20 channels |
|---|---|---|---|
| rectangular, no overlap (mpifxcorr now) | 1.000 | 1.000 / 0.984 / 0.938 | -16.9 / -23.0 dB |
| Tukey 12.5% tapers, 25% overlap (`--wola`) | 1.333 | 0.972 / 0.969 / 0.934 | -25.0 / -60.4 dB |
| Tukey 12.5% tapers, 12.5% overlap | 1.143 | 0.994 / 0.991 / 0.955 | -25.0 / -60.4 dB |
| sine 12.5% tapers, 12.5% overlap | 1.143 | 1.000 / 0.997 / 0.965 | -26.6 / -47.2 dB |
| Tukey 25% tapers, 25% overlap | 1.333 | 0.986 / 0.984 / 0.960 | -44.2 / -71.2 dB |
| sine 25% tapers, 25% overlap | 1.333 | 1.000 / 0.998 / 0.977 | -34.4 / -52.8 dB |
| Hann, 50% overlap | 2.000 | 0.973 / 0.972 / 0.949 | -53.5 / -83.7 dB |

- **The spectral improvement comes from the window shape, and it is large.** Leakage beyond 20
  channels drops from -23 dB to -60 dB with the 12.5% Tukey tapers.
- **Overlapping by more than the taper length costs sensitivity and gains nothing else.**
  Window shape alone sets resolution, leakage and the loss at a residual delay; overlap does
  not change any of them. Overlap sets how evenly the samples are weighted. With 12.5% tapers
  and 25% overlap, the samples in the flat parts of the overlaps are counted twice, so the
  weights run from 1 to 2. That costs 2.9% of continuum sensitivity, and it makes this design
  *less* sensitive than rectangular windows at every residual delay in the table. The same
  window with 12.5% overlap costs 0.6%, with 14% fewer FFTs.
- **Overlap equal to the taper length, with tapers whose squares sum to one across the
  overlap (sine tapers), loses no sensitivity at all.** These are better than rectangular
  windows at every residual delay. The price is slower far-out sidelobe decay than the Tukey
  (raised-cosine) tapers. At 25%, sine tapers give -34 / -53 dB for 1.333x FFTs; Tukey 25%
  tapers give -44 / -71 dB for 1.4% of sensitivity.

All of these are options of `fusebench` (`--taper`, `--overlap`, `--window`), so any of them can
be timed. The cost depends only on the number of windows.

### Not yet measured

- **The A100.** FP64 runs at half rate there and memory bandwidth is 3.5x higher. Build with
  `SM=80` and run `run-bench.sh`.
- Complex data (`--complex`) and other channel counts.
- Decode and fringe rotation inside the fused kernel (the next pre-FFT stage).
- The effect inside mpifxcorr, where the FFT and rotation overlap with transfers and the XMAC.
