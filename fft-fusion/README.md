# fft-fusion: the station stage, from the packed data to spectra in one kernel

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
- **from the packed data** (`--input packed`): the same, but starting from 1, 2, 4 or 8-bit
  multi-channel VDIF, so that the decode, sideband handling and fringe rotation come first. The
  fused kernel then goes all the way from the packed bytes to fringe-rotated, channelised,
  fractional-sample-corrected spectra. The unfused pipeline with `--rotator double` is gpudifx
  today: its decode + fringe kernel (copied verbatim into `../common/gpudecode_copy.cuh`), cuFFT and
  mpifxcorr's rotation, all with FP64 phases. Also timed: **two kernels**, decode + fringe into

The cross-multiply that follows is [`../xmac`](../xmac/README.md); channel counts beyond one
cuFFTDx FFT are [`../two-stage`](../two-stage/README.md).  The kernels themselves are in
[`../common`](../README.md#the-shared-code-common), shared with them.

- [FINDINGS.md](FINDINGS.md): what has been learned, with links to the measurements in `docs/`.
- [BACKLOG.md](BACKLOG.md): what is in progress and still to do.

Directions this follows (gpudifx, 2026-11-16):

Directions this follows (gpudifx, 2026-11-16):
- **Autocorrelations will move to the XMAC.** They are off by default (`--autocorr none`).
  `--autocorr band|full` restores mpifxcorr's in-rotation accumulation for comparison.
- **No per-element FP64 phases.** The fused kernel always uses the fixed-point rotator: each
  spectrum's phase slope and offset are computed once in double, and each channel's phase in
  64-bit fixed point.
- **FP16 out of decode and fringe rotation** (`--input fp16`). The FFT takes FP16 in and gives
  FP32 out (see "FP16 in, FP32 out" below).
- **A windowed overlap-add front end** (`--wola`): a window with 12.5% / 75% / 12.5%
  taper / flat / taper, each window overlapping each neighbour over its taper (25% of each
  window is overlapped, 12.5% at each end; consecutive windows start 87.5% of an FFT length
  apart), with Vorbis tapers, which lose no sensitivity. Tukey and sine tapers are options
  too. See "Window design" ([windows.md](docs/windows.md)).
- **Decode and fringe rotation in the same kernel, if it pays** (`--input packed`), against
  keeping the trip through global memory between fringe rotation and the FFT (two kernels).
  See "From the packed data" ([station-kernel.md](docs/station-kernel.md)).


## Run

Build in the repository root (`make`, see [../README.md](../README.md#building)); then here:

```bash
./fusebench --help
./fusebench --nchan 128 --nwin 10000 --input fp16 --wola --fpb sweep  # one case
./fusebench --nchan 128 --nwin 10000 --input packed --wola --ept sweep # from the packed data
./fusebench --input packed --xmac --nchan 1024 --nant 6 --nwin 1000    # the whole chain, with the XMAC
./run-bench.sh                                                       # the standard set (about 30 s)
./run-nchan.sh                                                       # cost against channel count
./run-wide.sh                                                        # the whole chain: bits x subbands, 4-64 stations
./run-matrix.sh                                                      # the whole chain over 108 shapes (corner cases)
python3 window_response.py 256                                       # window design numbers
python3 plot_windows.py                                              # the plots in plots/
```

`run-bench.sh` writes its logs to `results-local/<host>-<stamp>/` (git-ignored).  `results/` (in
the repository) holds summaries only, one CSV row per shape: `run-matrix.sh` writes
`results/<host>-matrix-<stamp>.csv` (columns: `matrix_summary.py`), from `fusebench --csv`, which
appends every timing and check of a run as CSV rows.

The main options:
The main options:
- shape: `--nant --nband --nchan`, and `--nwin`, the subint length in FFT lengths;
- `--complex` for complex sampling (real data use an FFT of 2 × nchan and keep the lower half);
- `--input fp32|fp16`;
- `--layout window|series`: one buffer per FFT window (mpifxcorr's layout), or one continuous
  series per antenna and band, which overlapping windows need;
- the window: `--taper F` (taper length at each end, as a fraction of the FFT length),
  `--overlap F` (the fraction of an FFT length consecutive windows share; the hop is 1 - F),
  `--window tukey|sine|vorbis` (taper shape), or `--wola` (`--taper 0.125 --overlap 0.125
  --window vorbis`; a `--window` given as well takes precedence);
- `--rotator`, `--autocorr`, `--fpb N|sweep` (FFTs per block of the fused kernel, default
  cuFFTDx's suggestion), `--batched` (one launch for all antennas), `--invalid F` (flag a
  fraction of the windows);
- `--xmac` (packed 1-, 2- or 4-bit input, lean decode, dual polarisation): the XMAC after the
  station processing: gpudifx's XMAC from the FP32 spectra, and the end-to-end kernel writing
  FP16 spectra for the staged tensor XMAC;
- `--csv FILE`: every timing and check also as CSV rows (`run-matrix.sh`, `results/`).

### Checks

Each run checks:
Each run checks:
- that the fused and unfused spectra agree (relative rms < 1e-5). They are bit-identical when
  both use the same rotator;
- that both agree with a double-precision DFT and rotation of the *exact* input on 16 random
  spectra, and that the fused result is at least as close as the unfused one. "Exact" means the
  generated noise before it is rounded to the storage precision, times the window. FP32 input
  gives 2-3e-7 rms. FP16 input gives 2e-4 rms, which is FP16's own rounding (about -74 dB),
  far below the quantisation noise of 2- or 4-bit data;
- that the autocorrelations agree (relative rms < 1e-4), when they are on;
- with `--xmac`, that the staged XMAC agrees with gpudifx's XMAC run on the same FP16 spectra
  (relative rms < 1e-4: only the order of the FP32 sums differs) and on the FP32 spectra (< 1e-3:
  FP16's rounding of the spectra).

## Code

| file | what |
|---|---|
| `fusebench.cu` | data, window, checks, timing, options |
| `run-*.sh` | the standard set, the channel sweep, the wider shapes, the big matrix |
| `matrix_summary.py` | the big matrix's summary CSV (`results/`) |
| `window_response.py`, `plot_windows.py` | sensitivity and spectral response of window designs, and their plots |

The kernels are in `../common/` (`stages.cuh`, `decode.cuh`, `fused_inst.cu`, `unfused.cu`, ...:
the table in [../README.md](../README.md#the-shared-code-common)).

Both pipelines are built from the same stage objects. A pre-FFT stage is a struct with
`operator()(geometry, ant, window, band, sample)` that returns the FP32 FFT input, plus
`sourceIndex()` and `weight()`, which the check uses to rebuild the exact input. For the kernels
whose threads each work on one spectrum (the fused kernel, and the 2-kernel pipeline's
`decode_to`) it also has `prepare(geometry, ant, window, band)`, called once per thread for
everything that depends only on the window and band, and `at(state, ..., sample)` per sample.
The unfused pipeline runs it as its own kernel into an FP32 staging buffer, then runs cuFFT on
that buffer. For packed input the check decodes the bytes on the host with its own decoder,
rotates in double precision and does a DFT, and requires both pipelines to be within 1e-5 rms of
that (they are within a few 1e-7). `--bad-frames F` flags a fraction of the VDIF frames invalid,
to test the decode's handling of them. (A tiled decode, the block decoding into shared memory
with band on the lane as gpudifx's kernel does, was measured ([docs/station-kernel.md](docs/station-kernel.md), "From the packed data") and removed: it never beat
the direct decode, and it cost build time.)

A note for that step: the series layout assumes the stage before the FFT produces one
continuous series per band. Each window then starts at its own sample (`window * hop`, and in
mpifxcorr plus its integer delay shift). Fringe rotation must then be a continuous function of
time, not a phase line fitted per window, because a sample in an overlap belongs to two
windows.

## FP16 in, FP32 out

cuFFTDx has no mixed-precision FFT. Its `Precision<>` is one type (`__half`, `float` or
`double`) for the whole transform, and `Precision<__half>` computes in FP16 and returns FP16.
It does not need one, though. With block execution the FFT works on values in registers, and
the loads from and stores to global memory are the caller's own code. The fused kernel loads
FP16, converts to FP32 in registers, does an FP32 FFT, and stores FP32. The memory saving is
all in the load, and the FFT itself is not the bottleneck. An FP16 FFT would compute at FP16
precision (unit roundoff 5e-4, with rounding error growing with the number of FFT stages) and
return FP16, for no speed gain in a kernel that is memory-bound.
