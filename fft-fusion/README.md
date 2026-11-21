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
- **from the packed data** (`--input packed`): the same, but starting from 1, 2, 4 or 8-bit
  multi-channel VDIF, so that the decode, sideband handling and fringe rotation come first. The
  fused kernel then goes all the way from the packed bytes to fringe-rotated, channelised,
  fractional-sample-corrected spectra. The unfused pipeline with `--rotator double` is gpudifx
  today: its decode + fringe kernel (copied verbatim into `gpudecode_copy.cuh`), cuFFT and
  mpifxcorr's rotation, all with FP64 phases. Also timed: **two kernels**, decode + fringe into
  an FP16 or FP32 buffer, then the fused FFT + correction reading it.
- **the cross-multiply** (`xmacbench`, a separate program): the XMAC stage on its own, from
  station spectra in the layout the fused kernel writes. gpudifx's kernel (copied verbatim into
  `xmac_copy.cuh`), a register-tiled FP32 kernel, and a tensor-core kernel (FP16 in, FP32 sums)
  with the transpose into its layout. See "The XMAC stage" below.

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
  too. See "Window design" below.
- **Decode and fringe rotation in the same kernel, if it pays** (`--input packed`), against
  keeping the trip through global memory between fringe rotation and the FFT (two kernels).
  See "From the packed data" below.

## Build and run

With a gpudifx `setup.bash` sourced, `make -j8` is all that is needed. The Makefile takes CUDA
from `CUDAROOT` and the GPU architecture from the `-arch=sm_NN` in `NVCCFLAGS`. Without them it
falls back to the `nvcc` on `PATH` and the GPU that `nvidia-smi` reports. It prints what it
chose at the start of every build.

cuFFTDx is part of NVIDIA's header-only MathDx package. It is not in the CUDA toolkit, and
NVIDIA's licence does not let us commit it here. On the first build the Makefile downloads the
`nvidia-mathdx` 25.6.0 wheel from PyPI, checks its sha256 and unpacks it into
`mathdx-25.6.0/` (git-ignored). On a machine whose build nodes have no network access, run
`make mathdx` first on one that does (an OzSTAR login node). To use a MathDx you already have,
set `MATHDX` to the directory that holds `include/cufftdx.hpp` and `external/cutlass`. Tested
with MathDx 25.6.0 (cuFFTDx 1.5.0) and CUDA 13.2.

```bash
source <gpudifx>/setup.bash
make -j8                                                             # as setup.bash says
make -j8 SM=80                                                       # A100, whatever setup.bash says
make -j8 MATHDX=<mathdx dir> CUDA_HOME=/usr/local/cuda-13.2          # everything given explicitly
./fusebench --help
./fusebench --nchan 128 --nwin 10000 --input fp16 --wola --fpb sweep  # one case
./fusebench --nchan 128 --nwin 10000 --input packed --wola --ept sweep # from the packed data
./run-bench.sh                                                       # the standard set below
./run-nchan.sh                                                       # cost against channel count
./xmacbench --nant 10                                                # the XMAC stage, 10 stations
./run-xmac.sh                                                        # XMAC against the number of stations
./run-staged.sh                                                      # the staged XMAC's super-blocks, 4-128 stations
./run-wide.sh                                                        # the whole chain: bits x subbands, 4-64 stations
./run-matrix.sh                                                      # the whole chain over 108 shapes (corner cases)
./run-twostage.sh                                                    # two-stage channelisation: PFB parameters, ~64k channels (NCHAN=8192 NWIN=128 for ~8k)
python3 window_response.py 256                                       # window design numbers
python3 plot_windows.py                                              # the plots in plots/
python3 plot_xmac_paths.py                                           # plots/xmac-paths.svg
```

On OzSTAR: run `make mathdx` on a login node, then `sbatch fusebench-ozstar.sbatch` from
`fft-fusion`. The job sources your gpudifx setup (`DIFX_SETUP`, default
`$HOME/setup_difx.claude`: module loads, then `setup.bash`), builds for the GPU it gets and runs
`run-bench.sh`, `run-nchan.sh`, `run-xmac.sh` and `run-staged.sh`. Its header has the details.
`ncu-ozstar.sbatch`, submitted the same way, profiles the kernels with Nsight Compute (`--set
full`) and packs the reports.

The fused kernel is compiled once per FFT length (`SIZES` in the Makefile, default 64 to
16384), in all its variants. That takes about 4 minutes with `-j8` (8 cores); `SIZES="256 2048"`
builds just the two the standard set uses. `fusebench` exits non-zero if a check fails.
`run-bench.sh` takes about 30 s and writes its logs to `results-local/<host>-<stamp>/`
(git-ignored).  `results/` (in the repository) holds summaries only, one CSV row per shape:
`run-matrix.sh` writes `results/<host>-matrix-<stamp>.csv` (columns: `matrix_summary.py`), from
`fusebench --csv`, which appends every timing and check of a run as CSV rows.

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

## Code layout: adding the pre-FFT steps

| file | what |
|---|---|
| `common.cuh` | `Geometry` (the `[ant][window][band][fftchannels]` layout of mpifxcorr's buffers), error macros |
| `stages.cuh` | the stages either side of the FFT: `PreLoad<T, SERIES>` (FP32/FP16 storage, per-window or series layout, optional window), `withPre()` (the one place stage types are instantiated); `FracRotateFixed` |
| `decode.cuh` | decode, sideband sign and fixed-point fringe rotation from the packed data, as pre-FFT stages: `PreDecodeLean<NBIT>` (the default) and `PreDecode` (gpudifx's per-sample arithmetic, `--decode direct`); the per-window descriptor (`WinDesc`) |
| `gpudecode_copy.cuh` | gpudifx's decode helpers and tiled decode + fringe kernel, copied verbatim (the commit and the two changes are noted there) |
| `unfused.cu` | the copied production kernel, cuFFT plans as `GPUMode` makes them, the kernel that materialises a pre-FFT stage, the generic rotation kernel, the fixed-point phase precompute, and for packed input the fringe phases and window descriptors (`launchPackedSetup`) |
| `fused_inst.cu` | the fused kernel, compiled per FFT length; each length registers its variants (FFTs per block, and cuFFTDx's elements per thread or half of it) at start-up; two outputs: FP32 spectra, and FP16 spectra in the same order (for the staged tensor XMAC) |
| `fusebench.cu` | data, window, checks, timing, options |
| `xmac.cuh`, `xmac.cu` | the XMAC stage: the shapes and layouts, gpudifx's XMAC (the reference) and the staged tensor XMAC from FP16 spectra, split by receiver, reading them through a map (any station layouts) |
| `xmac_copy.cuh` | gpudifx's XMAC kernel, copied verbatim, and its launch geometry as GPUCore sets it |
| `twostage.cuh`, `twostage1.cu`, `twostage2.cu`, `twostagebench.cu` | two-stage channelisation (docs/two-stage.md): the PFB stage (decode once a block, fold, coarse FFT, corrections), the fine-FFT stage, and the benchmark with its double-precision reference |
| `xmacbench.cu` | the XMAC benchmark: station spectra with a common signal, the double-precision check, timing; `--messy` for stations with different band layouts |
| `window_response.py` | sensitivity and spectral response of window designs |

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
with band on the lane as gpudifx's kernel does, was measured below and removed: it never beat
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

## The station + XMAC paths

[docs/xmac-paths.md](docs/xmac-paths.md) shows every path from packed data to visibilities in one
diagram: the layout of each buffer, which kernel writes it and which reads it, each kernel's time.
For each path it says what it does, what it saves, and what it costs.  In short: the end-to-end
kernel removes the station stage's DRAM round trips (8x), and the tensor cores make the XMAC
itself 10x faster.  Their catch is the layout: tensor cores want each receiver's 16 consecutive
windows of a channel as a row, and the station kernel produces channels fastest.  That
rearrangement can go in a transpose kernel (an extra DRAM pass, costing more than the tensor
cores save over the tiled FP32 XMAC),
in the station kernel (the layout path: a slower station kernel, tied to the XMAC), or in each
XMAC block's copy into shared memory (the staged path, the one kept: the cheapest station output,
at the cost of a slower XMAC kernel).  The other paths were retired on 2026-11-20 (Open items).

![The station + XMAC paths and their layouts](plots/xmac-paths.svg)

## Results, 2026-11-16

**ar313's RTX 2070 SUPER is power-capped at 150 W from 2026-11-18** (default 215 W). At 215 W
and 185 W the 7-year-old FSP500 PSU trips during `xmacbench --nchan 1024 --nant 32 --nwin
1000`, which cut the power to the whole box twice on 2026-11-17. The cap is set at boot by
`gpu-powercap.service` (`/etc/sysconfig/gpu-powercap`; investigation and load tests in
`~/packages/claude/ar313-reset-traps/`). **No 2070 result from before 2026-11-18 compares directly
with one from 2026-11-18 on.** The 2070 sections headed "215 W (before 2026-11-18)" are kept for
their analysis; the baselines for new work are in "RTX 2070 SUPER at the 150 W cap" below. The cap
slows the kernels that compute hardest, not only the tensor cores: the end-to-end station kernel
from packed data by 9-14%. The scripts record the limit in force in `system.txt`
(`nvidia-smi --query-gpu=enforced.power.limit --format=csv`): check it before trusting a 2070 run.

**From 2026-11-20 the 2070's SM clock is also locked at 1500 MHz** (`nvidia-smi -lgc 300,1500`, in
the same service), on top of the 150 W cap.  The cap only limits the power averaged over about a
second: between kernels the clock rebounded to 1875-1950 MHz, and the next load step hit the PSU
at full boost voltage, which tripped it again at 150 W (2026-11-18 21:17, during `run-wide.sh`).
The lock bounds the peak at every instant.  **No 2070 result from 2026-11-18 or 2026-11-19 (150 W
only) compares directly with one from 2026-11-20 on (150 W + 1500 MHz)**, nor with the 215 W ones;
compare within a run, or against a baseline from the same settings.  The scripts now also record
the lock in `system.txt` ("SM clock locked: GPU_MAX_CLOCK=1500", from `gpu-powercap.service`:
`nvidia-smi` does not report it).

`run-bench.sh`: 10 antennas, 8 bands, real data, 64 MHz bands, no autocorrelations. Median of
20 subints, ms per subint for all 10 antennas. Fused times are for the best FFTs-per-block
setting, with cuFFTDx's suggested setting in brackets where it differs. "vs mpifxcorr" is
relative to the first row: FP32 input, cuFFT, then mpifxcorr's rotation kernel. WOLA is 12.5%
tapers with hop 0.875 FFT lengths; "WOLA, hop 0.75" is the same window with consecutive windows
sharing 25% (`--overlap 0.25`), for comparison. The 2070 was timed with Tukey tapers, `--wola`'s
shape at the time, and the A100 with Vorbis tapers. The taper shape does not change the cost
(the window is a table of weights; Tukey, sine and Vorbis time the same).

### RTX 2070 SUPER at the 150 W cap (2026-11-18)

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
("The XMAC stage" below): `run-bench.sh` `c128-packed` at 10 stations, gpudifx 31.4 ms, the
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

#### The staged XMAC split by receiver

The second version (`xmac.cu`, "staged tensor") stages only the receivers a block multiplies.
The receivers are grouped in super-blocks of KS x 16; a block takes a pair of super-blocks and a
few channels, and its stage holds just those two super-blocks.  Up to 16 stations there is one
super-block, every receiver, 8 channels a block, as before; above that, super-blocks of 64
receivers (KS 4) with 4 channels and 16 warps a block: 49 KB at most, whatever the number of
stations, at the cost of reading each receiver once per super-block it is paired with.  Two more
changes: two neighbouring lanes now load the two halves of each 32-byte sector, with every load
of a block of windows issued before its stores into the stage (1.1-1.2x at 10-16 stations); and
the channel groups are fastest in the launch, so that blocks reading neighbouring parts of a
window run together (1.3-1.5x from 32 stations up).  Super-blocks of 16 (KS 1) or 32 (KS 2, 8
channels) were 1.9-3.5x and 1.1-2.0x slower than KS 4 from 24 stations up (`xmacbench
--staged-tile`, `results-local/ar313-adeller-staged-20261118-160542`).  Checked against double
precision from 2 to 100 stations, including odd counts, and 1020 channels.

`run-xmac.sh` (`results-local/ar313-adeller-xmac-20261118-163053`, every check passing), ms per
ms of data; 100 stations and 1024 channels at 64 from `xmacbench` alone (2000 and 500 windows):

| channels | stations | XMAC: tensor (from its layout) | XMAC: staged, first version | XMAC: staged, split by receiver |
|---|---|---|---|---|
| 128 | 4 | 0.021 | 0.041 | 0.028 |
| 128 | 6 | 0.031 | 0.048 | 0.034 |
| 128 | 10 | 0.055 | 0.088 | 0.086 |
| 128 | 16 | 0.106 | 0.126 | 0.117 |
| 128 | 24 | 0.186 | 0.367 | 0.269 |
| 128 | 32 | 0.407 | 0.658 | **0.382** |
| 128 | 48 | 0.799 | does not fit | 1.04 |
| 128 | 64 | 1.10 | does not fit | 1.47 |
| 128 | 100 | 2.27 | does not fit | 4.28 |
| 1024 | 10 | 0.057 | 0.129 | 0.119 |
| 1024 | 32 | 0.260 | 1.31 | 0.476 |
| 1024 | 64 | 1.02 | does not fit | 2.35 |

Whole chain (`chain.txt`, the same run):

| channels | stations | gpudifx today | end-to-end + tiled | end-to-end + transpose + tensor | end-to-end writing the layout + tensor | end-to-end writing FP16 spectra + staged |
|---|---|---|---|---|---|---|
| 128 | 4 | 0.737 | 0.133 | 0.192 | 0.131 | **0.122** |
| 128 | 10 | 2.37 | 0.419 | 0.502 | 0.343 | **0.328** |
| 128 | 16 | 4.46 | 0.766 | 0.810 | 0.560 | **0.547** |
| 128 | 32 | 16.7 | 3.29 | 1.72 | 1.27 | **1.20** |
| 1024 | 10 | 2.26 | 0.477 | 0.550 | 0.505 | **0.396** |

- **FP16 spectra + the staged XMAC is now the fastest chain at every shape measured**: 7% ahead of
  the layout path at 4 stations, 2-6% at 10-32, and 22% at 1024 channels and 10 stations (17% ahead
  of tiled there).  It needs no tile in the station kernel and no transpose.
- **Beyond 32 stations it trails the tensor XMAC alone**: 1.3x at 48 and 64 stations, 1.9x at 100,
  1.8-2.3x at 1024 channels.  Counting the station kernel (no chain runs past 32 stations fit in
  8 GB; estimated from the 10-station chain runs): about level with the layout path at 64
  stations and 128 channels, about 7% behind at 1024 channels and 64 stations, about 30% behind
  at 100 stations and 128 channels.  The re-reads are the cost: each receiver is read once per
  super-block, 4 times at 100 stations, partly from DRAM (the 2070's L2 is 4 MB).

### RTX 2070 SUPER, 215 W (before 2026-11-18)

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

#### What the numbers say

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

### A100-SXM4-80GB (OzSTAR, gina7)

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

#### What changes on the A100

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

### From the packed data: RTX 2070 SUPER, 215 W (before 2026-11-18)

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

### Channel count, chunks and register caps: RTX 2070 SUPER, 215 W (before 2026-11-18)

**A correction first.** Commit 36f5b28, which added the register caps, also changed the
uncapped kernel's `__launch_bounds__(threads)` to `__launch_bounds__(threads, 1)`. That should
mean the same thing, but the compiler then gave the kernel more registers (A100: 86-96 instead
of 64-72 at cuFFTDx's elements per thread; 2070: 92-96 instead of 64-80), and some variants
became up to 38% slower on the A100 (56% with the tiled decode). The 2070 tables of that commit
and the A100 run of it (gina303) have the slower uncapped kernels. The uncapped kernel
is now declared as before, and its registers and times are back to those of 501bc92 (within 2%
on the 2070). In a few cases the extra registers had been faster (2070, 2 kernels FP16, WOLA:
10.11 vs 10.44 ms at 128 channels).

The 2070's times vary by about 5% from run to run with its temperature (it is a desktop card),
so compare within a table, not across them.

**Channel count** (`run-nchan.sh`): the same 20 ms of data (2,560,000 samples per antenna and
band) for every channel count, from the packed data, best variant of each pipeline; ms per
subint (lower is better), and the end-to-end kernel relative to 128 channels.

| nchan | windows | end-to-end | 2 kernels, FP16 | 2 kernels, FP32 | end-to-end, WOLA | 2 kernels FP16, WOLA | end-to-end vs 128 ch |
|---|---|---|---|---|---|---|---|
| 32 | 40000 | 6.57 | 9.04 | 12.71 | 7.71 | 10.48 | 0.97 |
| 64 | 20000 | 7.12 | 8.93 | 11.98 | 8.27 | 11.14 | 1.05 |
| 128 | 10000 | 6.77 | 8.51 | 11.27 | 7.92 | 10.55 | 1 |
| 256 | 5000 | 6.81 | 7.94 | 11.52 | 7.86 | 9.29 | 1.01 |
| 512 | 2500 | 7.58 | 7.91 | 11.61 | 8.94 | 9.77 | 1.12 |
| 1024 | 1250 | 7.95 | 7.89 | 11.88 | 9.32 | 9.90 | 1.17 |
| 2048 | 625 | 9.07 | 8.70 | 12.65 | 10.31 | 10.40 | 1.34 |
| 4096 | 312 | 10.45 | 12.46 | 15.94 | 12.04 | 15.67 | 1.54 |
| 8192 | 156 | 12.19 | 13.64 | 16.87 | 14.40 | 17.18 | 1.80 |

(4096 and 8192 channels: 19.97 ms of data, whole windows.) There is no case for a minimum
channel count: 32 channels cost the same as 256. Larger FFTs cost more per unit of data, gently
up to 1024 channels (+17%), then steeply (+34% at 2048, +54% at 4096, +80% at 8192). That is
an argument for two-stage channelisation at high resolution (see the open items). The 2-kernel
pipeline with an FP16 buffer is 38% behind at 32 channels, level at 512-1024 and 4% ahead at
2048, then behind again.

**Chunks** (`--chunk W`): W windows at a time, each antenna on its own stream so that the
antennas' chunks run together, each antenna's part of the 2-kernel buffer reused for every
chunk. All the antennas' chunks are in flight at once, which is what an XMAC reading the
spectra from L2 would need; the chunk sizes give 84, 21, 10.5, 5.2 and 2.6 MB in FP16 at both
channel counts. WOLA, half of cuFFTDx's elements per thread, ms per subint (each row from its
own run; the whole-subint times of those runs agree within 6%):

| chunk (windows) | in flight, FP16 | end-to-end | 2 kernels, FP16 | 2 kernels, FP32 |
|---|---|---|---|---|
| 128 channels: whole subint | | 8.02 | 9.91 | 12.78 |
| 1024 | 84 MB | 8.31 | 10.34 | 13.00 |
| 256 | 21 MB | 8.40 | 10.96 | 13.69 |
| 128 | 10.5 MB | 8.60 | 11.27 | 14.37 |
| 64 | 5.2 MB | 8.85 | 11.98 | 14.81 |
| 32 | 2.6 MB | 9.47 | 13.35 | 15.52 |
| 1024 channels: whole subint | | 7.51 | 8.61 | 11.13 |
| 128 | 84 MB | 7.92 | 8.42 | 10.87 |
| 32 | 21 MB | 7.80 | 8.89 | 11.39 |
| 16 | 10.5 MB | 7.94 | 9.14 | 11.54 |
| 8 | 5.2 MB | 8.19 | 12.73 | 13.94 |
| 4 | 2.6 MB | 8.54 | 18.78 | 19.51 |

Running the antennas together removes most of the cost of chunking: the end-to-end kernel loses
4-18%, against 32% for chunks of 256 windows one antenna at a time (the first version of
`--chunk`). The only chunked 2-kernel time below its whole-subint time is at 84 MB and 1024
channels, 20 times the 2070's 4 MB L2, so it comes from the decode and FFT kernels of
different antennas overlapping, not from L2. Chunks small enough for the 2070's L2 (2.6 MB) are
so small that launches dominate (1024 channels, 4 windows: 5700 launches per subint). So the
2070 cannot show what L2 is worth; the A100 (40 MB) can.

**Register caps** (`--regcap 1024|1536`): `__launch_bounds__` with a minimum of blocks per SM,
to raise occupancy on the A100. A 2070 SM holds only 1024 threads, so a cap allows at most 64
registers there. That changes nothing at half of cuFFTDx's elements per thread (55-58
registers either way). At cuFFTDx's setting it takes some variants from 72-80 registers to 64,
which moves their times by -6% to +5%; none of them is the fastest variant.

### Channel count, chunks and register caps: A100-SXM4-80GB (OzSTAR, gina4)

`fusebench-ozstar.sbatch`, SLURM job 21042592, commit 89b4dfb (the corrected uncapped kernel and
the concurrent chunks), all 30 runs and the channel-count sweep passing their checks; Nsight
Compute `ncu-ozstar.sbatch`, job 21038155 on gina304, commit d74abd3. The uncapped and 2-kernel
times agree with those of job 21016518 (gina10, below) within 0.1%, so that table stands. An
earlier run of the register caps (jobs 21021335 and 21021345, gina303, commit 36f5b28) had the
slower uncapped kernel described above; its capped times agree with these.

**Channel count** (`run-nchan.sh`), as for the 2070: 20 ms of data per antenna and band, from
the packed data, best variant of each pipeline (no register caps), ms per subint.

| nchan | windows | end-to-end | 2 kernels, FP16 | 2 kernels, FP32 | end-to-end, WOLA | 2 kernels FP16, WOLA | end-to-end vs 128 ch |
|---|---|---|---|---|---|---|---|
| 32 | 40000 | 2.58 | 3.08 | 3.36 | 2.95 | 3.78 | 0.90 |
| 64 | 20000 | 2.90 | 3.11 | 3.35 | 3.30 | 3.83 | 1.02 |
| 128 | 10000 | 2.86 | 3.03 | 3.30 | 3.24 | 3.79 | 1 |
| 256 | 5000 | 2.86 | 2.93 | 3.21 | 3.25 | 3.45 | 1.00 |
| 512 | 2500 | 3.28 | 3.36 | 3.28 | 3.75 | 3.89 | 1.15 |
| 1024 | 1250 | 3.07 | 3.01 | 3.10 | 3.51 | 3.68 | 1.08 |
| 2048 | 625 | 3.70 | 3.31 | 3.29 | 4.15 | 4.09 | 1.30 |
| 4096 | 312 | 4.14 | 3.93 | 4.82 | 4.77 | 4.60 | 1.45 |
| 8192 | 156 | 4.44 | 4.81 | 4.85 | 5.26 | 5.60 | 1.56 |

- **The step at 512 channels comes from the variants built, not the GPU.** At 1024 points
  cuFFTDx suggests 32 elements per thread (127 registers), so the sweep's choices there are 32
  and 16 (72 registers); at 256 and 1024 channels the fastest variant uses 8. Building a quarter
  of cuFFTDx's elements per thread as well would probably put 512 channels near 2.9 ms. (The
  sweep of the buggy build had 3.83 ms here.)
- **As on the 2070, nothing argues for a minimum channel count**, and longer FFTs cost more:
  +30% at 2048 channels, +56% at 8192, less steep than the 2070's +34% and +80%.
- **From 1024 channels up the 2 kernels catch the end-to-end kernel**, and with an FP16 buffer
  pass it at 2048 and 4096, with or without WOLA (2048 channels, no window: 3.31 against 3.70
  ms); at 8192 the end-to-end kernel is ahead again. Below 512 channels it takes 2-16% less time
  than the 2 kernels without a window and 6-22% less with WOLA.

**Chunks** (`--chunk W`, all antennas' chunks in flight at once, as for the 2070): WOLA, half of
cuFFTDx's elements per thread, ms per subint. The whole-subint times of the ten runs agree
within 0.1%.

| chunk (windows) | in flight, FP16 | end-to-end | 2 kernels, FP16 | 2 kernels, FP32 |
|---|---|---|---|---|
| 128 channels: whole subint | | 3.30 | 3.60 | 4.02 |
| 1024 | 84 MB | 3.27 | 3.69 | 4.23 |
| 256 | 21 MB | 3.46 | 3.87 | 4.60 |
| 128 | 10.5 MB | 3.64 | 3.74 | 4.04 |
| 64 | 5.2 MB | 4.17 | 7.11 | 7.13 |
| 32 | 2.6 MB | 9.87 | 19.08 | 16.33 |
| 1024 channels: whole subint | | 2.83 | 3.02 | 3.14 |
| 128 | 84 MB | 2.77 | 3.01 | 3.22 |
| 32 | 21 MB | 2.85 | 3.60 | 3.94 |
| 16 | 10.5 MB | 2.93 | 5.15 | 5.43 |
| 8 | 5.2 MB | 4.23 | 9.33 | 9.32 |
| 4 | 2.6 MB | 8.03 | 17.30 | 17.06 |

**Keeping the 2-kernel buffer in L2 does not pay.** With chunks that fit the 40 MB L2 (21 MB
and less in FP16), the 2-kernel pipeline is 4-7% slower than without chunks at 128 channels and
19% or more at 1024, never faster. Below about 10 MB the launches dominate. The 2 kernels are
not limited by memory bandwidth to begin with (Nsight Compute: the decode kernel uses 24% of the
DRAM bandwidth, the fused FFT reading FP16 43%), so taking the buffer's round trip out of DRAM
saves little; their deficit is the work of a second pass. Chunking the pre-processing for its
own sake is not worth it; any case for chunks has to come from the XMAC. The end-to-end kernel
loses nothing in chunks of 84 MB and 1-5% at 21 MB.

**Register caps** (`--regcap`, packed data with WOLA, ms per subint; `wola-fp16` is the fused FFT
reading an FP16 buffer):

| | 128 ch | 1024 ch |
|---|---|---|
| end-to-end, cuFFTDx's elements per thread: uncapped (72 reg) / cap 64 reg | 3.54 / 3.36 | 3.06 / 2.97 |
| end-to-end, half of them: uncapped (48-56 reg) / **cap 40 reg** | 3.24 / **3.03** | 2.83 / **2.76** |
| `wola-fp16`, half: uncapped (48 reg) / cap 54 reg (1024) / cap 40 reg (1536) | 1.70 / 1.79 / 1.97 | |

(best FFTs per block of each row.) Capping pays only where the decode is in the kernel: 40
registers at half of cuFFTDx's elements per thread gives the fastest end-to-end kernel, 6.5%
ahead of uncapped at 128 channels and 2.4% at 1024. A cap of 64 at cuFFTDx's setting gains 3-5%
but stays behind. For the fused FFT on its own the caps cost time (a cap of 40 at cuFFTDx's
setting: 3.92 against 1.85 ms).

What Nsight Compute shows now (job 21038155; ncu runs the GPU at its base clock):
- The corrected uncapped kernels have 64-72 registers and 40-47% occupancy at cuFFTDx's
  elements per thread (86-96 and 28-30% in the buggy build), and run 5-17% faster per launch
  than they did there (end-to-end, 128 channels, WOLA: 390 against 472 µs).
- **The end-to-end kernel is limited by instruction issue, not memory.** It uses 9-13% of the
  DRAM bandwidth and 55-68% of its issue slots, and executes 3.1-3.4x the instructions of the
  fused FFT reading an FP32 buffer (116-126M against 37M per antenna at 128 channels). At 1024
  channels with half of cuFFTDx's elements per thread (61% occupancy), the stalls for an issue
  slot (not selected) and for the arithmetic pipes (math pipe throttle) are rising towards the
  load stalls (long scoreboard). Fewer decode instructions per sample is the lever.

### From the packed data: A100-SXM4-80GB (OzSTAR, gina10)

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

### The lean decode: RTX 2070 SUPER, 215 W (before 2026-11-18)

The A100 profiles above put two-thirds of the end-to-end kernel's instructions in the decode and
fringe rotation, about 100 per sample, most of them finding the sample: a division by the frame
length, the frame's address and its `validFrames` entry, every sample.  `PreDecodeLean<NBIT>`
(`--decode lean`, now the default) does the same arithmetic with the per-window work done once:

- a descriptor per (antenna, window), made with the fringe phases (`launchPackedSetup`): the byte
  offset of the window's first frame, the first sample's place in it, a bit mask of which of the
  window's frames are valid, and the parity of its first sample's number (sideband sign);
- per thread (`prepare`): those, the band's bit offset, the fringe phase of its first sample and
  its step, and the thread's signed levels (with an even sample stride, as in both kernels, all of
  a thread's samples have the same sideband sign);
- per sample (`at`): the frame by a multiply-high, the bit position, one byte load, the level by
  selects, the window weight, the phase advanced by a 64-bit add (exact) and one sincos.  No
  branches: the load of an invalid frame goes to a safe address and its value is zeroed, so that a
  thread's loads can be issued together.  The number of bits is a template parameter (1, 2, 4, 8).

The fused kernel itself now tests whether a thread has a spectrum once, not once per sample (that
test was a branch around every sample's load), and its index arithmetic is 32-bit.  Both help
the direct decode too.  The result is the same as gpudifx's arithmetic bit for bit (`--decode
direct` keeps it, for comparison); the check now also requires both pipelines to be within 1e-5
rms of the double-precision reference for packed input (they are within 2-4e-7), and
`--bad-frames F` tests the invalid-frame handling.  The 2-kernel pipeline's first kernel
(`decode_to`) now gives each warp one band at a time, 32 consecutive samples per pass, so that
the per-thread work is done once per band.

Nsight Compute on the 2070 (128 channels, WOLA, half of cuFFTDx's elements per thread, one
antenna; the stages in the order they were made):

| | instructions | time | decode + rotation per sample | main stall |
|---|---|---|---|---|
| direct decode | 131.2M | 766 µs | ~105 | waiting for loads |
| lean: descriptor, bits fixed at compile time | 93.3M | 554 µs | | |
| + no branches in the decode, window test once | 91.0M | 556 µs | | |
| + phase by addition, byte offset without the sample-in-frame | 89.0M | 551 µs | | |
| + the thread's test once, not per sample | 83.4M | 526 µs | ~45 | shared memory (the FFT's exchange) |

The decode and rotation are now 41% of the kernel's instructions (60% before), the FFT 35%, and
waiting for loads has fallen from the first stall reason to the fourth.  A fringe phase by
recurrence (one sincos per thread, a complex multiply per sample) was tried and dropped: with 8
samples a thread, the per-thread sincos cost what it saved, and it doubled the phase error until
the step was computed accurately.

`run-bench.sh` (`results-local/ar313-adeller-20261117-123426`, all 38 runs passing), best variant
of each line, ms per subint:

| pipeline | 128 ch | 128 ch, WOLA | 1024 ch | 1024 ch, WOLA |
|---|---|---|---|---|
| end-to-end, direct decode | 5.59 | 6.68 | 5.44 | 6.36 |
| **end-to-end, lean decode** | **4.06** | **4.69** | **3.97** | **4.60** |
| 2 kernels -> FP16, lean decode | 2.72 + 4.61 = 7.34 | 3.04 + 5.40 = 8.45 | 2.40 + 3.50 = 5.89 | 2.72 + 4.23 = 6.87 |
| 2 kernels -> FP16, direct decode | 4.02 + 4.61 = 8.65 | 4.60 + 5.41 = 10.01 | 2.92 + 3.50 = 6.48 | 3.37 + 4.27 = 7.57 |
| 2 kernels -> FP32, lean decode | 11.56 | 13.10 | 9.10 | 10.36 |

- **The lean decode takes 27-30% off the end-to-end kernel** at both channel counts, with and
  without a window, and 18-34% off the decode kernel of the 2-kernel pipeline.  On the 2070 the
  2 kernels now take 1.5-1.8 times as long as the end-to-end kernel.
- **cuFFTDx's own elements per thread is now the fastest on the 2070** (16 per thread: 4.69 ms
  at 128 channels with WOLA, against 5.50 at half of it), where half of it was fastest with the
  direct decode, whose registers it saved.  The register caps change nothing here.
- **The number of bits costs little** (end-to-end with WOLA, best variant):

  | bits | 128 ch | 1024 ch |
  |---|---|---|
  | 1 | 4.69 | 4.43 |
  | 2 | 4.69 | 4.60 |
  | 4 | 4.97 | 4.69 |
  | 8 | 4.89 | 4.83 |

  8-bit data is 4 times the bytes of 2-bit (8 bytes per sample time for 8 bands, against 2), and
  costs 4-5% more; 1-bit is no faster than 2-bit.
- **Chunks now cost the end-to-end kernel little** (1024 channels, WOLA, half of cuFFTDx's
  elements per thread: 5.26 ms whole, 5.19-5.67 ms in chunks of 32 down to 4 windows).  The
  2-kernel pipeline with an FP16 buffer gains up to 6% from chunks of 10-21 MB at 1024 channels
  (7.13-7.16 against 7.61 ms), which is far more than the 2070's 4 MB L2, so it comes from
  different antennas' kernels overlapping; at 128 channels it loses 2-16%.

Channel count (`run-nchan.sh`, `results-local/ar313-adeller-nchan-20261117-123703`; as the table
above, lean decode):

| nchan | end-to-end | 2 kernels, FP16 | 2 kernels, FP32 | end-to-end, WOLA | 2 kernels FP16, WOLA | end-to-end vs 128 ch |
|---|---|---|---|---|---|---|
| 32 | 5.02 | 9.18 | 12.21 | 5.69 | 10.58 | 1.12 |
| 64 | 5.29 | 8.30 | 12.18 | 6.07 | 9.67 | 1.18 |
| 128 | 4.50 | 7.31 | 11.48 | 5.19 | 8.47 | 1 |
| 256 | 4.84 | 7.35 | 11.49 | 5.72 | 8.50 | 1.08 |
| 512 | 5.09 | 7.61 | 11.39 | 5.85 | 8.82 | 1.13 |
| 1024 | 5.09 | 7.44 | 11.34 | 6.13 | 8.79 | 1.13 |
| 2048 | 5.66 | 7.44 | 11.50 | 6.40 | 8.75 | 1.26 |
| 4096 | 7.14 | 8.82 | 11.63 | 8.11 | 10.39 | 1.59 |
| 8192 | 7.97 | 10.41 | 14.21 | 9.28 | 13.74 | 1.77 |

Against yesterday's sweep on this card the end-to-end kernel is 24-38% faster at every channel
count, and now ahead of the 2 kernels at all of them.

### The lean decode: A100-SXM4-80GB (OzSTAR, gina15)

`fusebench-ozstar.sbatch`, SLURM job 21049107, commit 6712c95, all 38 runs and the channel-count
sweep passing their checks; Nsight Compute `ncu-ozstar.sbatch`, job 21049108 on gina9.  Best
variant of each line (register caps included), ms per subint:

| pipeline | 128 ch | 128 ch, WOLA | 1024 ch | 1024 ch, WOLA |
|---|---|---|---|---|
| end-to-end, before today (gina4, job 21042592) | 2.85 | 3.03 | 2.48 | 2.76 |
| end-to-end, direct decode (today's kernel) | 2.34 | 2.58 | 2.17 | 2.43 |
| **end-to-end, lean decode** | **1.51** | **1.73** | **1.50** | **1.68** |
| 2 kernels -> FP16, lean decode | 1.59 + 1.28 = 2.87 | 1.79 + 1.45 = 3.24 | 1.11 + 1.16 = 2.28 | 1.25 + 1.34 = 2.58 |
| 2 kernels -> FP16, direct decode | 1.76 + 1.28 = 3.03 | 1.98 + 1.45 = 3.42 | 1.25 + 1.16 = 2.41 | 1.41 + 1.34 = 2.75 |
| 2 kernels -> FP32, lean decode | 2.89 | 3.33 | 2.40 | 2.75 |

- **The end-to-end kernel is 39-47% faster than this morning.**  Testing a thread's spectrum once
  rather than per sample took 12-18% off (the direct-decode row), and the lean decode 31-36% off
  that.  With WOLA it now runs at 11.6x real time at 128 channels and 11.9x
  at 1024 for this array (10 antennas, 8 x 64 MHz bands, this stage alone).
- **The 2 kernels now take 1.5-1.9 times as long** (0.98-1.19 times this morning): the end-to-end
  kernel is the one to build.  Its decode costs little: per launch it takes 187 µs against 160
  µs for the fused FFT reading a ready FP32 buffer and 142 µs reading FP16 (Nsight Compute, 128
  channels, WOLA, base clock).
- **cuFFTDx's own elements per thread is now the fastest on the A100 too** (16; half of it: 1.84
  and 1.81 ms with WOLA), and the register caps no longer help: 40 registers at half (yesterday's
  best) gives 1.82 ms at 128 channels; a cap of 64 at cuFFTDx's setting gains 1.5% at 1024
  channels and nothing at 128.  The lean end-to-end kernel needs only 48-56 registers at 16
  elements per thread (72 with the direct decode).
- **What limits it now is arithmetic issue**: 69% of issue slots used, the FMA pipe 66% busy
  (the FFT's own arithmetic and the integer multiply-adds; the fused FFT of an FP16 buffer: 70%),
  the special-function unit (sincos, conversions) 28%; DRAM 22%.  The main stalls are waiting for
  an issue slot and for the arithmetic pipes, no longer for loads.
- **The number of bits costs little** (end-to-end with WOLA, best variant):

  | bits | 128 ch | 1024 ch |
  |---|---|---|
  | 1 | 1.69 | 1.65 |
  | 2 | 1.73 | 1.68 |
  | 4 | 1.80 | 1.72 |
  | 8 | 1.76 | 1.67 |

  8-bit data, 4 times the bytes of 2-bit, costs no more; 4-bit is the slowest, by 2-4%.
- **Chunks**: the end-to-end kernel loses nothing in chunks down to 21 MB in flight (128
  channels: 1.80-1.82 ms against 1.84 whole; 1024: 1.76-1.78 against 1.81) and 0-5% at 10.5 MB;
  below that the launches dominate (128 channels in chunks of 64 windows: 1790 launches per
  subint, 3.75 ms).  Keeping the 2-kernel buffer in L2 still does not pay (21 MB: 3.35 against
  3.32 ms at 128 channels, 2.97 against 2.76 at 1024).

Channel count (`run-nchan.sh`), lean decode, ms per subint:

| nchan | end-to-end | 2 kernels, FP16 | 2 kernels, FP32 | end-to-end, WOLA | 2 kernels FP16, WOLA | end-to-end vs 128 ch |
|---|---|---|---|---|---|---|
| 32 | 1.45 | 3.52 | 3.97 | 1.68 | 3.99 | 0.96 |
| 64 | 1.53 | 2.47 | 3.02 | 1.75 | 2.82 | 1.02 |
| 128 | 1.51 | 2.87 | 2.88 | 1.73 | 3.23 | 1 |
| 256 | 1.62 | 2.81 | 2.86 | 1.87 | 3.19 | 1.08 |
| 512 | 1.74 | 2.95 | 2.98 | 2.00 | 3.40 | 1.16 |
| 1024 | 1.84 | 2.81 | 2.95 | 2.11 | 3.19 | 1.22 |
| 2048 | 2.14 | 2.98 | 2.99 | 2.43 | 3.43 | 1.42 |
| 4096 | 2.65 | 3.62 | 3.60 | 3.01 | 4.17 | 1.76 |
| 8192 | 3.10 | 4.23 | 4.20 | 3.64 | 5.07 | 2.06 |

The end-to-end kernel is ahead of the 2 kernels at every channel count now (this morning they
passed it from 2048 channels up), and the step at 512 channels has gone: with the lean decode,
16 elements per thread at 1024 points needs 48 registers (72 before) and is the fastest.  Longer FFTs
still cost more: +42% at 2048 channels, twice the time at 8192.

### The XMAC stage: RTX 2070 SUPER, 215 W (before 2026-11-18)

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
  The fused kernel writing it, and the two stages together: see "The whole chain" below.

### The XMAC stage: A100-SXM4-80GB (OzSTAR, gina9)

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

### The whole chain, station processing + XMAC: RTX 2070 SUPER, 215 W (before 2026-11-18)

`fusebench --xmac` runs the XMAC after the station processing from 2-bit packed data (no window),
in sequence on one stream, with a fifth way in: the end-to-end kernel **writing the tensor XMAC's
layout itself** (`fused_fft_tensor` in `fused_inst.cu`), FP16, instead of its FP32 spectra.  A
block takes W consecutive windows of one (antenna, band), FFTs-per-block at a time, puts each
pass's channels into a shared-memory tile [plane][channel][window] (rows padded by one 32-bit
word against bank conflicts), then writes each channel's run of W windows with 16-byte stores.
W is 16 when the FFT's shared memory and the tile fit in a block, else 8, else FFTs per block.
The spectra are scaled by 2^-floor(log2(N) / 2) on the way to FP16 (N the FFT length), the
visibilities by its square.  Checks: the XMAC from the kernel's layout matches the XMAC through
the transpose to 1-2e-9 rms (the same FP16 values), and gpudifx's FP32 XMAC to 3e-4 rms.

Three things decided its cost (128 channels, 10 stations; the end-to-end kernel alone 4.06 ms):
- **Block order.**  With the band slowest in the launch, the blocks of one window group's 8 bands,
  which read the same packed bytes, ran far apart, and the packed data was read from DRAM 6.6
  times over (52.9 MB instead of 8.0): 15.2 ms.  Band fastest: 4.95 ms.
- **The stores.**  Padding the tile rows split the 16-byte global stores into four 4-byte ones
  (6.2 ms); `__stcg` of one `uint4` built from four 32-bit shared loads restores them: 4.79 ms.
- **FFTs per block**: 16 is best at 128 channels (4.89 ms; 5.81 at 8, 5.47 at 32, 12.9 at 1),
  4 at 1024 (6.43 ms).

`run-xmac.sh` (`results-local/ar313-adeller-xmac-20261117-171357`, `chain.txt`): ms per ms of
data, the fastest variant of each kind (`--fpb sweep --ept sweep`; the station kernel's variant
for FP32 spectra and for the tensor layout chosen separately).  gpudifx today: its station
processing (FP64 phases) and its XMAC.  10000 windows (2500 at 16 and 32 stations), 1000 at 1024
channels:

| channels | stations | gpudifx today | end-to-end + gpudifx XMAC | end-to-end + tiled | end-to-end + transpose + tensor | end-to-end writing the layout + tensor |
|---|---|---|---|---|---|---|
| 128 | 4 | 0.733 | 0.184 | 0.121 | 0.172 | **0.118** |
| 128 | 10 | 2.30 | 0.926 | 0.370 | 0.458 | **0.307** |
| 128 | 16 | 4.25 | 2.09 | 0.662 | 0.724 | **0.506** |
| 128 | 32 | 15.8 | 11.4 | 3.19 | 1.58 | **1.17** |
| 1024 | 10 | 2.15 | 0.914 | **0.435** | 0.497 | 0.462 |

Each combined time is the sum of its parts to 2%.  At 10 stations, ms per subint:

| | 128 channels (20 ms) | 1024 channels (16 ms) |
|---|---|---|
| end-to-end kernel, FP32 spectra | 4.06 | 3.92 |
| end-to-end kernel writing the tensor layout | 4.89 (+0.83) | 6.43 (+2.51) |
| transpose | 3.97 | 3.19 |
| XMAC: tiled / tensor | 3.26 / 1.11 | 3.13 / 0.92 |

- **Writing the layout beats the transpose at every shape**, by a lot at 128 channels (it adds
  0.83 ms to the station kernel and saves the 3.97 ms transpose), by 7% at 1024 (it adds 2.5 ms).
  With it the tensor path is the fastest chain at 128 channels from 4 stations up (a tie at 4,
  17% ahead of tiled at 10, 63% at 32), and 7.5 times gpudifx today at 10 stations.
- **At 1024 channels and 10 stations the tiled FP32 XMAC is still the fastest chain** (6% ahead).
  The tile is the cause: it holds 16 windows of every channel, so it grows with the channel
  count.  On the 2070 the FFT's shared memory (32 KB at 4 FFTs per block) and the tile for 16
  windows (72 KB), or even 8 (40 KB), do not fit in sm_75's 64 KB a block, so the fastest
  variant writes runs of 4 windows (8-byte stores); on the A100, where the 16-window tile fits,
  it costs occupancy instead (below).
- Transpose + tensor is behind tiled at 10 stations on the 2070 (24% at 128 channels, 14% at
  1024), because the transpose alone costs more than the tiled XMAC here; in the chain it is
  still behind at 16 stations (9%) and well ahead at 32.
- **The tiled XMAC slows more than in proportion to the subint length**: at 16 stations (128
  channels) `xmacbench` takes 1.52 ms for 2500 windows but 12.3 ms for 10000, 8 times the time
  for 4 times the data, while the transpose and the tensor XMAC scale with the windows.  So the
  tiled column of the XMAC table above (10000 windows up to 16 stations) is probably pessimistic,
  and where the tensor path overtakes the tiled one depends on the windows per subint.  Not yet
  understood (open items).

### The whole chain, station processing + XMAC: A100-SXM4-80GB (OzSTAR, gina4)

`fusebench-ozstar.sbatch`, SLURM job 21060735, commit e4b055d, all runs passing (`run-bench.sh`
38, `run-xmac.sh` 10 + 7 chain runs, and the channel sweep); `ncu-ozstar.sbatch` job 21060804.
`chain.txt` (`results-local/gina4-xmac-20261117-180958`), ms per ms of data, 10000 windows (1000
at 1024 channels):

| channels | stations | gpudifx today | end-to-end + gpudifx XMAC | end-to-end + tiled | end-to-end + transpose + tensor | end-to-end writing the layout + tensor |
|---|---|---|---|---|---|---|
| 128 | 4 | 0.167 | 0.0734 | 0.0459 | 0.0537 | **0.0428** |
| 128 | 10 | 0.590 | 0.356 | 0.126 | 0.134 | **0.107** |
| 128 | 16 | 1.20 | 0.828 | 0.241 | 0.208 | **0.166** |
| 128 | 32 | 4.05 | 3.29 | 0.677 | 0.427 | **0.344** |
| 128 | 64 | 14.5 | 13.0 | 3.02 | 0.921 | **0.755** |
| 1024 | 10 | 0.571 | 0.362 | **0.146** | 0.155 | 0.173 |
| 1024 | 32 | 3.46 | 2.79 | 0.872 | **0.490** | 0.564 |

At 10 stations, ms per subint (the fastest variant of each):

| | 128 channels (20 ms) | 1024 channels (16 ms) |
|---|---|---|
| end-to-end kernel, FP32 spectra | 1.50 (4 FFTs a block) | 1.49 (1) |
| end-to-end kernel writing the tensor layout | 1.79 (+0.29; 8 FFTs a block) | 2.45 (+0.96; 8) |
| transpose | 0.86 | 0.69 |
| XMAC: tiled / tensor | 1.01 / 0.35 | 0.84 / 0.31 |

- **At 128 channels writing the layout is the fastest chain at every station count**: 15% ahead
  of tiled at 10 stations, 18% ahead of transpose + tensor at 64, 5.5 times gpudifx today at 10
  stations and 19 times at 64.
- **At 1024 channels it is the slowest of the three**: it costs the station kernel 0.96 ms, more
  than the 0.69 ms transpose it saves.  Tiled leads at 10 stations, transpose + tensor at 32.
- **Why: occupancy.**  Nsight (`ncu-gina4-20261117-180735`, base clocks), the kernel writing the
  layout at 4 FFTs a block against the FP32 kernel:

  | | layout, 1024 channels | FP32 (WOLA), 1024 channels | layout, 128 channels |
  |---|---|---|---|
  | shared memory a block | 106.5 KB (the tile 72 KB) | 8 KB | 16 KB |
  | occupancy (achieved) | 25% (1 block an SM) | 53% | 48% |
  | issue slots used | 42% | 66% | 60% |
  | instructions | 62.5 M | 66.5 M | 67.4 M |
  | DRAM written | 26.7 MB | 57.6 MB | 34.3 MB |

  The work is the same and the output half the bytes; the 16-window tile leaves one block an SM,
  too few warps to hide the latency.  The tile is 2 x channels x 18 x 2 bytes: 9 KB at 128
  channels, 72 KB at 1024, 576 KB at 8192, where only W = 1 fits (lone 2-byte stores, a
  sixteenth of a sector each, left to L2 to merge).  Writing the layout from the station kernel
  does not scale with the channel count; see the open items for the alternatives.

### The staged XMAC split by receiver: A100-SXM4-80GB (OzSTAR, gina302)

`fusebench-ozstar.sbatch`, SLURM job 21141283, commit a790ec0, every check passing (`run-bench.sh`
38, `run-nchan.sh`, `run-xmac.sh` 10 + 8 chain runs, `run-staged.sh` 14 x 5 runs).  gpudifx today's
chain times match job 21060735's (gina4) to 0.1%.  `results-local/gina302-staged-20261118-171734`
(`table.txt`) and `gina302-xmac-20261118-171514` (`chain.txt`), ms per ms of data, 10000 windows at
128 channels, 1000 at 1024 (the chain at 100 stations: 4000):

| channels | stations | XMAC: tensor (from its layout) | XMAC: transpose + tensor | XMAC: staged | staged / tensor | gpudifx today, chain | layout + tensor, chain | FP16 spectra + staged, chain |
|---|---|---|---|---|---|---|---|---|
| 128 | 4 | 0.0065 | 0.0236 | 0.0099 | 1.5 | 0.167 | 0.0429 | **0.0409** |
| 128 | 6 | 0.0087 | 0.0352 | 0.0112 | 1.3 | | | |
| 128 | 10 | 0.0172 | 0.0604 | 0.0351 | 2.0 | 0.590 | **0.107** | 0.109 |
| 128 | 16 | 0.0236 | 0.0911 | 0.0420 | 1.8 | 1.20 | 0.166 | **0.164** |
| 128 | 24 | 0.0392 | 0.137 | 0.0997 | 2.5 | | | |
| 128 | 32 | 0.0579 | 0.189 | 0.125 | 2.2 | 4.04 | **0.344** | 0.368 |
| 128 | 48 | 0.108 | 0.304 | 0.313 | 2.9 | | | |
| 128 | 64 | 0.183 | 0.443 | 0.398 | 2.2 | 14.5 | **0.756** | 0.884 |
| 128 | 100 | 0.448 | 0.854 | 1.15 | 2.6 | 32.9 | **1.41** | 1.98 |
| 128 | 128 | 0.670 | 1.19 | 1.41 | 2.1 | | | |
| 1024 | 10 | 0.0202 | 0.0641 | 0.0380 | 1.9 | 0.572 | 0.173 (tiled 0.146) | **0.131** |
| 1024 | 32 | 0.0641 | 0.194 | 0.130 | 2.0 | 3.46 | 0.564 (transpose + tensor 0.490) | **0.427** |
| 1024 | 64 | 0.210 | 0.468 | 0.468 | 2.2 | | | |
| 1024 | 100 | 0.518 | 0.922 | 1.41 | 2.7 | | | |

- **Up to 16 stations the staged path is level with the best chain or ahead of it**: 5% ahead of
  the layout path at 4 stations, 2% behind at 10, 1% ahead at 16.  At 1024 channels it is the
  fastest chain by 10-13% (24% ahead of the layout path, which writes its tile at low occupancy
  here).  Against gpudifx today: 4.1x at 4 stations, 5.4x at 10, 7.3x at 16, 11x at 32, 16x at 64
  and 100; 4.4x and 8.1x at 1024 channels.
- **From 32 stations it falls behind the layout path**: 7% at 32, 17% at 64, 41% at 100.  The
  staged XMAC alone beats transpose + tensor up to 64 stations (and at 1024 channels up to 32), and
  trails it at 100-128.
- **The A100's 40 MB of L2 does not close the gap; the staging itself is the cost here.**  The
  staged XMAC is 1.3-2.9x the tensor XMAC at every station count, and 1.5x already at 4 stations,
  where it reads every receiver once.  Against the 2070 the tensor XMAC gains 3.2x at 10 stations,
  the staged one 2.3x.  Probably its occupancy (not yet profiled on the A100): KS 2 and KS 4 use
  128 and 112 registers a thread (sm_75 build), which allows 16 warps an SM, a quarter of the
  A100's 64; and the stores into the stage have 2-way bank conflicts, with two barriers per 16
  windows.
- **The automatic choice holds within 4% except at 48 stations**: KS 1 at 4-6 stations, 2 at 10-16,
  4 from 24 (KS 2 ties at 24 and is 8% faster at 48); KS 4 is 1.2-1.4x faster than KS 2 from 64
  stations up, and KS 1 is 1.4-3.3x slower than the choice from 24 up.

#### Why the staged XMAC trails on the A100, and the mma.sync version

Nsight Compute on the A100 (`ncu-ozstar.sbatch` with `CASES='^x'`, job 21142001 on gina10,
`results-local/ncu-gina10-20261118-172739`; base clocks):

| | staged, 10 stations (KS 2) | tensor, 10 | staged, 64 (KS 4) | staged, 64 (KS 2) | tensor, 64 |
|---|---|---|---|---|---|
| duration | 0.70 ms | 0.36 ms | 8.66 ms | 10.7 ms | 3.93 ms |
| warps active (of 64) | 24% | 40% | 25% | 25% | 41% |
| instructions | 139 M | | 1.11 G | | 0.71 G |
| DRAM / L2 hit rate | 31% / 67% | 58% / 41% | 30% / 71% | 59% / 62% | 40% / 72% |
| stalls per instruction issued | global loads 3.7, barrier 1.1 | | global loads 6.6, barrier 3.5 | | global loads 13.5 |

It was latency-bound, not limited by bandwidth (the re-reads came from L2): a quarter of the
warps, each waiting on its global loads and then at the barriers of its load, barrier, products,
barrier cycle with no other block to fill the gaps, and 1.6x the tensor XMAC's instructions (the
rearranging in registers).  At 10-16 stations KS 2 also spilled at its 128-register cap (CUDA 12.8).

So the staged XMAC is now written with PTX `mma.sync` (`xmac_staged_kernel`; the WMMA version stays
as `--staged-impl wmma` for comparison).  The same super-blocks, items and plan; what changes:
- **The stage holds the spectra as written, (re, im) interleaved**: a row is 16 windows x (re, im),
  64 bytes, rearranged nowhere.  Along the sum, (re, im) pairs: one product gives re1 re2 + im1 im2,
  the real part of X1 X2*; the imaginary part, im1 re2 - re1 im2, is the product with A's pairs
  turned into (im, -re), a byte permute and a sign flip per register, which `mma.sync`'s documented
  fragment layout (a (re, im) pair per register) allows and WMMA's does not.  The same four
  tensor-core multiplies per 16 windows.  A row's 16-byte chunks are XOR-swizzled, so `ldmatrix`
  reads without bank conflicts and without padding: 33 KB a stage at KS 4 instead of 49.
- **Each warp loads each A and B fragment once per 8 windows** and uses it for all its items; the
  sums go to the visibilities from the accumulator registers.
- **The copy**: a task is 4 windows of 4 channels of one receiver, four 16-byte loads, transposed
  in registers into four 16-byte stores, one per channel, each exactly one chunk of the row.  The
  next 16 windows load into registers while the tensor cores work on these and are stored after:
  into the other of two stages where two stages cost no blocks an SM (the occupancy calculator
  decides: on the A100 always, on the 2070 at KS 4 with one super-block), with one barrier per 16
  windows; else into the one stage, with two.  On the 2070 two stages halve the blocks an SM at
  KS 2 (64 KB of shared memory an SM) and were 1.15-1.35x slower there than one.
- Registers: 119, 128 and 63 (KS 4, 2, 1; KS 1 capped at 64 for 4 blocks an SM), no spills.

Checked against double precision from 2 to 100 stations (odd counts, 1020 channels, 17 and 1001
windows, flagged windows, each KS): the same errors as the WMMA version.

On the 2070 (`run-staged.sh`, `results-local/ar313-adeller-staged-20261118-191617`, every check
passing), ms per subint:

| channels | stations | tensor XMAC alone | transpose + tensor | staged, mma.sync | staged, WMMA | WMMA / mma.sync |
|---|---|---|---|---|---|---|
| 128 | 4 | 0.418 | 2.03 | 0.444 | 0.466 | 1.05 |
| 128 | 6 | 0.623 | 3.04 | 0.619 | 0.655 | 1.06 |
| 128 | 10 | 1.11 | 5.12 | 1.47 | 1.61 | 1.09 |
| 128 | 16 | 2.11 | 7.84 | 1.97 | 2.30 | 1.16 |
| 128 | 24 | 1.49 | 4.96 | 1.77 | 2.06 | 1.17 |
| 128 | 32 | 3.25 | 7.87 | **2.58** | 2.99 | 1.16 |
| 128 | 48 | 6.38 | 13.3 | 6.69 | 8.28 | 1.24 |
| 128 | 64 | 8.99 | 18.2 | 9.47 | 11.7 | 1.23 |
| 128 | 100 | 9.23 | 16.5 | 14.2 | 17.1 | 1.20 |
| 128 | 128 | 14.3 | 23.4 | 19.0 | 23.2 | 1.22 |
| 1024 | 10 | 0.915 | 4.17 | 1.63 | 1.88 | 1.15 |
| 1024 | 32 | 3.44 | 12.7 | 7.62 | 7.63 | 1.00 |
| 1024 | 64 | 6.85 | 15.7 | 17.0 | 18.9 | 1.11 |
| 1024 | 100 | 23.2 | 30.7 | 28.3 | 30.4 | 1.07 |

(24-128 stations at 4000 or 2000 windows, 1024 channels at 1000 to 250: compare along a row.)  The
mma.sync version is level with or faster than WMMA everywhere, 1.05-1.24x.  Against the tensor XMAC
alone (which needs the layout written by the station kernel): within 6% at 4-6 stations, ahead at
16 and 32, 1.05x behind at 48-64, 1.2-1.3x at 10 and 24, 1.3-1.5x at 100-128 and 1.8-2.5x at 1024
channels from 10 to 64 stations.  Faster than transpose + tensor at every shape.  Its own KS choice
is within 4% of the best forced one at every row (the same configuration differed by up to 7%
between runs, at 32 stations).

The first version's copy took a task as one window (16 bytes, 4 channels), with 4-byte writes
into the four channels' stages: on sm_75 a load and four stores, level with WMMA on the 2070; on
sm_80 four 4-byte `cp.async` into two stages, with no registers in between.  On the A100 that
was 1.3-1.6x slower than the WMMA version wherever KS 4 is used (24 stations up: 12.5 against
8.0 ms at 64 stations, 2.84 against 2.08 at 1024 channels and 32 stations) and level to 1.2x slower
below (job 21143266 on gina1, commit 378af9e, `results-local/gina1-staged-20261118-181811`).  A
`cp.async` cannot split 16 bytes across four stages, so each of the four reads the same 32
sectors (one a lane: the lanes are 16 windows of 2 rows) for 4 bytes of each, four times the L1
traffic of one 16-byte load; KS 4, with one channel quad a block, has no neighbouring lane in the
same sector to share it.

#### The mma.sync staged XMAC on the A100 (OzSTAR, gina302)

`fusebench-ozstar.sbatch` with `SCRIPTS="run-xmac.sh run-staged.sh"`, SLURM job 21148710, commit
4a08705, every check passing; two stages at every shape (the registers limit the blocks an SM
first).  `results-local/gina302-staged-20261118-200436` (`table.txt`) and
`gina302-xmac-20261118-200210` (`chain.txt`), ms per ms of data, 10000 windows at 128 channels,
1000 at 1024 (the chain at 100 stations: 4000):

| channels | stations | XMAC: tensor (from its layout) | XMAC: staged, mma.sync | staged / tensor | staged, WMMA / mma.sync | gpudifx today, chain | layout + tensor, chain | FP16 spectra + staged, chain | chain gain over gpudifx today |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 4 | 0.0065 | 0.0095 | 1.46 | 1.04 | 0.167 | 0.0428 | **0.0404** | 4.1x |
| 128 | 6 | 0.0087 | 0.0114 | 1.31 | 1.04 | | | | |
| 128 | 10 | 0.0180 | 0.0234 | 1.30 | 1.39 | 0.590 | 0.107 | **0.0996** | 5.9x |
| 128 | 16 | 0.0230 | 0.0328 | 1.42 | 1.28 | 1.20 | 0.166 | **0.155** | 7.8x |
| 128 | 24 | 0.0392 | 0.0712 | 1.82 | 1.40 | | | | |
| 128 | 32 | 0.0578 | 0.0927 | 1.60 | 1.34 | 4.05 | 0.343 | **0.336** | 12x |
| 128 | 48 | 0.109 | 0.241 | 2.22 | 1.29 | | | | |
| 128 | 64 | 0.183 | 0.317 | 1.73 | 1.26 | 14.5 | **0.752** | 0.801 | 18x |
| 128 | 100 | 0.444 | 0.918 | 2.07 | 1.25 | 32.8 | **1.41** | 1.76 | 19x |
| 128 | 128 | 0.664 | 1.16 | 1.75 | 1.22 | | | | |
| 1024 | 10 | 0.0201 | 0.0277 | 1.38 | 1.37 | 0.571 | 0.173 (tiled 0.146) | **0.121** | 4.7x |
| 1024 | 32 | 0.0641 | 0.111 | 1.73 | 1.17 | 3.46 | 0.563 (transpose + tensor 0.489) | **0.408** | 8.5x |
| 1024 | 64 | 0.210 | 0.424 | 2.02 | 1.10 | | | | |
| 1024 | 100 | 0.518 | 1.31 | 2.53 | 1.08 | | | | |

- **The mma.sync version is 1.1-1.4x faster than WMMA from 10 stations up** (4% at 4-6), where
  the cp.async first version was 1.3-1.6x slower.  Against the WMMA run of job 21141283 (a790ec0,
  above) the staged XMAC alone gains 1.5x at 10 stations, 1.3x at 16-32, 1.25x at 64-128.
- **FP16 spectra + the staged XMAC is now the fastest chain from 4 to 32 stations and at 1024
  channels**: 6% ahead of the layout path at 4 stations, 7% at 10 and 16, 2% at 32, and 17% ahead
  of the next best at 1024 channels (10 and 32 stations).  It trails the layout path by 6% at 64
  stations and 25% at 100 (was 17% and 41%).
- **The staged XMAC alone is still 1.3-2.5x the tensor XMAC from its layout**; what remains from
  24 stations up is mostly KS 4's re-reads (each receiver once per super-block: 2 at 64 stations,
  4 at 100 and 128, 2 at 1024 channels and 64 stations), and the A100 keeps one block of 16
  warps an SM at KS 4 (119 registers).  Its 164 KB of shared memory would hold 128-receiver
  super-blocks (KS 8, about 65 KB a stage at 4 channels), halving the re-reads from 100 stations,
  with the items split differently among the warps (a KS 4 warp's sums already take 64
  registers).

### Wider shapes, bits and subbands: RTX 2070 SUPER at 150 W + 1500 MHz (2026-11-20)

`run-wide.sh` (new): the whole chain (`fusebench --xmac`, every variant timed, every check) for 1,
2 and 4 bits a sample x 1, 4 and 32 dual-polarisation subbands (2, 8, 64 bands) x 128 channels at
4, 10 and 32 stations and 1024 channels at 10.  The end-to-end kernel's tensor-layout and FP16
outputs are now compiled for the 1- and 4-bit lean decode as well as 2 bits (`fused_inst.cu`,
`xmacDecode`), and `--xmac` takes `--nbit 1`, `2` or `4`.  The windows scale with the bands
(10000 at 128 channels and up to 8 bands, 1250 at 64; a quarter of that at 32 stations; at 1024
channels 1000 and 125).  `results-local/ar313-adeller-wide-20261120-115150` (`wide.txt`), all 36
runs passing, commit 4ba1d8c plus these changes; the first attempt (18 Nov, 150 W only) cut the
power to the box during it (see the note on the clock lock above).

FP16 spectra + staged, ms per ms of data, and in brackets its time over the best other chain
(end-to-end + tiled, + transpose + tensor, or writing the layout + tensor; below 1: the staged
path is the fastest):

| bits | subbands (bands) | 128 ch, 4 stations | 128 ch, 10 | 128 ch, 32 | 1024 ch, 10 |
|---|---|---|---|---|---|
| 1 | 1 (2) | 0.0331 (0.90) | 0.0914 (0.99) | 0.365 (1.07) | 0.104 (0.87) |
| 1 | 4 (8) | 0.123 (0.89) | 0.338 (0.95) | 1.23 (0.94) | 0.391 (0.80) |
| 1 | 32 (64) | 0.997 (0.90) | 2.77 (0.96) | 10.7 (1.03) | 3.36 (0.87) |
| 2 | 1 (2) | 0.0338 (0.91) | 0.0949 (0.98) | 0.372 (1.05) | 0.104 (0.85) |
| 2 | 4 (8) | 0.126 (0.90) | 0.344 (0.95) | 1.26 (0.93) | 0.399 (0.81) |
| 2 | 32 (64) | 1.04 (0.91) | 2.86 (0.92) | 11.0 (0.98) | 3.55 (0.86) |
| 4 | 1 (2) | 0.0342 (0.90) | 0.0941 (0.94) | 0.376 (1.01) | 0.105 (0.85) |
| 4 | 4 (8) | 0.127 (0.88) | 0.349 (0.86) | 1.27 (0.86) | 0.404 (0.81) |
| 4 | 32 (64) | 1.09 (0.89) | 3.00 (0.83) | 11.5 (0.84) | 4.01 (0.87) |

- **The staged path is the fastest chain in 32 of the 36 shapes**, by 1-20% (9-12% at 4 stations,
  13-20% at 1024 channels).  It is behind only at 32 stations with 1 subband (by 7%, 5% and 1% at 1,
  2 and 4 bits) and with 32 subbands at 1 bit (3%), each time behind writing the layout + tensor.
  The split by stage shows why (as on the RTX PRO 6000 below): the staged XMAC alone takes
  1.1-1.9x the tensor XMAC kernel alone here (whose layout the station kernel writes, its cost
  in the station time) (1.94x at 32 stations with 1 subband), and the
  FP16 station kernel saves least with one subband (0.83 of the layout-writing one's time, against
  0.4-0.6 at 1024 channels), so where the XMAC's share is largest the saving no longer covers it.
- **More bits favour the staged path**: the layout path's station kernel slows more with the
  4-bit decode (writing the layout + tensor: 1.32 ms/ms at 1 bit, 1.48 at 4, 4 subbands at 32
  stations; FP16 spectra + staged: 1.23 and 1.27), so at 4 bits the staged path leads at 32
  stations by 14-16% with 4 or 32 subbands.
- **The band count changes little else**: per band of data the times scale nearly linearly from
  1 to 32 subbands, and the staged path's lead holds within a few percent across them.
- **Against gpudifx today: 5.4-13.8x** (6-8x up to 10 stations, 10-14x at 32, 5.4-6.9x at 1024
  channels).

### Wider shapes, bits and subbands: RTX PRO 6000 Blackwell Server Edition (OzSTAR, gina18, 2026-11-20)

`run-wide.sh` on sm_120 (96 GB, 188 SMs, 450 W of 600 W allowed), so with 64 stations at 128
channels and 32 at 1024 as well (54 shapes).  The first attempt (job 21263199, commit 880996a)
failed every 1024-channel shape: the tensor-layout tile was planned for 227 KB of shared memory a
block, taking sm_120 for a datacentre part like sm_90 ("sm_90 and up"), but it has 99 KB, like
sm_86/89; and the register-capped variants assumed 2048 threads an SM, where it holds 1536
(ptxas's "minnctapersm out of range" warnings).  Both are now listed per architecture
(`fused_inst.cu`), and fusebench skips a tensor-layout variant whose shared memory does not fit.
`results-local/gina18-wide-20261120-140632` (job 21267207, commit 92a16a2), all 54 runs passing;
the 128-channel shapes repeat the first attempt's to within about 1%.

FP16 spectra + staged, ms per ms of data, and in brackets its time over the best other chain:

| bits | subbands (bands) | 128 ch, 4 stations | 128 ch, 10 | 128 ch, 32 | 128 ch, 64 | 1024 ch, 10 | 1024 ch, 32 |
|---|---|---|---|---|---|---|---|
| 1 | 1 (2) | 0.0055 (0.99) | 0.0148 (1.18) | 0.0523 (1.09) | 0.116 (1.06) | 0.0164 (0.82) | 0.0640 (0.80) |
| 1 | 4 (8) | 0.0172 (0.93) | 0.0471 (0.86) | 0.173 (0.97) | 0.436 (1.07) | 0.0549 (0.79) | 0.243 (1.01) |
| 1 | 32 (64) | 0.139 (0.86) | 0.383 (0.78) | 1.52 (0.94) | 3.54 (0.98) | 0.477 (0.77) | 1.90 (0.78) |
| 2 | 1 (2) | 0.0052 (0.96) | 0.0142 (1.18) | 0.0506 (1.08) | 0.114 (1.10) | 0.0158 (0.79) | 0.0626 (0.80) |
| 2 | 4 (8) | 0.0169 (0.94) | 0.0464 (0.85) | 0.176 (0.98) | 0.440 (1.05) | 0.0562 (0.79) | 0.249 (1.01) |
| 2 | 32 (64) | 0.139 (0.81) | 0.388 (0.76) | 1.55 (0.92) | 3.62 (0.96) | 0.507 (0.79) | 1.99 (0.75) |
| 4 | 1 (2) | 0.0052 (0.97) | 0.0150 (1.16) | 0.0519 (1.09) | 0.117 (1.12) | 0.0159 (0.79) | 0.0640 (0.85) |
| 4 | 4 (8) | 0.0175 (0.95) | 0.0473 (0.85) | 0.178 (0.98) | 0.443 (1.04) | 0.0564 (0.79) | 0.250 (0.99) |
| 4 | 32 (64) | 0.147 (0.67) | 0.403 (0.71) | 1.60 (0.81) | 3.69 (0.84) | 0.544 (0.81) | 2.13 (0.71) |

- **The staged path is the fastest chain in 40 of the 54 shapes**, by up to 33%, and 15-29% at
  1024 channels (tied, 0.99-1.01, with 4 subbands at 32 stations).  It is behind with 1 subband
  at 128 channels from 10 stations (by 6-18%, most at 10 stations) and with 4 subbands at 64
  stations (4-7%), each time behind writing the layout + tensor.
- **Why, split by stage** (2 bits; best variant of each station kernel, and each XMAC kernel
  alone).  The layout path needs no transpose: its station kernel writes the tensor layout
  itself, so the rearrangement's cost is in its station time, and its XMAC kernel reads a layout
  ready for it.  The FP16-spectra station kernel takes 0.47-0.85 of the layout-writing one's
  time, least saving with 1 subband at 128 channels (0.79-0.85) and most at 1024 channels
  (0.47-0.55); the staged XMAC, which does the rearrangement in its own copy, takes 1.07-2.75x
  the tensor kernel alone (1.07-1.18x at 4 stations, 1.55-2.4x from 10 stations with 1 subband,
  1.8x at 64 stations).  The stages add up to the chains (1 subband, 10 stations: 0.192 + 0.051
  ms for the layout path, 0.159 + 0.121 for the staged one, against 0.252 and 0.284 measured with
  the phase and setup kernels).  So the staged path loses where the XMAC's share is large and the
  station saving small.  Against the transpose route (FP32 spectra, transpose, tensor) the staged
  XMAC is the faster: there the transpose alone is 0.246 ms.  The 2070 shows the same split (above), so it is
  the staged XMAC, not sm_120: closing its gap to the tensor XMAC is the lever (Open items,
  "Large N").
- **More bits favour the staged path, as on the 2070**: at 4 bits with 32 subbands it leads by
  16-33% at 128 channels.
- **Against gpudifx today: 7.2-18.4x** (7-10x up to 10 stations and at 1024 channels, 13-16x at
  32 stations, 17-18x at 64).
- Not comparable in absolute terms with the 2070 or the A100 sections; the ratios are.

### The big matrix, 108 shapes: RTX 2070 SUPER (150 W + 1500 MHz) and A100-SXM4-80GB (gina10), 2026-11-20

`run-matrix.sh` (new): the whole chain for 1, 2 and 4 bits x 1, 8 and 64 dual-pol subbands x 32,
512 and 4096 channels x 4, 16, 64 and 256 stations, looking for corner cases.  Each shape gets the
most windows that fit the GPU's memory, about 2e10 complex multiply-accumulates and 1e8 spectral
values (at most 10000, at least 16), unrounded, so most counts are not a multiple of 16; shapes
that do not fit at 16 windows are skipped.  Commit 3ff7185.  Summaries (one row per shape) in
`results/ar313-adeller-matrix-20261120-144918.csv` and `results/gina10-matrix-20261120-145410.csv`;
the logs and every timing and check (`raw.csv`) in `results-local/` under the same names.  The
A100 took 846 s.

| | 2070 | A100 |
|---|---|---|
| shapes run, every check passing | 79 | 92 |
| skipped (too big at 16 windows: mostly the visibilities) | 27 | 12 |
| refused (a window over the lean decode's 32 VDIF frames) | 2 | 4 |
| the check closest to its limit (rms / limit) | 0.40 | 0.42 |
| staged chain the fastest | 60 of 79 | 62 of 92 |
| staged chain over the best other chain | 0.48-1.43 | 0.66-1.32 |
| staged chain over the layout path | 0.24-1.21 | 0.31-1.32 |
| gpudifx today over the staged chain | 3.5-30.8x | 2.2-28.7x |

- **No wrong results anywhere.**  The closest any check came to its limit was 42% (the station
  kernel against the double-precision DFT); every XMAC check had more room.
- **One limitation found: the lean decode's 32-frame window.**  The window descriptor holds a
  32-bit mask of the frames a window spans, so a window may span at most 32 VDIF frames.  With 128
  bands in a frame at 2 or 4 bits, an 8192-point window spans more (1 bit fits), and fusebench
  refuses the shape (`run-matrix.sh` records it as "unsupported").  Open items.
- **Where the staged chain loses, it is the XMAC's share again**, as in the wider shapes: at 64 and
  256 stations (fastest in 9 of 18 and 6 of 12 shapes on the 2070, 10 of 24 and 12 of 18 on the
  A100), and at 32 channels on the A100 (13 of 36), where the FFT is short, the station saving
  small, and the staged XMAC up to 2.2x the tensor XMAC kernel (0.8-2.2x).  At 512 and 4096 channels it is the
  fastest in 25 of 27 and 15 of 16 shapes on the 2070, 28 of 33 and 21 of 23 on the A100, up to 4x
  ahead of the layout path at 4096 channels (whose station kernel's tile is largest there).
- **Very short subints with many baselines** (20-23 windows at 64-256 stations, where the
  visibilities outweigh the data): the XMAC's per-subint output (zeroing, window chunks' atomics,
  the stores) dominates, the tensor XMAC is 1.3-4.3x slower than the staged one, and the tiled
  FP32 XMAC beats both (the chain by 1.3-1.4x).  Not a regime DiFX gets into: even multiple
  phase centres at L band (8k channels in 32 MHz, 40 ms subints) have about 160 windows a subint.

### Stations with different band layouts: `xmacbench --messy` (RTX 2070 SUPER, 150 W + 1500 MHz, 2026-11-20)

The staged XMAC now takes a map (`XmacMap`, `xmac.cuh`): per correlated frequency and receiver,
where that receiver's FP16 spectra start and their window stride, and which station it is.  So
the stations need not share one layout: different bandwidths and numbers of bands, and stations
missing a frequency.  The regular layout is the special case `xmacRegularMap`, so it is one
kernel, not two.  Each row's place is a 32-bit offset (in `__half2`) and a 32-bit stride: as a
64-bit pointer and stride each, the KS 2 kernel spilled at its 128 registers and was 12-20%
slower at 10 stations; now the registers are as before (119, 128, 63, 64; no spills) and the regular
case is within 1-3% of before (4-64 stations, 128 and 1024 channels).  The spectra may span at
most 2^32 `__half2` (16 GiB), checked.

`xmacbench --messy`: the correlated frequencies are F dual-pol bands of the narrowest bandwidth, C
channels each.  Half the stations have bands of 2 frequencies (twice the bandwidth), the rest of 1,
and one has bands of 3 (a bandwidth and channel count that are not powers of two) covering
frequencies 1-6 of 8 only, so it lacks two.  Every band is whole correlated frequencies (no band
straddles one's edge: the 62.5 MHz case, where zoom bands must be stitched, is still open).
gpudifx's XMAC is configured as GPUCore would (per baseline each station's spectra and strides; per
frequency, baseline and product a band index in units of C channels, -1 where a baseline lacks
the frequency), and both are checked against a double-precision reference, with the absent
baselines checked to be exactly zero.  Every shape passes:

| stations | frequencies x channels | windows | gpudifx (tables) | staged (map) | staged, regular layout of the same size |
|---|---|---|---|---|---|
| 3 | 8 x 128 | 2001 | 0.423 | 0.195 | 0.199 |
| 4 | 8 x 128 | 4000 | 1.79 | 0.417 | 0.407 |
| 10 | 8 x 128 | 2000 | 6.38 | 0.810 | 0.803 |
| 10 | 6 x 512 | 501 | 4.68 | 0.757 | 0.779 |
| 32 | 8 x 128 | 1000 | 38.5 | 2.01 | 1.93 |
| 64 | 8 x 128 | 500 | 84.0 | 3.84 | 3.52 |
| 64 | 14 x 32 | 777 | 68.3 | 2.19 | 2.21 |

(ms per subint.)  The map costs nothing measurable: the messy layout is within 4% of the regular one
of the same size, 9% at 64 stations with 8 frequencies (the stations' wider bands spread a block's
reads), while it does less work (the station missing two frequencies).

### Window design

All the designs here overlap neighbouring windows over exactly their tapers (with 12.5% tapers,
consecutive windows start 0.875 FFT lengths apart). Three taper shapes, with x running from 0
to 1 across the taper:

- **Tukey**, the raised cosine 0.5 (1 - cos πx).
- **Sine**, sin(πx/2).
- **Vorbis**, sin(π/2 · raised cosine). This is the taper of the window used in Ogg Vorbis,
  and the `--wola` default.

![Three taper shapes](plots/tapers.png)

Where two windows overlap, the cross-power weights each sample by the sum of the squares of the
two tapers.
- **Tukey tapers add up to one in amplitude, not in power**, so that weight dips to 0.5 in the
  middle of each overlap. This costs continuum sensitivity: 0.6% at 12.5% tapers, 1.4% at 25%.
  Shorter tapers lose less, but no Tukey width loses nothing, because the loss comes from the
  shape.
- **Sine and Vorbis tapers add up to one in power**, so every sample has weight 1 and there is
  no loss at any taper length.
- **The Vorbis taper is also smooth at both ends**, like the Tukey taper, so its far sidelobes
  fall off as fast as Tukey's. The sine taper's sidelobes fall off more slowly.

![Tapered FFT windows](plots/windows.png)

![Sensitivity and leakage against taper length](plots/tradeoffs.png)

Longer tapers cut the leakage further and cost more FFTs, 1 / (1 - taper). At 12.5% the near
leakage (beyond 5 channels) is much the same for all three shapes, -23 to -27 dB, and the far
leakage (beyond 20 channels) is -59 to -60 dB for Tukey and Vorbis, against -47 dB for sine.
Between about 18% and 25%, Tukey's near leakage drops faster than the others' (-44 dB at 25%,
against -34 dB for sine and Vorbis), but Tukey then loses 1.4% of sensitivity.

![A channel's response to a narrow line](plots/response.png)

![Continuum sensitivity against residual delay](plots/sensitivity.png)

Window shape alone sets how much correlation survives a residual delay; overlap sets the
sensitivity at zero delay. With 12.5% tapers each shape beats rectangular windows over a range
of residual delays (256-point FFT): Tukey from 2 to 43 samples, Vorbis from 1 to 52, sine from
1 to 69. The sine taper keeps the most correlation.

`window_response.py` gives these numbers for a 256-point FFT (continuum sensitivity relative to
rectangular windows without overlap; white noise; all channels). The spectral figures do not
depend on the FFT length.

| design | FFTs | sensitivity, delay 0 / 4 / 16 samples | leakage beyond 5 / 20 channels |
|---|---|---|---|
| rectangular, no overlap (mpifxcorr now) | 1.000 | 1.000 / 0.984 / 0.938 | -16.9 / -23.0 dB |
| Tukey 12.5% | 1.143 | 0.994 / 0.991 / 0.955 | -25.0 / -60.4 dB |
| sine 12.5% | 1.143 | 1.000 / 0.997 / 0.965 | -26.6 / -47.2 dB |
| Vorbis 12.5% (`--wola`) | 1.143 | 1.000 / 0.997 / 0.957 | -23.4 / -59.2 dB |
| Tukey 25% | 1.333 | 0.986 / 0.984 / 0.960 | -44.2 / -71.2 dB |
| sine 25% | 1.333 | 1.000 / 0.998 / 0.977 | -34.4 / -52.8 dB |
| Vorbis 25% | 1.333 | 1.000 / 0.998 / 0.970 | -34.2 / -70.7 dB |
| Tukey 12.5%, hop 0.75 | 1.333 | 0.972 / 0.969 / 0.934 | -25.0 / -60.4 dB |
| Hann, 50% overlap | 2.000 | 0.973 / 0.972 / 0.949 | -53.5 / -83.7 dB |

The "hop 0.75" row places Tukey 12.5% tapers 0.75 FFT lengths apart, so the flat parts of the
overlaps are counted twice (weights 1 to 2). It costs more and does worse on every measure here.

**Choosing.** For no loss of continuum sensitivity at any taper length, use the Vorbis taper.
It keeps the Tukey taper's far-sidelobe suppression. At 12.5% it costs 1.143x the FFTs
(`--wola`). Going to 25% improves the near leakage by about 11 dB and the far
leakage by about 11 dB, for 17% more windows: 10-13% more fused-kernel time on the 2070 (the
"hop 0.75" rows in the results above have the same window count). The window is a table of
weights, so the shape costs nothing; only the taper length (the number of windows) does.
All of these are `fusebench` options (`--taper`, `--overlap`, `--window tukey|sine|vorbis`).

### Open items

- **One XMAC path: the staged one, done (2026-11-20).**  With the mma.sync version FP16 spectra
  + the staged XMAC is the fastest chain from 4 to 32 stations on both GPUs and at 1024 channels,
  and works at any station count ("The mma.sync staged XMAC on the A100", and the 2070 table
  before it); the big matrix found no shape that needed the others.  Retired: the transpose, the
  station kernel's tensor-layout output, the tensor XMAC from that layout, the tiled FP32 XMAC and
  the staged XMAC's WMMA version.  gpudifx's XMAC stays as the reference, and fusebench's tight
  XMAC check now runs it on the staged path's own FP16 spectra.  The results sections above keep
  their measurements of the retired paths, as the record of the comparison (and
  `docs/xmac-paths.md`).
- **Large N (about 100 stations and up): an XMAC-focused optimisation.**  There the staged path
  trailed the (now retired) layout path by 25% in the chain on the A100 (6% at 64 stations), and the staged
  XMAC alone is 1.75-2.5x the tensor XMAC from its layout: the XMAC, not the station processing,
  dominates, and each receiver is re-read once per super-block it is paired with (4 times at 100
  stations).  Candidates: larger super-blocks where the shared memory allows (KS 8 on the A100,
  half the re-reads, with the items split differently among the warps for the registers); more
  output per stage load (bigger tiles per warp, more blocks an SM); and the XMAC literature for
  large N (e.g. the tensor-core correlators built for LOFAR-class arrays).  Not needed for
  DiFX's usual arrays.  The same gap shows below 100 stations wherever the XMAC's share is
  large: on the RTX PRO 6000 and the 2070 the staged XMAC takes 1.1-2.75x the tensor XMAC kernel
  alone at every shape (the layout path pays for its rearrangement in the station kernel instead,
  less than that gap where the FP16 station kernel saves least), and with 1 subband at 128 channels (where the FP16 station kernel saves
  least) that already loses the chain from 10 stations ("Wider shapes", both GPUs).
- **The lean decode's 32-frame window** (found by the big matrix): a window may span at most 32
  VDIF frames (`leanDecodeFits`; the window descriptor's valid-frame mask is 32 bits), which 2- and
  4-bit data with 128 bands in a frame exceed at 8192-point windows.  A 64-bit mask (the descriptor
  is an `int4`: the parity bit could share a word), or a slower path for the rare long window.  The
  XMAC outputs need the lean decode, so such shapes cannot run the whole chain today.
- **A wider test matrix**: 1, 2 and 4 bits x 1, 4 and 32 dual-polarisation subbands x 4-32
  stations at 128 and 1024 channels done on the 2070, and with 64 stations at 128 channels and
  32 at 1024 on the RTX PRO 6000 ("Wider shapes, bits and subbands", `run-wide.sh`); the A100
  still to run.  Still open: channel counts from 64 to 16384 and not powers of two, single
  polarisation, 8 bits (not compiled for the XMAC outputs), and one subband from 10 stations
  (the 2070 at 32, the RTX PRO 6000 from 10), where the layout path still leads by 1-18%.
- **The end-to-end kernel is now arithmetic-bound on the A100** (FMA pipe 66%, special-function
  unit 28%): what is left is mostly the FFT itself, the integer multiply-adds of the indexing,
  a sincos per sample for the fringe phase and one per channel for the fractional-sample
  correction.  cuFFTDx's shared-memory exchange (bank conflicts; the main stall on the 2070).
- **The XMAC with the station processing**: the end-to-end kernel writing the tensor layout is
  done and measured on both GPUs ("The whole chain" above): it wins at 128 channels and loses at
  1024 (occupancy).  Next the XMAC doing the reshuffle itself (next item), then the two stages
  overlapped in chunks, against the number of stations.  Chunking all antennas a range of windows at a time keeps a chunk's
  spectra in L2 for the XMAC, but the XMAC must then read and write its accumulators once per
  chunk. With N stations, B bands, C channels and W windows per
  chunk, accumulator traffic over spectrum traffic is about 2N / W (FP16 spectra, FP32
  accumulators), and with W set by the L2 size, about 8 N^2 B C / L2: 1 at N ~ 70 for B C = 1024
  on the A100 (40 MB), at N ~ 17 for B C = 16384. Beyond that chunking costs more than it saves,
  unless the XMAC keeps its accumulators on chip across chunks (a persistent producer/consumer
  design).  The tensor XMAC at small N: fewer wasted products (a 16 x 16 tile holds at most 8
  stations' pairs), autocorrelations in it.
- **DiFX's flexibility in the tensor XMAC**: zoom bands (a channel range of a recorded band),
  mixed bandwidths, channel counts that are not powers of two, stations that do not record every
  frequency, single-polarisation stations, baselines not correlated, channel averaging.  The
  tensor kernel itself need not see any of it: the channels are its batch dimension (each one an
  independent X X^H), so their number is free, and the tensor cores only constrain the windows
  (blocks of 16, padded with zeros) and the receivers (16-row tiles, the spare rows discarded).
  Done for the staged XMAC's input: its map ("Stations with different band layouts", `xmacbench
  --messy`: different bandwidths and band counts, stations missing a frequency).  Still open: bands
  that straddle a correlated frequency's edge (ALMA's 62.5 MHz against 64 MHz: zoom bands per
  overlapping piece, stitched into synthetic channels), baselines not correlated, channel
  averaging, and the station side (per-station FFT lengths in fusebench, non-power-of-two FFTs).
  The rest can be two tables, like the band indexes and result offsets gpudifx's XMAC takes now:
  an input map for whatever writes the tensor layout (per correlated frequency and receiver: the
  station's band and channel offset; a zoom band and a narrower band at another station are both
  a channel range of a spectrum; stations without the frequency have no row), and an output map
  for the XMAC's store (per frequency and receiver pair: the result offset, or none for a
  baseline not wanted, whose products are computed anyway and dropped).  Channel averaging in the
  store or a pass after it.  To show it in `xmacbench`: a "messy" configuration (per-frequency
  station subsets, a zoom band at a channel offset, a subset of baselines, a non-power-of-two
  channel count such as 1000) against the reference, and what the tables cost.  Still open:
  FP16's range (65504): a strong tone in a long FFT can approach it, so the writer should
  scale the spectra (free in the fractional-sample rotation), with a test with a strong tone; BF16
  has the range but only 8 bits of mantissa.
- **Pulsar binning and multiple phase centres**: each in its own directory, as a backlog with the
  approaches to compare ([../pulsar-binning](../pulsar-binning/README.md),
  [../phase-centres](../phase-centres/README.md)).  Both build on the staged XMAC; multiple phase
  centres also on the two-stage channelisation (fine channels).
- **FFT lengths that are not powers of two.**  The main case is the EHT: ALMA's 62.5 MHz channels
  need FFTs of N x 125 points to match the channel width of the other stations, which record
  power-of-two bandwidths (so this is also mixed-bandwidth correlation, below).  cuFFTDx (MathDx
  25.6) does some lengths directly and others only with Bluestein's algorithm, which needs a
  workspace in global memory and several FFTs of a longer power-of-two length, and is capped at
  a Bluestein length of 16384:

  | points | sm_80 and sm_75 |
  |---|---|
  | 125, 500, 1000, 8000 | direct (25, 50, 10 and 20 elements per thread: 50 is a lot of registers) |
  | 250, 2000, 4000 | Bluestein |
  | 16000 | not available (its Bluestein length, 32768, is over the cap) |

  (Other lengths: 400 direct; 640, 800, 1536, 1800, 3200, 6000 Bluestein.)  The fused kernel
  rejects the workspace for now, and in places assumes the elements per thread divide the length
  (true for the direct ones) and a multiple of 32 (`decode_to`).  To do: build and time the
  N x 125 lengths against the neighbouring powers of two, direct and Bluestein; cuFFT (any
  length, mixed radix) as the fallback in the 2-kernel pipeline where Bluestein costs too much
  or is not available; and see whether other N x 125 decompositions (e.g. 2000 as a 2-stage
  16 x 125) do better.
- **Mixed-bandwidth correlation** (e.g. 32 MHz bands at one station, 128 MHz at another; ALMA's
  62.5 MHz against 64 MHz or more elsewhere in the EHT): the channel width, so the FFT duration
  and the window count, are the same at both, but the FFT lengths differ (4 times longer at the
  128 MHz station), so each station gets its own fused-kernel size (it is launched per station
  already), and the wide station's spectrum supplies a quarter of its channels to each of the
  narrow station's bands (the input map above).  `fusebench` assumes
  one shape for every station: per-station shapes, then the two stages together with mixed
  bandwidths.  Mixed real and complex sampling works the same way.
- **The reshuffle into the tensor layout in the XMAC**: done, the staged XMAC (above, and
  "One XMAC path").
- **The tiled XMAC's cost against the subint length** (above): 8 times the time for 4 times
  the windows at 16 stations on the 2070.  Sweep the windows at fixed stations, profile the
  long case (DRAM traffic, TLB: the spectra are 2.4 GiB at 10000 windows), fix it or explain it,
  and redo the tiled-against-tensor comparisons at a realistic subint length.
- **More channel counts for the whole chain** (`fusebench --xmac`, `run-xmac.sh`): so far only
  128 and 1024.  Writing the tensor layout costs the end-to-end kernel about 20% at 128 channels
  but 64% at 1024 on both GPUs (the tile for W windows grows with the channel count: on the 2070
  W falls and the stores shrink, on the A100 occupancy falls), and that cost decides whether the
  tensor path beats the tiled FP32 XMAC.  Sweep 64 to 16384 channels on both GPUs to find where the direct layout stops
  paying, and how that moves with the GPU's shared memory.  This bears on two-stage
  channelisation (next item): a second stage of short FFTs could write the layout with the
  small tiles of a short FFT whatever the total channel count, so the sweep also says at what
  channel count a two-stage design gains on the XMAC side as well as on the FFT side.  But the
  two stages need a DRAM round trip between them (stage 2 needs 16 x its length of each coarse
  channel's samples, which for all coarse channels is the same large tile), about as much
  traffic as the transpose: two-stage moves that cost rather than removing it, so it should rest
  on its other gains (shorter FFTs, the fractional-sample correction's accuracy).
- **Two-stage channelisation for high spectral resolution**: analysed, planned and implemented
  (PFB first stage; `twostagebench`, `run-twostage.sh`): see [docs/two-stage.md](docs/two-stage.md),
  "Implementation and first results" and "At about 8k channels".
  - **The policy** the results point to: a single stage (WOLA) whenever the channel count fits one
    cuFFTDx FFT (per subband, so narrower subbands reach finer resolution); as soon as it does not
    (or a large delay rate makes the fractional-sample loss of one long window too large, e.g.
    spacecraft), a PFB first stage of a moderate number of coarse channels (32) and an FFT second
    stage.  Two stages cost ~2x one per ms of data whatever the final channel count; their loss is
    < 0.1% except with 4 taps at O = 4/3.
  - **Next:** the A100 and RTX PRO 6000; the decode overlap between stage-1 blocks; the whole chain
    with the staged XMAC; and the setup in the kernel.  The setup is now a separate kernel, one
    thread per (antenna, coarse window, band), turning a host-made fractional delay per coarse
    window into a fixed-point phase offset and slope (FP64, 16 bytes written and read back per
    window and band), with an integer sample index per coarse window uploaded as well.  Per coarse
    window (16-96 samples a band) that metadata is larger than the 2-bit data itself.  In the kernel: pass the
    delay polynomial per (antenna, subint) (mpifxcorr's interpolator) and the clock and LO terms per
    (antenna, band); stage 1's thread 0 evaluates it in FP64 once per block (as it already makes the
    block's fringe constants), and each window's integer shift and fractional delay follow from the
    block's value and the local rate in FP32 (a block spans a few thousand samples, so the change is
    a few hundredths of a sample).  See the root README's "For DiFX proper" for reporting the
    losses at startup.  The original note: (say above 512 channels): a coarse
  first stage (e.g. 32 channels), then fringe rotation and fractional-sample correction, then a
  second stage of 32 x 32 channels. The fractional-sample correction stays accurate (it becomes
  inaccurate with very long windows and high fringe rates), and two short FFTs may be faster
  than one long one (see the channel-count table), and short second-stage FFTs keep the
  tensor-layout tile small (see the item above). The first stage is the question: a WOLA FFT
  risks artefacts at the coarse channel edges, a PFB costs complexity and computation.
- Two kernels with WOLA through a series buffer (each sample decoded, rotated and stored once,
  as `--layout series`) rather than a buffer per window, which repeats the 12.5% overlap. That
  needs the fringe phase to be continuous from window to window.
- Phases by recurrence (one sincos per thread, then a complex multiply per element) instead of a
  sincos per element: tried for the fringe phase and dropped (no gain at 8 samples a thread, see
  "The lean decode"); might pay for the fractional-sample correction at large elements per thread.
- Complex packed data, phase cal, and CODIF in the end-to-end kernel.
- Complex data (`--complex`).
- The effect inside mpifxcorr, where the FFT and rotation overlap with transfers and the XMAC.
