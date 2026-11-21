# difx-prototyping

Quick prototyping code for performance improvements (or correctness tests) for DiFX.

Prototypes and standalone benchmarks for DiFX development are kept here, out of the main
[difx](https://github.com/difx/difx) repository. Code here is for trying ideas out and measuring
them. It is not built or installed with DiFX, and nothing in difx depends on it. When an idea
works, it is reimplemented in difx and the prototype stays here as the record of how it was
evaluated.

## Layout

| directory | what |
|---|---|
| `common/` | the shared GPU code (VDIF decode, the station kernels, the two-stage channeliser, the XMAC, the benchmark helpers) and scripts (OzSTAR jobs, binary lookup, result copying); see "The shared code" below |
| [`fft-fusion/`](fft-fusion/README.md) | the station stage: the FFT and the fractional-sample correction fused into one cuFFTDx kernel, from the packed data, against gpudifx today; FP16 FFT input; windowed overlap-add front end; the whole chain over many shapes |
| [`xmac/`](xmac/README.md) | the cross-multiply-accumulate: gpudifx's XMAC against the staged tensor-core XMAC, alone and after the station stage; stations with different band layouts |
| [`two-stage/`](two-stage/README.md) | two-stage channelisation (PFB, then an FFT per coarse channel) for resolutions one cuFFTDx FFT cannot reach |
| [`phase-centres/`](phase-centres/README.md) | GPU correlation to many phase centres at once (not started) |
| [`pulsar-binning/`](pulsar-binning/README.md) | GPU pulsar gating and binning in the XMAC (not started) |

Each sub-project has:
- `README.md`: what it is, how to build and run it, its code, its checks.  The current state.
- `FINDINGS.md`: what has been learned, one short entry each, linking to `docs/` for the
  measurements and analysis.  Done work.
- `BACKLOG.md`: what is in progress and what is still to do, with the reasoning.

[BACKLOG.md](BACKLOG.md) here is the one-line index of every sub-project's backlog, and holds
what should be carried into DiFX itself.

## Building

One Makefile, in this directory, builds every sub-project's binaries.  With a gpudifx `setup.bash`
sourced, `make -j8` is all that is needed. The Makefile takes CUDA from `CUDAROOT` and the GPU
architecture from the `-arch=sm_NN` in `NVCCFLAGS`. Without them it falls back to the `nvcc` on
`PATH` and the GPU that `nvidia-smi` reports. It prints what it chose at the start of every build.

cuFFTDx is part of NVIDIA's header-only MathDx package. It is not in the CUDA toolkit, and
NVIDIA's licence does not let us commit it here. On the first build the Makefile downloads the
`nvidia-mathdx` 25.6.0 wheel from PyPI, checks its sha256 and unpacks it into
`mathdx-25.6.0/` (git-ignored). On a machine whose build nodes have no network access, run
`make mathdx` first on one that does (an OzSTAR login node). To use a MathDx you already have,
set `MATHDX` to the directory that holds `include/cufftdx.hpp` and `external/cutlass`. Tested
with MathDx 25.6.0 (cuFFTDx 1.5.0) and CUDA 12.8 and 13.2.

```bash
source <gpudifx>/setup.bash
make -j8                                                     # as setup.bash says: every binary
make -j8 SM=80                                               # A100, whatever setup.bash says
make -j8 MATHDX=<mathdx dir> CUDA_HOME=/usr/local/cuda-13.2  # everything given explicitly
make -j8 xmacbench twostagebench                             # only some (fusebench is the slow one)
make -j8 SIZES="256 2048" fusebench                          # only some FFT lengths in the fused kernel
```

Each GPU architecture builds in its own `build-sm<SM>/`, so builds for different GPUs never
touch each other.  `fft-fusion/fusebench`, `xmac/xmacbench` and `two-stage/twostagebench` are
links to the last build's binaries, for running by hand; the `run-*.sh` scripts take the
binaries built for the GPU they find (`common/scripts/bench-bin.sh`).  `fusebench`'s fused
kernel is compiled once per FFT length (`SIZES`, default 64 to 16384), in all its variants:
about 4 minutes with `-j8` on a desktop, 25 on an OzSTAR node.  Every benchmark exits non-zero
if a check fails.

## Results

Each `run-*.sh` script writes its logs to its sub-project's `results-local/<host>-<what>-<stamp>/`
(git-ignored), with `system.txt` (the GPU, its power limit and clock lock) and a summary table.
`results/` in a sub-project (in the repository) holds summaries only, e.g. one CSV row per shape
for `fft-fusion/run-matrix.sh`.  The `docs/` pages name the `results-local/` directory behind each
table.

## OzSTAR

On a login node, in a clone of this repository: `make mathdx`, then

```bash
sbatch common/scripts/ozstar.sbatch                    # the default set (below)
sbatch --export=ALL,SCRIPTS="xmac/run-xmac.sh" common/scripts/ozstar.sbatch
sbatch --time=0:45:00 --export=ALL,SCRIPTS=two-stage/run-twostage.sh,NCHAN=8192,NWIN=128 common/scripts/ozstar.sbatch
sbatch common/scripts/ncu-ozstar.sbatch                # Nsight Compute profiles (--set full)
```

The job sources your gpudifx setup (`DIFX_SETUP`, default `$HOME/setup_difx.claude`: module
loads, then `setup.bash`), builds for the GPU it gets (`TARGETS` to build only some binaries),
and runs the scripts in `SCRIPTS` (paths from the root; default `fft-fusion/run-bench.sh
fft-fusion/run-nchan.sh xmac/run-xmac.sh xmac/run-staged.sh`), each in its own sub-project.  Each
script's results are packed into a `.tgz` next to them.  At the end the job writes
`copy-results.<jobid>.sh` in the root: run it once the job has finished, and it copies the SLURM
log (`ozstar.<jobid>.log`) and the `.tgz` files with one `rsync` to the same places in
`RESULTS_DEST` (default ar313's checkout).  The headers of the two `.sbatch` files have the
details.

## Benchmark hosts

| host | GPU | notes |
|---|---|---|
| ar313 | RTX 2070 SUPER (sm_75, 8 GB) | power-capped and clock-locked, see below |
| OzSTAR gina nodes | A100-SXM4-80GB (sm_80) | most nodes |
| OzSTAR gina18 | RTX PRO 6000 Blackwell Server Edition (sm_120, 99 KB of shared memory a block) | |

**ar313's RTX 2070 SUPER is power-capped at 150 W from 2026-11-18** (default 215 W). At 215 W
and 185 W the 7-year-old FSP500 PSU trips during `xmacbench --nchan 1024 --nant 32 --nwin
1000`, which cut the power to the whole box twice on 2026-11-17. The cap is set at boot by
`gpu-powercap.service` (`/etc/sysconfig/gpu-powercap`; investigation and load tests in
`~/packages/claude/ar313-reset-traps/`). **No 2070 result from before 2026-11-18 compares directly
with one from 2026-11-18 on.** The 2070 sections headed "215 W (before 2026-11-18)" are kept for
their analysis; the baselines for new work are in "RTX 2070 SUPER at the 150 W cap"
([fft-fusion/docs/station-kernel.md](fft-fusion/docs/station-kernel.md)). The cap
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

## The shared code (`common/`)

| file | what |
|---|---|
| `common.cuh` | `Geometry` (the `[ant][window][band][fftchannels]` layout of mpifxcorr's buffers), error macros |
| `stages.cuh` | the stages either side of the FFT: `PreLoad<T, SERIES>` (FP32/FP16 storage, per-window or series layout, optional window), `withPre()` (the one place stage types are instantiated); `FracRotateFixed` |
| `decode.cuh` | decode, sideband sign and fixed-point fringe rotation from the packed data, as pre-FFT stages: `PreDecodeLean<NBIT>` (the default) and `PreDecode` (gpudifx's per-sample arithmetic, `--decode direct`); the per-window descriptor (`WinDesc`) |
| `gpudecode_copy.cuh` | gpudifx's decode helpers and tiled decode + fringe kernel, copied verbatim (the commit and the two changes are noted there) |
| `pipelines.cuh` | the station pipelines' interface: the unfused (gpudifx today) and fused launchers, their outputs |
| `unfused.cu` | the copied production kernel, cuFFT plans as `GPUMode` makes them, the kernel that materialises a pre-FFT stage, the generic rotation kernel, the fixed-point phase precompute, and for packed input the fringe phases and window descriptors (`launchPackedSetup`) |
| `fused_inst.cu` | the fused kernel, compiled per FFT length; each length registers its variants (FFTs per block, and cuFFTDx's elements per thread or half of it) at start-up; two outputs: FP32 spectra, and FP16 spectra in the same order (for the staged tensor XMAC) |
| `xmac.cuh`, `xmac.cu` | the XMAC stage: the shapes and layouts, gpudifx's XMAC (the reference) and the staged tensor XMAC from FP16 spectra, split by receiver, reading them through a map (any station layouts) |
| `xmac_copy.cuh` | gpudifx's XMAC kernel, copied verbatim, and its launch geometry as GPUCore sets it |
| `twostage.cuh`, `twostage1.cu`, `twostage2.cu` | two-stage channelisation: the PFB stage (decode once a block, fold, coarse FFT, the delay model and corrections in the kernel), the fine-FFT stage |
| `benchutil.cuh` | what every benchmark uses: random numbers, device buffers, the VDIF format and the host's own decode, timing |
| `scripts/` | `bench-bin.sh` (the binaries for this GPU), `ozstar.sbatch` and `ncu-ozstar.sbatch` (the OzSTAR jobs), `make-copy-script.sh` (their copy script) |

## Conventions

- Every benchmark checks its results (against gpudifx's kernels or a double-precision reference)
  before timing them; a run whose checks fail is not a result.
- Times are medians of repeated runs, in ms per subint and ms per ms of data; compare only within
  one GPU and one power/clock era (above).
- Measurements of retired code stay in the `docs/` pages, marked as such: they are the record of
  why it was retired.
