# Channel count, chunks and register caps

_Moved from `fft-fusion/README.md` in the 2026-11-21 reorganisation, unchanged but for the heading levels. `results-local/` paths are in `fft-fusion/` (runs of `run-xmac.sh` and `run-staged.sh`: `xmac/`). Section names in quotes refer to these docs: see [the findings index](../FINDINGS.md)._

## Channel count, chunks and register caps: RTX 2070 SUPER, 215 W (before 2026-11-18)

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

## Channel count, chunks and register caps: A100-SXM4-80GB (OzSTAR, gina4)

`fusebench-ozstar.sbatch`, SLURM job 21042592, commit 89b4dfb (the corrected uncapped kernel and
the concurrent chunks), all 30 runs and the channel-count sweep passing their checks; Nsight
Compute `ncu-ozstar.sbatch`, job 21038155 on gina304, commit d74abd3. The uncapped and 2-kernel
times agree with those of job 21016518 (gina10, [station-kernel.md](station-kernel.md)) within 0.1%, so that table stands. An
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
