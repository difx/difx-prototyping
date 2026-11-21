# The lean decode

_Moved from `fft-fusion/README.md` in the 2026-11-21 reorganisation, unchanged but for the heading levels. `results-local/` paths are in `fft-fusion/` (runs of `run-xmac.sh` and `run-staged.sh`: `xmac/`). Section names in quotes refer to these docs: see [the findings index](../FINDINGS.md)._

## The lean decode: RTX 2070 SUPER, 215 W (before 2026-11-18)

The A100 profiles ("From the packed data: A100-SXM4-80GB (OzSTAR, gina10)" in
[station-kernel.md](station-kernel.md)) put two-thirds of the end-to-end kernel's instructions in the decode and
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

## The lean decode: A100-SXM4-80GB (OzSTAR, gina15)

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
