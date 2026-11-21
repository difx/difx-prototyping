# Wider shapes and the big matrix

_Moved from `fft-fusion/README.md` in the 2026-11-21 reorganisation, unchanged but for the heading levels. `results-local/` paths are in `fft-fusion/` (runs of `run-xmac.sh` and `run-staged.sh`: `xmac/`). Section names in quotes refer to these docs: see [the findings index](../FINDINGS.md)._

## Wider shapes, bits and subbands: RTX 2070 SUPER at 150 W + 1500 MHz (2026-11-20)

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

## Wider shapes, bits and subbands: RTX PRO 6000 Blackwell Server Edition (OzSTAR, gina18, 2026-11-20)

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

## The big matrix, 108 shapes: RTX 2070 SUPER (150 W + 1500 MHz) and A100-SXM4-80GB (gina10), 2026-11-20

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
