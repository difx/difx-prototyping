# The whole chain: station processing + XMAC

_Moved from `fft-fusion/README.md` in the 2026-11-21 reorganisation, unchanged but for the heading levels. `results-local/` paths are in `fft-fusion/` (runs of `run-xmac.sh` and `run-staged.sh`: `xmac/`). Section names in quotes refer to these docs: see [the findings index](../FINDINGS.md)._

## The whole chain, station processing + XMAC: RTX 2070 SUPER, 215 W (before 2026-11-18)

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
  tiled column of the XMAC table ([xmac-stage.md](xmac-stage.md)) (10000 windows up to 16 stations) is probably pessimistic,
  and where the tensor path overtakes the tiled one depends on the windows per subint.  Not yet
  understood (open items).

## The whole chain, station processing + XMAC: A100-SXM4-80GB (OzSTAR, gina4)

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
