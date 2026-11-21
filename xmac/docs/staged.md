# The staged XMAC split by receiver, and its mma.sync version

_Moved from `fft-fusion/README.md` in the 2026-11-21 reorganisation, unchanged but for the heading levels. `results-local/` paths are in `fft-fusion/` (runs of `run-xmac.sh` and `run-staged.sh`: `xmac/`). Section names in quotes refer to these docs: see [the findings index](../FINDINGS.md)._

## The staged XMAC split by receiver: RTX 2070 SUPER at the 150 W cap (2026-11-18)

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

## The staged XMAC split by receiver: A100-SXM4-80GB (OzSTAR, gina302)

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

### Why the staged XMAC trails on the A100, and the mma.sync version

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

### The mma.sync staged XMAC on the A100 (OzSTAR, gina302)

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
