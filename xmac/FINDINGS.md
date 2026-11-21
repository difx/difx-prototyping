# xmac: findings

Detail, by topic, in `docs/`: [xmac-paths.md](docs/xmac-paths.md) (every station + XMAC path,
its layouts, what it saves and costs), [xmac-stage.md](docs/xmac-stage.md) (the XMAC alone),
[whole-chain.md](docs/whole-chain.md) (station + XMAC), [staged.md](docs/staged.md) (the staged
XMAC split by receiver, and its mma.sync version), [messy.md](docs/messy.md) (stations with
different band layouts).  Hardware eras: [../README.md](../README.md#benchmark-hosts).

## The XMAC (2026-11-16 to 17)

- **Once the station stage is fast, gpudifx's XMAC is the most expensive stage.**  Register tiling
  gains 3-6x on it, the tensor cores another 2-13x (more the more stations).  ([xmac-stage.md](docs/xmac-stage.md))
- **The tensor cores' catch is the layout**: they want each receiver's 16 consecutive windows of a
  channel as a row, and the station stage makes channels fastest.  A separate transpose costs
  1.4-3.0x the tensor XMAC itself; writing the layout from the station kernel wins at 128 channels
  and loses at 1024 (occupancy).  ([xmac-paths.md](docs/xmac-paths.md), [whole-chain.md](docs/whole-chain.md))

## One XMAC path: the staged one (2026-11-20)

- **The staged XMAC** reads FP16 spectra in the station kernel's own order and rearranges them in
  shared memory, split by receiver (super-blocks of 16 KS receivers), with mma.sync (1.1-1.4x
  faster than WMMA from 10 stations up).  FP16 spectra + the staged XMAC is the fastest chain from
  4 to 32 stations on both GPUs and works at any station count.  ([staged.md](docs/staged.md))
- **Retired after the comparison** (the measurements stay in `docs/`): the transpose, the station
  kernel's tensor-layout output, the tensor XMAC from that layout, the tiled FP32 XMAC and the
  staged XMAC's WMMA version.  gpudifx's XMAC stays as the reference.  Retired 2026-11-20:

  > **One XMAC path: the staged one, done (2026-11-20).**  With the mma.sync version FP16 spectra
  > + the staged XMAC is the fastest chain from 4 to 32 stations on both GPUs and at 1024 channels,
  > and works at any station count ("The mma.sync staged XMAC on the A100", and the 2070 table
  > before it); the big matrix found no shape that needed the others.  Retired: the transpose, the
  > station kernel's tensor-layout output, the tensor XMAC from that layout, the tiled FP32 XMAC and
  > the staged XMAC's WMMA version.  gpudifx's XMAC stays as the reference, and fusebench's tight
  > XMAC check now runs it on the staged path's own FP16 spectra.  The results sections above keep
  > their measurements of the retired paths, as the record of the comparison (and
  > `docs/xmac-paths.md`).
- **Where it trails: large N** (about 100 stations and up): the staged XMAC alone is 1.3-2.5x the
  tensor XMAC from its layout; each receiver is re-read once per super-block it is paired with.
  Not needed for DiFX's usual arrays (BACKLOG.md).  ([staged.md](docs/staged.md))
- **Any station layout through a receiver map** (per frequency and receiver: an offset, a window
  stride, a station): different bandwidths and band counts, stations missing a frequency, checked
  against a double-precision reference with `xmacbench --messy`; the map costs nothing measurable
  (32-bit offsets per row: 64-bit pointers spilled registers).  ([messy.md](docs/messy.md))
