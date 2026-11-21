# xmac: backlog

The open items, moved verbatim from fft-fusion's README "Open items" (2026-11-21); section names in
quotes are in [FINDINGS.md](FINDINGS.md)'s `docs/`.

## Next

- **DiFX's flexibility in the tensor XMAC**: zoom bands (a channel range of a recorded band),
  mixed bandwidths, channel counts that are not powers of two, stations that do not record every
  frequency, single-polarisation stations, baselines not correlated, channel averaging.  The
  tensor kernel itself need not see any of it: the channels are its batch dimension (each one an
  independent X X^H), so their number is free, and the tensor cores only constrain the windows
  (blocks of 16, padded with zeros) and the receivers (16-row tiles, the spare rows discarded).
  Done for the staged XMAC's input: its map ("Stations with different band layouts" ([messy.md](docs/messy.md)), `xmacbench
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

## Later

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
  least) that already loses the chain from 10 stations ("Wider shapes" ([shapes-and-matrix.md](../fft-fusion/docs/shapes-and-matrix.md)), both GPUs).
- **The XMAC with the station processing**: the end-to-end kernel writing the tensor layout is
  done and measured on both GPUs ("The whole chain" ([whole-chain.md](docs/whole-chain.md))): it wins at 128 channels and loses at
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
  (Written when the station kernel could write the tensor layout, retired 2026-11-20; the chunking
  and accumulator-traffic analysis still applies to the staged XMAC.)

## Obsolete

- **The tiled XMAC's cost against the subint length** (above): 8 times the time for 4 times
  the windows at 16 stations on the 2070.  Sweep the windows at fixed stations, profile the
  long case (DRAM traffic, TLB: the spectra are 2.4 GiB at 10000 windows), fix it or explain it,
  and redo the tiled-against-tensor comparisons at a realistic subint length.
  (The tiled FP32 XMAC was retired on 2026-11-20.)

## Done

- **The reshuffle into the tensor layout in the XMAC**: done, the staged XMAC (above, and
  "One XMAC path").
- One XMAC path, the staged one (2026-11-20): [FINDINGS.md](FINDINGS.md).
