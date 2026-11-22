# phase-centres: backlog

mpifxcorr's implementation is read: [docs/mpifxcorr.md](docs/mpifxcorr.md).  The key ratio there:
shift / XMAC ≈ 2 × centres / windows per piece.

## Done

- `phasecentrebench`, the shift after the XMAC, checked and swept on the 2070 and the A100
  ([docs/results.md](docs/results.md)).  It has the pieces from `maxNSBetweenUVShifts`, delays per
  (antenna, centre), and the naive, SIMT and tensor shifts.
- The optimisation round (2070, docs/results.md "The optimisation round"):
  - the shift's occupancy: 32 centres a block, 3 blocks an SM; outputs stored once; inner phasors
    by recurrence (2-3x at 100+ centres);
  - the staged XMAC skips zeroing when every visibility is stored (5-20%).
- The A100 after the optimisation round (job 21370182): the tensor shift 1.1-2.1x faster.
- Lower sideband (`--sideband`).
- The lag-window term of the decorrelation estimate checked by simulation (`lagwindow.py`).

## Tried and dropped

- The XMAC with the pieces in one launch: 0-1%.  The short pieces' cost is the 16-window MMA
  tile and a full set of fine visibilities a piece, not the launches.
- The shift in the XMAC's epilogue: a block has 8-16 channels, an output 64+; the atomics alone
  cost 2-260x the whole shift.
- Outer phasors by recurrence: slower.  Centre tiles fastest in the grid: no gain.
- Double buffering: not tried (3 blocks an SM overlap the loads already; 2 chunks a piece at 64
  averaged; a second buffer would cost the third block).

## Next

- **The A100's occupancy** (only if the shift matters there: it is +45% over the XMAC at 100
  centres).  The A100 run after the optimisation round (docs/results.md) gained 1.1-2.1x, not the
  2070's 1.4-3.0x: 80 registers allow 3 blocks an SM, 24 of its 64 warps.  Set up: `--blocks 4`
  (64 registers, a 32-byte spill; 3-12% slower on the 2070, where shared memory allows only 3
  blocks anyway), and the pc-* cases in `common/scripts/ncu-ozstar.sbatch`: ncu and a timing run
  at 3 and 4 blocks, for the occupancy and how much the FP32 work (phasors, outer phasor) costs.
- **Short pieces** (only if a user wants pieces of a few ms): windows per piece not a multiple of
  16 pay for the whole MMA tile; pieces rounded to multiples of 16 windows would avoid it.

## Later

- A NUFFT shift for centres far more than the channels averaged.
- In DiFX: the shift kernel after gpudifx's XMAC; `maxNSBetweenUVShifts` to pieces as here.  How
  its tunables (centres per block, blocks per SM) are chosen: see "Carry to DiFX" in
  [../BACKLOG.md](../BACKLOG.md).

## The GPU design (prototyped in phasecentrebench)

- **Pieces from the same keyword.**  `maxNSBetweenUVShifts` sets the number of pieces a subint,
  K = ceil(subint / maxNS).  The subint's W windows are split as evenly as possible: piece k is
  windows [round(kW/K), round((k+1)W/K)).
  - Every piece is ≤ maxNS, and they differ by at most one window.  mpifxcorr instead rounds
    down to a multiple of `numBufferedFFTs` and leaves a short remainder.
  - A window's time is its centre, which is well-defined for the overlapping WOLA and two-stage
    windows too.
  - Each piece is shifted at its own midpoint.
- **Delays per antenna, not per baseline.**  Once a subint, the host fits a quadratic over the
  subint per (antenna, centre), as the two-stage kernel's `TwoStageModel` does: antennas x
  centres x 3 doubles.  The shift kernel evaluates it at each piece's midpoint, including the
  (1 + rate) factor and the decorrelation estimate.  mpifxcorr calls the interpolator per baseline,
  so every antenna's delays are computed once per baseline it is in.
- **The XMAC writes one set of fine visibilities per piece** (K output slots, or K launches over
  window ranges).
  - A piece's set is baselines x bands x pols x channels x 8 bytes.  32 stations (528 products
    with the autos), 8 bands of 8192 channels, 4 pols: 1.1 GB.
  - So on the A100 all K slots can be held at once; on the 2070 one at a time.
- **The shift as a matrix product.**  Per (baseline, band, piece), with A = `channelinc` channels
  averaged into each of G output channels, fine channel width δf and differential delay Δτ_s:
  out[g, s] = e^{2πi Δτ_s (f_LO + g A δf)} · Σ_k V[g, k] e^{2πi Δτ_s k δf}.
  - The sum is a [G·pols × A] x [A × S] complex matrix product, because the pols share Δτ.  It
    runs on tensor cores (TF32, or FP16 with a scale).  Then the product is scaled by G x S outer
    phasors.
  - Its sin/cos count is (A + G) x S per baseline instead of A·G x S.
  - This is mpifxcorr's two-level rotator, with the averaging folded into the inner level.

## Approaches to compare (prototype and measure each)

- **Shift after the XMAC**, as above, from the fine visibilities in DRAM:
  - the direct form (CUDA cores, mpifxcorr's arithmetic) against the matrix product;
  - several centres per read of the fine visibilities.
- **Shift in the XMAC's epilogue**: the staged XMAC holds a piece's sums in registers, so the
  fine visibilities never reach DRAM (they cost 16 bytes per baseline-band-pol-channel per piece,
  written and read).
  - A block would need a whole group of A channels.
  - The shift's work (centres/windows times the XMAC's) lands in the XMAC kernel.
  - Probably wins only for a few centres or long pieces.
- **Shift per station before the XMAC**: an XMAC per centre.  Only for a handful of centres.
- **Later, for centres ≫ A**: the sum over k is a non-uniform DFT, from A channels to the S
  arbitrary delays.  An FFT of each group to lags, then interpolation at each Δτ_s, costs
  ~A log A + S x taps per group instead of A x S (a type-2 NUFFT).

## Overlap ("hiding the shift's latency")

- On the GPU this is a matter of throughput, not latency.  The XMAC and the matrix-product shift
  each fill the GPU and are compute-bound, so running them concurrently gains little beyond tail
  and launch gaps.
- Order within a subint:
  1. the station stage over the whole subint;
  2. per piece, the XMAC then the shift, in one stream (or the shift of piece k on a second
     stream while the XMAC does piece k+1).
- gpudifx's existing overlap across subints (copies, the next subint's station stage) is
  unaffected.
- Splitting the station stage into pieces too mainly saves memory, not time.  Worth it only if
  the fine visibilities of all K pieces do not fit.
- Measure sequential first.  Add streams only if the timeline shows idle SMs (small shift
  kernels at few centres).

## Watch: short pieces make the XMAC output-bound

- The big matrix found the XMAC output-bound below about 20 windows a flush at many baselines
  ("The big matrix" in [../fft-fusion/docs/shapes-and-matrix.md](../fft-fusion/docs/shapes-and-matrix.md)).
- It said DiFX does not go there, counting windows per subint.  With K pieces a subint it is
  windows per piece that counts, and that can go there.
- The fine visibilities' DRAM traffic per piece against the station spectra read in that piece
  is about 4 x stations / windows per piece.
- In practice: pieces of 10s of ms are usually acceptable for smearing.  At 4 kHz channels (a
  250 µs window) that is O(100) windows a piece, clear of it.  Only a user who insists on minimal
  time smearing (pieces of a few ms) gets there.

## What to measure

- Cost against:
  - centres (1, 10, 100, 1000);
  - windows per piece (4-200);
  - the averaging A (1-256);
  - stations (10, 32, 64).
- For each, the share of shift, XMAC, and fine-visibility traffic.
- Check the visibilities against a direct correlation at each centre (that centre as the delay
  centre), and the decorrelation estimate against mpifxcorr's formula.
- Memory per piece.
