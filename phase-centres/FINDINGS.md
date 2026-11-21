# phase-centres: findings

Detail: [docs/mpifxcorr.md](docs/mpifxcorr.md) (how mpifxcorr does it),
[docs/results.md](docs/results.md) (phasecentrebench's timings and checks).

## mpifxcorr's implementation (read 2026-11-21)

- **It shifts every `maxNSBetweenUVShifts`, per thread.**  Each thread shifts its share of the
  subint, at each piece's midpoint.
  - The shift applies the delay only: a phase across the band, with no rate within a piece.
  - Then it averages in frequency into each centre's copy of the output.
  - Pieces are rounded down to a multiple of `numBufferedFFTs`, so the last one is short.
- **shift / XMAC ≈ 2 x centres / windows per piece** in its arithmetic.  gpudifx does not support
  multiple phase centres.  ([mpifxcorr.md](docs/mpifxcorr.md))

## phasecentrebench (2026-11-21, RTX 2070 SUPER at 150 W / 1500 MHz; A100)

- **The shift as a matrix product per baseline**: [frequencies x products x output channels, A]
  x [A, centres], with an outer phasor per output.
  - It is 13x the naive port at the default shape.
  - The FP16 tensor version runs at 380-730 Gcmac/s.
  - Checked to 1e-4 (tensor) against a double-precision shift, and against the exact coherent
    sum for a simulated source.  ([results.md](docs/results.md))
- **Shift / XMAC ≈ 0.5-0.8 x centres / windows per piece** on the 2070 at 8k channels, averages of 64 or
  more.
  - 100 centres at 100 windows a piece: +80% over the XMAC alone.
  - 10 centres: +15%.  1000 centres: 7.7x.  ([results.md](docs/results.md))
- **On the A100** the XMAC is 2.4-3.7x faster than on the 2070, the tensor shift 2.9-5.0x.
  - The tensor shift runs at 1.5-3.4 Tcmac/s (4-9% of the tensor-core rate).
  - Shift / XMAC ≈ 0.5-0.65 x centres / windows per piece.  100 centres at 100 windows: +65% over
    the XMAC, 0.039 ms per ms for the whole chain.  ([results.md](docs/results.md), "A100")
- **Short pieces cost the XMAC too**: 1.27x at 50 windows a piece, 1.8x at 25, 3.7x at 10 on the
  2070, 1.4x / 2.3x / 5.1x on the A100.  With 10s of ms pieces (O(100) windows) it does not
  matter.  ([results.md](docs/results.md))
- **Averages of 8-16 channels: the per-output work dominates**, and the tensor version loses to
  FP32 on the CUDA cores.  But they come with few centres, where the shift is cheap (0.04 ms per
  ms at 10 centres, against the XMAC's 0.075).  An outer phasor by recurrence was slower.  The
  kernel is latency-bound: 25% occupancy.  ([results.md](docs/results.md))
