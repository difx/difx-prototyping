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
- **The optimisation round (2026-11-22, 2070): the tensor shift 2.0-3.0x faster with 100+ centres**
  (1.4-1.5x at 1-10, where it was cheap already), the XMAC 5-20%.
  - 32 centres a block, 3 blocks an SM (80 registers, 21 KB), each output stored once instead of
    zeroed and added, inner phasors by a short recurrence.  23-34% of the tensor-core rate.
  - Shift / XMAC ≈ 0.25-0.36 x centres / windows per piece.  100 centres at 100 windows: +37% over
    the XMAC alone (was +80%); 1000 centres: 3.8x (was 7.7x).
  - The staged XMAC no longer zeroes its output when every visibility is stored anyway (5-20%).
  - Ruled out: the shift in the XMAC's epilogue (its atomics alone cost 2-260x the whole shift);
    the XMAC's pieces in one launch (0-1%).  ([results.md](docs/results.md), "The optimisation round")
- **The A100 after the optimisation round: the tensor shift 1.1-2.1x faster** (1.55x at the
  default, 1.8x at 1000 centres), less than on the 2070.
  - 2.4-3.7 Tcmac/s with 100+ centres, 6-9% of the tensor-core rate.  Shift / XMAC ≈ 0.3-0.45 x
    centres / windows per piece.  100 centres: +45% over the XMAC (was +65%); 1000: 4.1x (was 5.8x).
  - The gain is mostly storing each output once; the matrix product barely moved (1.08x at 10
    pieces).  The same 80 registers fit only 3 blocks an SM: 37.5% of the A100's warps (75% on
    the 2070).  Four blocks (64 registers) is 3-12% slower on the 2070, untested on the A100.
  - XMAC 4-9% faster, within its 9% run-to-run spread.  ([results.md](docs/results.md), "A100 ...
    after the optimisation round")
- **Lower sideband** works through a base frequency per band, as mpifxcorr's rotator.
- **mpifxcorr's lag-window term, 1 - |Δτ| / (nchan / bw), is right** to 5e-4 in a time-domain
  simulation (`lagwindow.py`).  ([results.md](docs/results.md))
- **Averages of 8-16 channels: the per-output work dominates**, and the tensor version loses to
  FP32 on the CUDA cores.  But they come with few centres, where the shift is cheap (0.04 ms per
  ms at 10 centres, against the XMAC's 0.075).  An outer phasor by recurrence was slower.  The
  kernel is latency-bound: 25% occupancy.  ([results.md](docs/results.md))
