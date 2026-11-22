# phase-centres

GPU correlation to many phase centres at once (wide-field / multi-field VLBI) for gpudifx.
- mpifxcorr's implementation: [docs/mpifxcorr.md](docs/mpifxcorr.md).
- The GPU design and what is next: [BACKLOG.md](BACKLOG.md).
- What has been learned: [FINDINGS.md](FINDINGS.md).
- **In progress**: `phasecentrebench`, the shift after the XMAC.  First results in
  [docs/results.md](docs/results.md).

## The problem

DiFX can correlate one pointing to many phase centres in a single pass.  The visibilities are
formed at high time and frequency resolution for the pointing centre.  Then, for each phase
centre, they are shifted (the differential delay and rate to that centre, a phase per baseline,
channel and short time step) and averaged down to the output resolution.  The averaging can
then be coarse without smearing the target sources, so each centre's output is small.  mpifxcorr
shifts every `maxNSBetweenUVShifts` (each thread over its share of the subint), applies the
delay only (no rate within a piece), and averages in frequency
([docs/mpifxcorr.md](docs/mpifxcorr.md)).

## Why it follows the two-stage FFT

The pointing-centre correlation must keep fine channels (bandwidth smearing across the field) and
short time steps (time smearing).  Fine channels mean long FFTs, which is where the two-stage
channelisation in [`../two-stage`](../two-stage/README.md) should help: a coarse stage, then short FFTs.

## The XMAC's output cost: a concern if the pieces are short

fft-fusion's big matrix found that with very few windows a subint (20-23, at 64-256 stations) the
XMAC's per-subint output (zeroing, atomics, stores) outweighs its products ("The big matrix" in
[`../fft-fusion/docs/shapes-and-matrix.md`](../fft-fusion/docs/shapes-and-matrix.md)).  Counted per subint, multiple phase centres do not get there: at L band, 8k channels in a
32 MHz subband (4 kHz) with a 40 ms subint (time smearing comparable to, still a bit below, the
frequency smearing) is about 160 FFT windows a subint.  But the XMAC flushes once per shift piece, not once per subint,
so it is the windows per piece that count: with `maxNSBetweenUVShifts` at a few ms the pieces can
drop below 20 windows (BACKLOG, "Watch").

Depends on [`../xmac`](../xmac/README.md)'s staged XMAC and, for fine channels, [`../two-stage`](../two-stage/README.md).

## phasecentrebench

`phasecentrebench.cu` runs:
1. **Pieces**: the subint's windows split into `--pieces` K, as evenly as possible (or K = ceil(subint
   / `--uvshift-ns`), DiFX's `maxNSBetweenUVShifts`).
2. **The XMAC**: the staged tensor XMAC (`../common/xmac.cu`) once per piece, from FP16 station
   spectra, writing each piece's fine visibilities.
3. **The delays**: Δτ per (piece, baseline, centre) at the piece's midpoint, from a quadratic per
   (antenna, centre), as mpifxcorr forms it, including the pointing centre's (1 + rate).  Also
   mpifxcorr's decorrelation estimate.
4. **The shift**, in three versions, accumulated over the pieces into each centre's output,
   `[frequency][baseline][centre][product][output channel]`:
   - `naive`: a thread per output, a sincos per fine channel (the obvious port);
   - `simt`: the shift as a matrix product per baseline, FP32 on the CUDA cores;
   - `tensor`: the same with FP16 WMMA (FP32 sums), 32 centres a block, 3 blocks an SM.

   The product is [frequencies x products x output channels, A] x [A, centres] for A channels
   averaged.  Then an outer phasor per output, computed in FP32 from split constants (no FP64 per
   element).

The station spectra are synthetic: one source at a phase centre, each station's signal delayed
relative to the pointing centre per window, and optional per-station noise.

```bash
make phasecentrebench                  # in the repository root
phase-centres/phasecentrebench         # 10 stations, 4 x 32 MHz, 8192 channels, 100 windows, 100 centres, avg 64
phase-centres/phasecentrebench --centres 1000 --pieces 4 --avg 128 --only simt,tensor
phase-centres/run-centres.sh           # the sweeps of docs/results.md
```

`--help` lists the options: stations, frequencies, channels, bandwidth, sideband, windows, pieces,
averaging, centres, the field's delay spread, the source's centre, noise, the tensor shift's
centres a block and blocks an SM (`--blocks 3|4`).  Nsight Compute profiles of the tensor shift on
OzSTAR: the pc-* cases of `common/scripts/ncu-ozstar.sbatch` (its header).

`lagwindow.py` checks the lag-window term of mpifxcorr's decorrelation estimate by a time-domain
simulation (numpy).

**Checks** (the run fails if any fails):
- the delays and the decorrelation estimate against the host, in double;
- every shift version against a double-precision shift of the same fine visibilities (random
  picks, all products);
- the source's centre against the exact coherent sum from the delay model, which also checks
  the XMAC and the piece midpoints.

