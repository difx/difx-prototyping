# two-stage: channelisation beyond one FFT

Two-stage channelisation for spectral resolutions one cuFFTDx FFT cannot reach (more than 8192
channels per real-sampled subband), or delay rates at which one long window's fractional-sample
correction loses (spacecraft VLBI).  Stage 1 is an oversampled polyphase filter bank (PFB) of Nc
coarse channels, from the packed data, with the integer delay shift per coarse window, fringe
rotation, and the fractional-sample correction at each coarse channel's centre; stage 2 is an
FFT of each coarse channel's series, keeping the central 1/O of its bins, written FP16 in the
staged XMAC's layout.  The kernels are `../common/twostage.cuh`, `twostage1.cu` and
`twostage2.cu`.

The direction (2026-11-20): one stage whenever it can do the job; two stages, with a moderate
number of PFB coarse channels (about 32), only where it cannot.

- [FINDINGS.md](FINDINGS.md): what has been learned, with links to the analysis and the
  measurements in `docs/`.
- [BACKLOG.md](BACKLOG.md): what is in progress and still to do.

## Run

Build in the repository root (`make`, see [../README.md](../README.md#building)); then here:

```bash
./twostagebench --help
./twostagebench --nc 32 --os 4:3 --taps 8              # ~64k channels (49152 at O = 4/3)
./twostagebench --nc 32 --os 2 --taps 4 --nchan 8192 --nwin 128
./twostagebench --rate 1e-4 --accel 0.5                # stress the delay model
./run-twostage.sh                                      # 16/32/64 coarse channels x O 4:3, 2 x 4/8/16 taps, ~64k channels
NCHAN=8192 NWIN=128 ./run-twostage.sh                  # the same at ~8k channels
python3 two_stage.py                                   # the analysis tables (modes: offset, ghost, pfb)
python3 plot_two_stage.py                              # the plots in plots/
```

`run-twostage.sh` also times gpudifx today at the same channel count and the single-stage
end-to-end kernel at 8192 channels (`../fft-fusion`'s `fusebench`, skipped if it is not built),
for the slowdown column.  Results go to `results-local/<host>-twostage[-8192]-<stamp>/`.

The main options: `--nc` (16, 32, 64 coarse channels: stage-1 FFTs of 32, 64, 128 points), `--os
4:3|2` (oversampling), `--taps P`, `--nchan` (the channel count wanted; the nearest the
oversampling allows, with stage-2 FFTs of 128-8192 points), `--nant --nband --nbit --bw --nwin`,
`--rate` and `--accel` (the delay model), `--csv FILE`.

### Checks

Each run checks the whole chain against a double-precision host reference (decode from the
packed bytes, fold, DFT, corrections, stage-2 DFT) on `--check N` series, coarse channels 0 and Nc
always among them: 2e-3 rms (FP16 storage gives about 4e-4).  The benchmark passes null pointers
for every per-coarse-window array, so stage 1 provably reads nothing per window.

## Code

| file | what |
|---|---|
| `twostagebench.cu` | the benchmark: packed data, the delay model, the PFB taps, the reference, timing |
| `run-twostage.sh` | the sweep over the PFB's parameters |
| `two_stage.py` | the analysis: exact sensitivity, leakage, alias images, delay accuracy, PFB design |
| `plot_two_stage.py` | the plots in `plots/` |
