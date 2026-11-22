# binbench results

## The bins and the naive port (2026-11-22, RTX 2070 SUPER, 150 W / 1500 MHz)

`binbench` (README.md): 10 stations, 4 dual-pol 32 MHz bands from 1400 MHz, a 25.6 ms subint,
DM 30, the phase at the window's start (mpifxcorr's convention), the pulse (a Gaussian, FWHM 0.03
turns, at phase 0.5) in the subint's middle.  ms per subint; every case passing its checks (the bins
against mpifxcorr's rule in double: no differences; the naive port against the host's double sums
of the same FP16 spectra; the sum of its bins against the unbinned XMAC's sums).

| P | channels (window) | bins | empty bins | XMAC (unbinned) | bins kernel | naive | naive / XMAC |
|---|---|---|---|---|---|---|---|
| 1.5 ms | 128 (4 µs) | gate | 0/2 | 1.219 | 0.039 | 4.66 | 3.8 |
| | | 16 separate | 0/16 | 1.205 | 0.063 | 8.29 | 6.9 |
| | | 16 scrunched | 0/16 | 1.208 | 0.063 | 4.73 | 3.9 |
| | | 128 separate | 0/128 | 1.210 | 0.105 | 226.9 | 188 |
| | 1024 (32 µs) | gate | 0/2 | 1.451 | 0.037 | 9.02 | 6.2 |
| | | 16 separate | 0/16 | 1.485 | 0.049 | 23.7 | 16.0 |
| | | 16 scrunched | 0/16 | 1.462 | 0.050 | 9.10 | 6.2 |
| | | 128 separate | 0/128 | 1.450 | 0.084 | 138.9 | 96 |
| 500 ms | 128 (4 µs) | gate | 1/2 | 1.212 | 0.038 | 4.70 | 3.9 |
| | | 16 separate | 2/16 | 1.215 | 0.067 | 6.52 | 5.4 |
| | | 16 scrunched | 2/16 | 1.220 | 0.067 | 4.77 | 3.9 |
| | | 128 separate | 118/128 | 1.206 | 0.086 | 6.65 | 5.5 |
| | 1024 (32 µs) | gate | 1/2 | 1.495 | 0.038 | 8.92 | 6.0 |
| | | 16 separate | 2/16 | 1.499 | 0.065 | 11.3 | 7.5 |
| | | 16 scrunched | 2/16 | 1.481 | 0.064 | 8.86 | 6.0 |
| | | 128 separate | 118/128 | 1.487 | 0.084 | 12.2 | 8.2 |

- **The naive port** (a thread per output, over the windows): 3.8-6.2x the unbinned tensor XMAC
  when one sum suffices (the gate, scrunched, or the 500 ms pulsar's long bins), which is about a
  CUDA-core XMAC's cost; 7-16x for 16 separate bins at 1.5 ms; 96-188x for 128 separate bins at
  1.5 ms, where the bin changes every 0.4-3 windows and each change adds the sum into global memory.
  The reference, not a candidate.
  - Dispersion is what hurts it: at DM 0 the 128-bin case (1.5 ms, 128 channels) takes 12.4 ms,
    not 227.  Without it a warp's channels change bin on the same window and their adds into
    global memory coalesce; at DM 30 adjacent channels' bins differ by 5.7 windows, so they do not.
- Also checked: `--phase mid` (the window's middle) and 32 stations (16 bins scrunched: naive
  21x the XMAC's 4.8 ms) pass.
- **The bins kernel**: 0.04-0.11 ms, 3-9% of the unbinned XMAC at 10 stations (less with more
  stations: the XMAC grows with the baselines, the bins do not).  More than "negligible": the
  binary search over the ends sets it (0.04 ms with 2 bins, 0.1 with 128).  A lookup table on the
  phase's top bits would make it flat.
- **The 500 ms pulsar**: a subint covers 5% of the period, so the gate's subint is all on-pulse,
  and 118 of the 128 bins are empty (2 of the 16).
- **The signal check** (each bin's mean power against the profile's, over every baseline): within
  1.6-5.9 "sigma", where the sigma treats the products as independent; they share each station's
  noise, so it is an underestimate.  Informational, not a pass/fail.
