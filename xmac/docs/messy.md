# Stations with different band layouts (`xmacbench --messy`)

_Moved from `fft-fusion/README.md` in the 2026-11-21 reorganisation, unchanged but for the heading levels. `results-local/` paths are in `fft-fusion/` (runs of `run-xmac.sh` and `run-staged.sh`: `xmac/`). Section names in quotes refer to these docs: see [the findings index](../FINDINGS.md)._

## Stations with different band layouts: `xmacbench --messy` (RTX 2070 SUPER, 150 W + 1500 MHz, 2026-11-20)

The staged XMAC now takes a map (`XmacMap`, `xmac.cuh`): per correlated frequency and receiver,
where that receiver's FP16 spectra start and their window stride, and which station it is.  So
the stations need not share one layout: different bandwidths and numbers of bands, and stations
missing a frequency.  The regular layout is the special case `xmacRegularMap`, so it is one
kernel, not two.  Each row's place is a 32-bit offset (in `__half2`) and a 32-bit stride: as a
64-bit pointer and stride each, the KS 2 kernel spilled at its 128 registers and was 12-20%
slower at 10 stations; now the registers are as before (119, 128, 63, 64; no spills) and the regular
case is within 1-3% of before (4-64 stations, 128 and 1024 channels).  The spectra may span at
most 2^32 `__half2` (16 GiB), checked.

`xmacbench --messy`: the correlated frequencies are F dual-pol bands of the narrowest bandwidth, C
channels each.  Half the stations have bands of 2 frequencies (twice the bandwidth), the rest of 1,
and one has bands of 3 (a bandwidth and channel count that are not powers of two) covering
frequencies 1-6 of 8 only, so it lacks two.  Every band is whole correlated frequencies (no band
straddles one's edge: the 62.5 MHz case, where zoom bands must be stitched, is still open).
gpudifx's XMAC is configured as GPUCore would (per baseline each station's spectra and strides; per
frequency, baseline and product a band index in units of C channels, -1 where a baseline lacks
the frequency), and both are checked against a double-precision reference, with the absent
baselines checked to be exactly zero.  Every shape passes:

| stations | frequencies x channels | windows | gpudifx (tables) | staged (map) | staged, regular layout of the same size |
|---|---|---|---|---|---|
| 3 | 8 x 128 | 2001 | 0.423 | 0.195 | 0.199 |
| 4 | 8 x 128 | 4000 | 1.79 | 0.417 | 0.407 |
| 10 | 8 x 128 | 2000 | 6.38 | 0.810 | 0.803 |
| 10 | 6 x 512 | 501 | 4.68 | 0.757 | 0.779 |
| 32 | 8 x 128 | 1000 | 38.5 | 2.01 | 1.93 |
| 64 | 8 x 128 | 500 | 84.0 | 3.84 | 3.52 |
| 64 | 14 x 32 | 777 | 68.3 | 2.19 | 2.21 |

(ms per subint.)  The map costs nothing measurable: the messy layout is within 4% of the regular one
of the same size, 9% at 64 stations with 8 frequencies (the stations' wider bands spread a block's
reads), while it does less work (the station missing two frequencies).
