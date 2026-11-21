# fft-fusion: findings

Detail, by topic, in `docs/`:
[station-kernel.md](docs/station-kernel.md) (fused against unfused, from the packed data, the
150 W baseline), [lean-decode.md](docs/lean-decode.md), [channel-count.md](docs/channel-count.md)
(channel count, chunks, register caps), [windows.md](docs/windows.md),
[shapes-and-matrix.md](docs/shapes-and-matrix.md) (the whole chain over bits, subbands, stations
and channels).  Hardware eras (the 2070's power cap and clock lock): [../README.md](../README.md#benchmark-hosts).

## The fused station kernel (2026-11-16)

- **Fusing the FFT and the fractional-sample correction is 2.7-2.8x faster than mpifxcorr's FFT +
  rotation on the 2070, 1.9-2.0x on the A100** (FP32 input).  On the 2070 fixed-point phases give
  1.35x (mpifxcorr's rotation is FP64-bound there) and fusion 2.0x; on the A100 (FP64 at half
  rate) the gain is fusion alone.  ([station-kernel.md](docs/station-kernel.md), "What the numbers
  say", "What changes on the A100")
- **FP16 input: 3.8-4.0x on the 2070, 2.1-2.2x on the A100.**  The FFT stays FP32: FP16 loads,
  converted in registers (cuFFTDx has no mixed precision, and needs none; [README.md](README.md),
  "FP16 in, FP32 out").  With FP32 input the kernel is memory-bound on both GPUs; with FP16 input
  on the A100 it is not.
- **WOLA (12.5% Vorbis tapers) costs 23-36%** over the fused kernel without it (1.143x the
  windows); overlapping 25% instead costs another 10-14%.
- **The Vorbis taper loses no continuum sensitivity at any taper length** and keeps the Tukey
  taper's far-sidelobe suppression; the shape costs nothing, only the taper length does.
  ([windows.md](docs/windows.md))
- **cuFFTDx's suggested FFTs per block is within about 6% of the best** (the best on the A100).

## From the packed data (2026-11-16 to 17)

- **The end-to-end kernel (decode, fringe rotation, FFT, correction in one) is the fastest station
  stage: 4.6x gpudifx today at 128 channels, 3.8x at 1024 on the 2070.**  ([station-kernel.md](docs/station-kernel.md),
  "From the packed data", both GPUs.)  On the A100 two kernels with an FP16 buffer between them
  catch it from 1024 channels up ([channel-count.md](docs/channel-count.md)).
- **The decode was half the end-to-end kernel; the lean decode takes 27-30% off it on the 2070,
  39-47% on the A100.**  gpudifx's tiled decode does not pay inside the fused kernel.  The number
  of bits costs little.  ([lean-decode.md](docs/lean-decode.md))
- **On the A100 the end-to-end kernel is now limited by arithmetic issue** (FMA pipe 66%), not
  memory: the FFT, the indexing and the sincos per sample and per channel.
- **Nothing argues for a minimum channel count**; longer FFTs cost more.  Keeping a 2-kernel
  buffer in L2 by chunking does not pay.  Register caps pay only with the decode in the kernel (a
  cap of 40 registers: 2.4-6.5% on the A100).  ([channel-count.md](docs/channel-count.md))
- **The 150 W cap on the 2070 slows the compute-bound kernels 9-14%** and memory-bound ones barely.
  ([station-kernel.md](docs/station-kernel.md), "RTX 2070 SUPER at the 150 W cap")

## The whole chain over many shapes (2026-11-20)

- **FP16 spectra + the staged XMAC is the fastest chain in 32 of 36 shapes on the 2070 and 40 of 54
  on the RTX PRO 6000**, and 5.4-13.8x (2070) and 7.2-18.4x (RTX PRO 6000) faster than gpudifx
  today.  Where it loses, it is the XMAC's share (many stations, one subband).
  ([shapes-and-matrix.md](docs/shapes-and-matrix.md), "Wider shapes")
- **108 shapes (1/2/4 bits x 4-256 stations x 32-4096 channels x 1-64 subbands) on the 2070 and the
  A100: no wrong results.**  One limitation found: the lean decode's window may span at most 32
  VDIF frames (2- and 4-bit, 128 bands, 4096 channels).  ([shapes-and-matrix.md](docs/shapes-and-matrix.md),
  "The big matrix")
