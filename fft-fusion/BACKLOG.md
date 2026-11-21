# fft-fusion: backlog

The open items, moved verbatim from the README's "Open items" (2026-11-21); section names in
quotes are in [FINDINGS.md](FINDINGS.md)'s `docs/`.

## Next

- **The lean decode's 32-frame window** (found by the big matrix): a window may span at most 32
  VDIF frames (`leanDecodeFits`; the window descriptor's valid-frame mask is 32 bits), which 2- and
  4-bit data with 128 bands in a frame exceed at 8192-point windows.  A 64-bit mask (the descriptor
  is an `int4`: the parity bit could share a word), or a slower path for the rare long window.  The
  XMAC outputs need the lean decode, so such shapes cannot run the whole chain today.
- **The end-to-end kernel is now arithmetic-bound on the A100** (FMA pipe 66%, special-function
  unit 28%): what is left is mostly the FFT itself, the integer multiply-adds of the indexing,
  a sincos per sample for the fringe phase and one per channel for the fractional-sample
  correction.  cuFFTDx's shared-memory exchange (bank conflicts; the main stall on the 2070).

## Later

- **A wider test matrix**: 1, 2 and 4 bits x 1, 4 and 32 dual-polarisation subbands x 4-32
  stations at 128 and 1024 channels done on the 2070, and with 64 stations at 128 channels and
  32 at 1024 on the RTX PRO 6000 ("Wider shapes, bits and subbands" ([shapes-and-matrix.md](docs/shapes-and-matrix.md)), `run-wide.sh`); the A100
  still to run.  Still open: channel counts from 64 to 16384 and not powers of two, single
  polarisation, 8 bits (not compiled for the XMAC outputs), and one subband from 10 stations
  (the 2070 at 32, the RTX PRO 6000 from 10), where the layout path still leads by 1-18%.
- **FFT lengths that are not powers of two.**  The main case is the EHT: ALMA's 62.5 MHz channels
  need FFTs of N x 125 points to match the channel width of the other stations, which record
  power-of-two bandwidths (so this is also mixed-bandwidth correlation, below).  cuFFTDx (MathDx
  25.6) does some lengths directly and others only with Bluestein's algorithm, which needs a
  workspace in global memory and several FFTs of a longer power-of-two length, and is capped at
  a Bluestein length of 16384:

  | points | sm_80 and sm_75 |
  |---|---|
  | 125, 500, 1000, 8000 | direct (25, 50, 10 and 20 elements per thread: 50 is a lot of registers) |
  | 250, 2000, 4000 | Bluestein |
  | 16000 | not available (its Bluestein length, 32768, is over the cap) |

  (Other lengths: 400 direct; 640, 800, 1536, 1800, 3200, 6000 Bluestein.)  The fused kernel
  rejects the workspace for now, and in places assumes the elements per thread divide the length
  (true for the direct ones) and a multiple of 32 (`decode_to`).  To do: build and time the
  N x 125 lengths against the neighbouring powers of two, direct and Bluestein; cuFFT (any
  length, mixed radix) as the fallback in the 2-kernel pipeline where Bluestein costs too much
  or is not available; and see whether other N x 125 decompositions (e.g. 2000 as a 2-stage
  16 x 125) do better.
- **Mixed-bandwidth correlation** (e.g. 32 MHz bands at one station, 128 MHz at another; ALMA's
  62.5 MHz against 64 MHz or more elsewhere in the EHT): the channel width, so the FFT duration
  and the window count, are the same at both, but the FFT lengths differ (4 times longer at the
  128 MHz station), so each station gets its own fused-kernel size (it is launched per station
  already), and the wide station's spectrum supplies a quarter of its channels to each of the
  narrow station's bands (the input map above).  `fusebench` assumes
  one shape for every station: per-station shapes, then the two stages together with mixed
  bandwidths.  Mixed real and complex sampling works the same way.
  (The XMAC side, the input map, is done: [../xmac](../xmac/BACKLOG.md).)
- **FP16's range** for the FP16 spectra: see [../xmac/BACKLOG.md](../xmac/BACKLOG.md), "DiFX's
  flexibility in the tensor XMAC" (the writer, this kernel, should scale the spectra).
- Two kernels with WOLA through a series buffer (each sample decoded, rotated and stored once,
  as `--layout series`) rather than a buffer per window, which repeats the 12.5% overlap. That
  needs the fringe phase to be continuous from window to window.
- Phases by recurrence (one sincos per thread, then a complex multiply per element) instead of a
  sincos per element: tried for the fringe phase and dropped (no gain at 8 samples a thread, see
  "The lean decode" ([lean-decode.md](docs/lean-decode.md))); might pay for the fractional-sample correction at large elements per thread.
- Complex packed data, phase cal, and CODIF in the end-to-end kernel.
- Complex data (`--complex`).
- The effect inside mpifxcorr, where the FFT and rotation overlap with transfers and the XMAC.

## Probably obsolete

Written when the station kernel could also write the tensor layout (retired 2026-11-20, see
[../xmac/FINDINGS.md](../xmac/FINDINGS.md)); kept for the reasoning.  The big matrix has since run
the whole chain at 32, 512 and 4096 channels.

- **More channel counts for the whole chain** (`fusebench --xmac`, `run-xmac.sh`): so far only
  128 and 1024.  Writing the tensor layout costs the end-to-end kernel about 20% at 128 channels
  but 64% at 1024 on both GPUs (the tile for W windows grows with the channel count: on the 2070
  W falls and the stores shrink, on the A100 occupancy falls), and that cost decides whether the
  tensor path beats the tiled FP32 XMAC.  Sweep 64 to 16384 channels on both GPUs to find where the direct layout stops
  paying, and how that moves with the GPU's shared memory.  This bears on two-stage
  channelisation (next item): a second stage of short FFTs could write the layout with the
  small tiles of a short FFT whatever the total channel count, so the sweep also says at what
  channel count a two-stage design gains on the XMAC side as well as on the FFT side.  But the
  two stages need a DRAM round trip between them (stage 2 needs 16 x its length of each coarse
  channel's samples, which for all coarse channels is the same large tile), about as much
  traffic as the transpose: two-stage moves that cost rather than removing it, so it should rest
  on its other gains (shorter FFTs, the fractional-sample correction's accuracy).
