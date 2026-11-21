# Two-stage channelisation: the implementation plan (history)

_The plan as written on 2026-11-20, before the implementation.  What was built differs: a PFB
first stage only (no WOLA); a separate benchmark, `twostagebench`, rather than a `fusebench
--two-stage` option; the delay model evaluated in the stage-1 kernel; no Python model of the whole
chain (the GPU kernels are checked against a double-precision host reference instead).  See
[results.md](results.md)._


1. **Python reference.**  A numpy model of the whole two-stage chain from a sample series (both
   stations, delays and rates), checked against `two_stage.py`'s predicted responses (a tone
   sweep: gain, alias levels) and against one long FFT where they should agree.  It is the GPU
   kernels' reference for small cases.
2. **Stage-1 kernel** (`coarse_inst.cu`, one instantiation per L1 and P, cuFFTDx; P = 1 is a WOLA
   window, P > 1 a PFB, so both first stages are the same kernel with a different weight table).
   - **Block:** G consecutive coarse windows of one (antenna, band).
   - **Input:** the packed data decoded once into shared memory with the lean decode, with an
     integer shift per coarse window.
   - **Per sample:** fringe rotation (as now).
   - **Per coarse window:** the P-tap fold to L1 samples, then a real FFT of L1 points.
   - **Per coarse channel:** derotation × delay phase at the channel's centre, then a scale.
   - **Output:** FP16, time fastest per coarse channel.
   - **History:** the PFB needs (P − 1) L1 samples before each subint, from the previous one (in
     mpifxcorr, from the datastream buffer).
   - **Invalid data:** coarse samples whose input is invalid are written as zeros.
3. **Stage-2 kernel.**
   - **Per (antenna, band, coarse channel, stage-2 window):** an N2-point complex FFT (cuFFTDx,
     batched in the block), keeping K bins.
   - **Per fine channel (optional):** the window mean of the residual delay phase.
   - **Correction:** none is needed.  The PFB's passband ripple is under 0.001 dB at 16 taps
     (O = 4/3) and 0.011 dB at 4 taps (O = 2); only the 8-tap O = 4/3 design reaches 0.09 dB.
   - **Output:** FP16 into the staged XMAC's layout, unchanged downstream; a stage-2 window's data
     weight is the fraction of its coarse samples that were valid.
4. **fusebench `--two-stage Nc,P,O`**, an option beside the single-stage kernel (selected for about
   2k-4k channels and up, as DiFX would select it).
   - **Timing:** against the single-stage kernel up to 8192 channels and cuFFT beyond.
   - **Checks:** a tone sweep against `two_stage.py`; a double-precision reference of the same
     chain on a few spectra; the whole chain through the staged XMAC against gpudifx's XMAC (as
     now); and a large synthetic delay rate, two stages against one.
5. **Benchmarks.** nchan 4k-1M, Nc 16/32/64, (P, O) in {(16, 4/3), (4, 2), (8, 2)}, on the 2070,
   the A100 and the RTX PRO 6000.
6. **Later.**
   - Both stages in one block for small N2 × Nc.
   - Zoom bands by skipping the coarse channels not needed: stage 2 runs only where a zoom band is.
   - The phase-centres work (`../../phase-centres`), which wants exactly these fine channels.

## The original backlog note (fft-fusion, before 2026-11-20)

(say above 512 channels): a coarse
  first stage (e.g. 32 channels), then fringe rotation and fractional-sample correction, then a
  second stage of 32 x 32 channels. The fractional-sample correction stays accurate (it becomes
  inaccurate with very long windows and high fringe rates), and two short FFTs may be faster
  than one long one (see the channel-count table), and short second-stage FFTs keep the
  tensor-layout tile small (see the item above). The first stage is the question: a WOLA FFT
  risks artefacts at the coarse channel edges, a PFB costs complexity and computation.
