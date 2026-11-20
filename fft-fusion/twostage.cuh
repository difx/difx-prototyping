// twostage.cuh - two-stage channelisation (docs/two-stage.md): a polyphase filter bank (PFB) of Nc
// coarse channels, then an FFT of each coarse channel's series.  Real-sampled data.
//
// Stage 1 (twostage1.cu): coarse window n starts at sample n H (plus its integer delay shift) and
// is P x L1 samples long, L1 = 2 Nc.  Each sample is decoded and fringe-rotated as the
// single-stage kernel does it (PreDecodeLean: the stage-1 geometry's "windows" are the coarse
// windows, its FFT length P x L1, its window weights the PFB taps), the P segments of L1 are
// summed, and an L1-point FFT gives coarse channels c = 0 .. Nc (bin Nc is the Nyquist one).  Each
// coarse channel then gets the fractional-sample correction at its centre (FracRotateFixed with
// the stage-1 geometry's nchan = Nc: channel c's frequency is c B / Nc) and the oversampling
// derotation e^{-2 pi i c n H / L1}, and is stored FP16 (times scale1), one series per (antenna,
// band, coarse channel): coarse[((ant nband + band)(Nc + 1) + c) ncoarse + n].
//
// Stage 2 (twostage2.cu): stage-2 window m is coarse samples m N2 .. m N2 + N2 - 1 of each coarse
// channel; an N2-point FFT, and the K = N2 / O bins k = -K/2 .. K/2 - 1 kept: fine channel
// j = c K + k, j = 0 .. nchan - 1 (coarse channel 0 gives only k >= 0, coarse channel Nc only
// k < 0), nchan = Nc K.  Stored FP16 (times scale2) as the staged XMAC reads it:
// out[((ant nwin2 + m) nband + band) nchan + j] (XmacShape::halfIndex).
#pragma once
#include <cuda_fp16.h>
#include "pipelines.cuh"

struct TwoStagePlan {
    int Nc;              // coarse channels (stage-1 FFT of L1 = 2 Nc points)
    int L1, P, H;        // stage-1 FFT points, PFB taps (segments of L1), hop (samples): O = L1 / H
    int N2, K;           // stage-2 FFT points, fine bins kept a coarse channel: K = N2 H / L1
    int nchan() const { return Nc * K; }
};

// Is this stage-1 FFT size / nbit, and stage-2 FFT size, compiled in?
bool twoStage1Supported(int L1, int nbit);
bool twoStage2Supported(int N2);

// Stage 1.  g1 = {nant, nband, ncoarse, P L1, Nc}; pre: the packed input with the stage-1 windows
// (sampleIndex, fringe and window descriptors per coarse window: launchPackedSetup(g1, ...) first)
// and the PFB taps as its window; p: the fractional-sample parameters per coarse window
// (launchFixedPhase(g1, p) first).  The lean decode of pk.ms.nbit (1, 2 or 4) bits.
void launchStage1(const TwoStagePlan& t, const Geometry& g1, const PreVariant& pre, const FracParams& p,
                  __half2* coarse, float scale1, cudaStream_t st);

// Stage 2: coarse (as stage 1 wrote it, g1.nwin = nwin2 N2 coarse samples) -> FP16 spectra.
void launchStage2(const TwoStagePlan& t, const Geometry& g1, int nwin2, const __half2* coarse, __half2* out,
                  float scale2, cudaStream_t st);
