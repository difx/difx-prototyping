// decode.cuh - the pre-FFT stage that starts from the packed data: decode, sideband transform,
// fringe rotation and an optional window, as one Pre stage (see stages.cuh) that the fused kernel
// runs on its way into the FFT.
//
// The packed data is multi-channel VDIF as mpifxcorr receives it: frames of framebytes bytes,
// payloadoffset bytes of header, then framesamples sample times of nchan (= nband) interleaved
// nbit-bit samples. The decode and sideband arithmetic is gpudifx's own (gpudecode_copy.cuh).
// The fringe phase of sample j of a window, exp(-2 pi i (A j + B)) in gpudifx with A and B in
// double precision, is formed here in 64-bit fixed point (offset B, slope A, in 2^-64 turns;
// launchFringePhase), as for the fractional-sample correction: no FP64 per sample.
#pragma once
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

// The fields of mark5access's struct mark5_stream that gpudifx's decode reads.
struct mark5_stream {
    int framesamples;       // sample times per frame
    int framebytes;         // bytes per frame, header included
    int payloadoffset;      // header bytes
    int nbit;               // bits per sample
    int nchan;              // channels (bands) per sample time
    int decimation;         // 1 here
};
#define OPTIMAL_2BIT_HIGH 3.3359    // mark5access/mark5_stream.h

#define GPUDECODE_BAND_CONJ           0x1    // from mpifxcorr/src/gpudecode.cuh
#define GPUDECODE_BAND_ALTSIGN        0x2
#define GPUDECODE_BAND_ALTSIGN_WINDOW 0x4
#define GPUDECODE_BAND_CONJ_DC        0x8

#include "gpudecode_copy.cuh"
#include "common.cuh"

// The packed input of all antennas, and what gpudifx's GPUMode holds to decode it.
struct PackedInput {
    const unsigned char* packed;      // [ant][bytesPerAnt]
    size_t bytesPerAnt;
    mark5_stream ms;
    int nframes;                      // frames per antenna (framestounpack)
    const bool* validFrames;          // [ant][nframes]
    const int* sampleIndex;           // [ant][win]: first sample of each FFT window
    const unsigned char* decodeFlags; // [ant][band] GPUDECODE_BAND_*
    long datasamples;                 // absolute sample number of the first sample (ALTSIGN)
    const double* bigA;               // [ant][win][band] fringe phase slope, turns per sample
    const double* bigBred;            // [ant][win][band] fringe phase at sample 0, turns
    ulonglong2* fringePhase;          // [ant][win][band] the same in fixed point: (offset, slope)
                                      // in 2^-64 turns (filled by launchFringePhase)
};

// Fills d.fringePhase from bigA and bigBred: one thread per (antenna, window, band).
void launchFringePhase(const Geometry& g, const PackedInput& d, cudaStream_t st);

// TILED selects how the fused kernel decodes:
//   false  each FFT thread decodes its own samples of its own band straight from the packed bytes
//          (the bands of a sample time share bytes, so the other bands re-read them through L1)
//   true   the block first decodes all its spectra into shared memory with band on the lane, as
//          gpudifx's tiled kernel does (a warp reads a few contiguous bytes, every byte used by
//          several lanes), then the FFT threads read their samples from there
// Anywhere else (the unfused pipeline's materialise kernel) both decode directly.
template<bool TILED>
struct PreDecode {
    static constexpr bool staged = TILED;
    static constexpr bool packedInput = true;
    PackedInput d;
    const float* window;     // fftchannels weights, or nullptr

    bool direct() const { return false; }

    // decoded, sideband-transformed real sample s of (ant, win, band): gpudifx's arithmetic
    __device__ __forceinline__ float raw(const Geometry& g, int ant, int win, int band, int s) const {
        const int global_sample = d.sampleIndex[(size_t)ant * g.nwin + win] + s;
        float v = 0.f;
        if (global_sample >= 0) {
            const int frame = (int)((unsigned)global_sample / (unsigned)d.ms.framesamples);
            if (frame < d.nframes && d.validFrames[(size_t)ant * d.nframes + frame]) {
                const int sample_in_frame = global_sample - frame * d.ms.framesamples;
                const unsigned char* fp = d.packed + (size_t)ant * d.bytesPerAnt +
                                          (size_t)frame * d.ms.framebytes + d.ms.payloadoffset;
                const int nbit = d.ms.nbit;
                const int bit = sample_in_frame * nbit * d.ms.nchan + band * nbit;
                if (nbit <= 2) {
                    const int c = (fp[bit >> 3] >> (bit & 7)) & ((1 << nbit) - 1);
                    const float m = nbit == 1 ? 1.f : (c == 0 || c == 3) ? (float)OPTIMAL_2BIT_HIGH : 1.f;
                    v = (nbit == 1 ? c != 0 : c >= 2) ? m : -m;
                } else {
                    v = multibitsread_gpu(((const int32_t*)fp)[bit >> 5], bit & 31, nbit);
                }
            }
        }
        return band_transform_gpu(make_cuFloatComplex(v, 0.f), d.decodeFlags[ant * g.nband + band],
                                  d.datasamples + (long)global_sample, s).x;
    }

    // fringe rotation (fixed point) and window of a decoded sample
    __device__ __forceinline__ cuFloatComplex rotate(const Geometry& g, int ant, int win, int band, int s,
                                                     float x) const {
        const ulonglong2 ph = d.fringePhase[((size_t)ant * g.nwin + win) * g.nband + band];
        const unsigned long long turns = ph.x + (unsigned long long)s * ph.y;
        const float t = (float)(int)(turns >> 32) * 2.3283064365386963e-10f;     // [-0.5, 0.5) turns
        cuFloatComplex cr;
        __sincosf(-(float)TWO_PI * t, &cr.y, &cr.x);
        if (window) x *= window[s];
        return make_cuFloatComplex(x * cr.x, x * cr.y);
    }

    __device__ __forceinline__ cuFloatComplex operator()(const Geometry& g, int ant, int win, int band, int s) const {
        return rotate(g, ant, win, band, s, raw(g, ant, win, band, s));
    }

    // ---- the fused kernel's interface (stage, then load) ----
    // shared-memory tile: fpb rows of n decoded samples, row stride padded so that the band-on-lane
    // writes (lane = row + fpb * sample) hit 32 different banks
    static __host__ __device__ constexpr unsigned tileStride(unsigned n, unsigned fpb) {
        return n + (fpb <= 32 ? 32 / fpb : 0);
    }
    static constexpr unsigned tileBytes(unsigned n, unsigned fpb) {
        return TILED ? tileStride(n, fpb) * fpb * (unsigned)sizeof(float) : 0;
    }

    // Spectra f0 + y (y < fpb, band fastest) of the launch; every thread of the block takes part.
    __device__ __forceinline__ void stage(const Geometry& g, int ant0, unsigned long long f0, unsigned long long nfft,
                                          unsigned n, unsigned fpb, float* tile) const {
        if constexpr (TILED) {
            const unsigned nthreads = blockDim.x * blockDim.y;
            const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
            const unsigned stride = tileStride(n, fpb);
            for (unsigned i = tid; i < n * fpb; i += nthreads) {
                const unsigned y = i % fpb, s = i / fpb;      // band (row) on the lane
                const unsigned long long f = f0 + y;
                if (f < nfft) {
                    const int band = f % g.nband;
                    const unsigned long long t = f / g.nband;
                    tile[y * stride + s] = raw(g, ant0 + (int)(t / g.nwin), (int)(t % g.nwin), band, s);
                }
            }
        }
    }

    __device__ __forceinline__ cuFloatComplex load(const Geometry& g, int ant, int win, int band, int s,
                                                   unsigned y, unsigned n, unsigned fpb, const float* tile) const {
        if constexpr (TILED)
            return rotate(g, ant, win, band, s, tile[y * tileStride(n, fpb) + s]);
        else
            return (*this)(g, ant, win, band, s);
    }
};
