// stages.cuh - the per-element processing stages either side of the FFT.
//
// Both pipelines are built from the same stage objects, so they do the same arithmetic and differ
// only in how the work is split into kernels:
//   unfused:  [materialise kernel: Pre -> FP32 buffer]  ->  cuFFT  ->  rotation kernel
//   fused:    one cuFFTDx kernel: Pre into registers -> FP32 FFT in shared memory -> Post -> store
//
// A pre-FFT stage ("Pre") produces FP32 FFT input sample s of (ant, window, band):
//     __device__ cuFloatComplex operator()(const Geometry&, int ant, int win, int band, int s) const;
//     __device__ size_t sourceIndex(...) const, float weight(int s) const
//         where the sample comes from, and the weight applied to it; the host check rebuilds the
//         exact (unrounded) FFT input from these
//     bool direct() const   true if the unfused pipeline can hand the stored buffer straight to
//                           cuFFT (FP32, one buffer per window, no weights), as mpifxcorr does now
// To try a new pre-FFT step, write a struct like PreLoad and add it to withPre() below.
//
// The post-FFT stage is FracRotateFixed: the fractional-sample correction with each spectrum's
// phase slope computed once in double and each channel's phase in 64-bit fixed point.  The
// unfused pipeline can instead run mpifxcorr's own kernel (copied verbatim in unfused.cu), which
// computes every phase in double precision.
#pragma once
#include <type_traits>
#include <cuda_fp16.h>
#include "common.cuh"

__device__ __forceinline__ float2 toFloat2(float2 v) { return v; }
__device__ __forceinline__ float2 toFloat2(__half2 v) { return __half22float2(v); }

// The FFT input as stored by the stage before it (today: complex_fringe_rotated_gpu, written by
// mpifxcorr's fused decode + fringe-rotation kernel), in FP32 (float2) or FP16 (__half2), in
// one of two layouts:
//   SERIES=false  one buffer per window, [ant][win][band][fftchannels] (mpifxcorr's layout)
//   SERIES=true   one continuous series per (ant, band), [ant][band][seriesLen]; window w starts
//                 at sample w * hop, so windows overlap when hop < fftchannels
// with an optional window function (weights, one per FFT input sample) applied on load: with
// hop < fftchannels this is a windowed overlap-add (WOLA) front end.
template<class T, bool SERIES>
struct PreLoad {
    const T* in;
    const float* window;      // fftchannels weights, or nullptr for none
    int hop;                  // SERIES: samples between window starts
    size_t seriesLen;         // SERIES: samples per (ant, band)

    bool direct() const { return std::is_same<T, float2>::value && !SERIES && window == nullptr; }
    __device__ __forceinline__ size_t sourceIndex(const Geometry& g, int ant, int win, int band, int s) const {
        if (SERIES)
            return ((size_t)ant * g.nband + band) * seriesLen + (size_t)win * hop + s;
        return g.index(ant, win, band, s);
    }
    __device__ __forceinline__ float weight(int s) const { return window ? window[s] : 1.0f; }
    __device__ __forceinline__ cuFloatComplex operator()(const Geometry& g, int ant, int win, int band, int s) const {
        float2 v = toFloat2(in[sourceIndex(g, ant, win, band, s)]);
        if (window) {
            const float w = window[s];
            v.x *= w;
            v.y *= w;
        }
        return v;
    }
};

// One pre-FFT stage chosen at run time.
struct PreVariant {
    bool half;                // FP16 storage
    bool series;              // continuous-series layout
    const void* in;
    const float* window;
    int hop;
    size_t seriesLen;
};

// Calls f(pre) with the PreLoad matching v. Every pre-FFT stage type the pipelines can run is
// instantiated here, so this is the one place to add a new one.
template<class F>
void withPre(const PreVariant& v, F&& f) {
    if (v.half) {
        if (v.series) f(PreLoad<__half2, true>{(const __half2*)v.in, v.window, v.hop, v.seriesLen});
        else          f(PreLoad<__half2, false>{(const __half2*)v.in, v.window, v.hop, v.seriesLen});
    } else {
        if (v.series) f(PreLoad<float2, true>{(const float2*)v.in, v.window, v.hop, v.seriesLen});
        else          f(PreLoad<float2, false>{(const float2*)v.in, v.window, v.hop, v.seriesLen});
    }
}

// Parameters of the fractional-sample correction for all antennas, laid out antenna-major. The
// per-antenna slices are what one GPUMode holds in mpifxcorr.
struct FracParams {
    const float* fracSampleError;           // [ant][win], microseconds
    const bool* validSamples;               // [ant][win]
    const double* clockOffsets;             // [ant][band]  recordedfreqclockoffsets
    const double* clockOffsetsDelta;        // [ant][band]  recordedfreqclockoffsetsdelta
    const double* fracFreqOffset;           // [ant][band]  0 except on the CPU-compat test path
    const unsigned char* bandFlags;         // [ant][band]  GPUDECODE_BAND_CONJ_DC only matters here
    const unsigned int* indices;            // [freq][MAX_INDICIES] band pairs (same for all antennas)
    const int* counts;                      // [freq] bands per frequency (same for all antennas)
    cuFloatComplex* autocorr;               // [ant][2 * nband][nchan]: band autos, then cross-pol
    const ulonglong2* fixedPhase;           // [ant][win][band] (offset, slope) in 2^-64 turns
                                            // (filled by launchFixedPhase)
    double recordedBandwidth;               // MHz
    int numRecordedFreqs;
    int numBlocks;                          // windows to process (mpifxcorr's numblocks)
    bool crossPol;                          // calccrosspolautocorrs
    bool crossPolInBlock;                   // fused pipeline: form the cross-pol products inside
                                            // the fused kernel (needs pairs (2f, 2f+1), even nband)
};

// GPUDECODE_BAND_CONJ_DC from mpifxcorr/src/gpudecode.cuh
#define FUSE_BAND_CONJ_DC 0x08

// What every rotation stage shares: window validity and the autocorrelation accumulation as
// mpifxcorr's rotation kernel does it (only used with --autocorr band|full).
template<bool DOAUTOCORR>
struct FracCommon {
    static constexpr bool doAutocorr = DOAUTOCORR;
    FracParams p;

    __device__ __forceinline__ bool windowValid(const Geometry& g, int ant, int win) const {
        return p.validSamples[(size_t)ant * g.nwin + win] && win < p.numBlocks;
    }

    __device__ __forceinline__ void accumulate(const Geometry& g, int ant, int band, int chan, cuFloatComplex v) const {
        if (DOAUTOCORR) {
            float* a = (float*)&p.autocorr[(size_t)ant * 2 * g.nband * g.nchan + (size_t)band * g.nchan + chan];
            atomicAdd(a, v.x * v.x + v.y * v.y);
        }
    }

    // The cross-polarisation pass for one (window, channel), reading the rotated spectra of this
    // antenna back (spectra = the antenna's [win][band][fftchannels] block).
    __device__ __forceinline__ void crossPol(const Geometry& g, int ant, int win, int chan,
                                             const cuFloatComplex* spectra) const {
        for (int f = 0; f < p.numRecordedFreqs; f++) {
            if (p.crossPol && p.counts[f] > 1) {
                const size_t bandA = p.indices[f * MAX_INDICIES + 0];
                const size_t bandB = p.indices[f * MAX_INDICIES + 1];
                const size_t windowBase = (size_t)win * g.fftchannels * g.nband + chan;
                crossPolPair(g, ant, bandA, bandB, chan, spectra[windowBase + bandA * g.fftchannels],
                             spectra[windowBase + bandB * g.fftchannels]);
            }
        }
    }

    // accumulate a conj(b) and b conj(a), a and b being channel chan of bands bandA and bandB
    __device__ __forceinline__ void crossPolPair(const Geometry& g, int ant, size_t bandA, size_t bandB, int chan,
                                                 cuFloatComplex a, cuFloatComplex b) const {
        cuFloatComplex* autos = p.autocorr + (size_t)ant * 2 * g.nband * g.nchan;
        const size_t crossBase = (size_t)g.nband * g.nchan + chan;
        float* x = (float*)&autos[crossBase + bandA * g.nchan];
        const cuFloatComplex ab = cuCmulConjf(a, b);
        atomicAdd(x, ab.x); atomicAdd(x + 1, ab.y);
        float* y = (float*)&autos[crossBase + bandB * g.nchan];
        const cuFloatComplex ba = cuCmulConjf(b, a);
        atomicAdd(y, ba.x); atomicAdd(y + 1, ba.y);
    }
};

// The fractional-sample rotation with each channel's phase from the per-spectrum fixed-point
// slope and offset: phase = offset + chan * slope (mod 1 turn, by 64-bit wrap-around).  The slope
// and offset come from the same double-precision expressions as mpifxcorr's kernel, once per
// spectrum (launchFixedPhase); per element there is no FP64 at all.  The phase agrees with the
// double-precision one to ~1e-12 turns before the conversion to float.
template<bool DOAUTOCORR>
struct FracRotateFixed : FracCommon<DOAUTOCORR> {
    using FracCommon<DOAUTOCORR>::p;
    __device__ __forceinline__ cuFloatComplex apply(const Geometry& g, int ant, int win, int band, int chan,
                                                     cuFloatComplex in) const {
        const ulonglong2 ph = p.fixedPhase[((size_t)ant * g.nwin + win) * g.nband + band];
        const unsigned long long turns = ph.x + (unsigned long long)chan * ph.y;
        const float t = (float)(int)(turns >> 32) * 2.3283064365386963e-10f;     // [-0.5, 0.5) turns
        cuFloatComplex cr;
        __sincosf((float)TWO_PI * t, &cr.y, &cr.x);
        if (chan == 0 && (p.bandFlags[ant * g.nband + band] & FUSE_BAND_CONJ_DC))
            in.y = -in.y;
        const cuFloatComplex v = cuCmulf(in, cr);
        this->accumulate(g, ant, band, chan, v);
        return v;
    }
};

// Which rotation the unfused pipeline runs.
//   DOUBLE  mpifxcorr's kernel, verbatim: the phase of every element in double precision
//           (FP64-bound on GPUs with slow FP64, such as the RTX 2070: 1/32 of FP32)
//   FIXED   FracRotateFixed in the same launch geometry and loop; the fused kernel always uses it
enum class Rotator { DOUBLE, FIXED };
