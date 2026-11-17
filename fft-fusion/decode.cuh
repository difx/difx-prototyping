// decode.cuh - the pre-FFT stages that start from the packed data: decode, sideband transform,
// fringe rotation and an optional window, as Pre stages (see stages.cuh) that the fused kernel
// runs on its way into the FFT.
//
// The packed data is multi-channel VDIF as mpifxcorr receives it: frames of framebytes bytes,
// payloadoffset bytes of header, then framesamples sample times of nchan (= nband) interleaved
// nbit-bit samples. The decode and sideband arithmetic is gpudifx's own (gpudecode_copy.cuh):
// PreDecode does it per sample as gpudifx does, PreDecodeLean gets the same result with the
// per-window work done once per thread.
// The fringe phase of sample j of a window, exp(-2 pi i (A j + B)) in gpudifx with A and B in
// double precision, is formed here in 64-bit fixed point (offset B, slope A, in 2^-64 turns;
// launchPackedSetup), as for the fractional-sample correction: no FP64 per sample.
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
                                      // in 2^-64 turns (filled by launchPackedSetup)
    int4* winDesc;                    // [ant][win] where each window lies in the packed data, for
                                      // PreDecodeLean (filled by launchPackedSetup; see WinDesc)
    unsigned fsMagic;                 // frameMagic(ms.framesamples)
};

// q / framesamples as __umulhi(q, frameMagic(framesamples)): exact while q * framesamples < 2^32
// (q < 2^32 / framesamples), which leanDecodeFits checks.
inline unsigned frameMagic(int framesamples) {
    return (unsigned)((0x100000000ull + (unsigned)framesamples - 1) / (unsigned)framesamples);
}

// A window's place in the packed data, one int4 per (antenna, window), so that the per-sample work
// of PreDecodeLean needs no division and no lookup of validFrames:
//   x  byte offset of the payload of the frame holding the window's first sample, from the start
//      of the antenna's data (negative if the window starts before the data)
//   y  position of the window's first sample in that frame
//   z  bit k set if frame k of the window (counting from that frame) exists and is valid
//   w  parity of the first sample's absolute sample number (datasamples + sampleIndex: ALTSIGN)
// A window may span up to 32 frames (leanDecodeFits).
inline bool leanDecodeFits(int framesamples, int fftchannels, size_t bytesPerAnt) {
    const unsigned long long qmax = (unsigned long long)framesamples + fftchannels;
    return (framesamples - 1 + fftchannels - 1) / framesamples + 1 <= 32 &&
           qmax * framesamples < 0x100000000ull && bytesPerAnt < 0x7fffffffull;
}

// Fills d.fringePhase from bigA and bigBred (one thread per (antenna, window, band)) and, if
// d.winDesc is set, the window descriptors (one thread per (antenna, window)).
void launchPackedSetup(const Geometry& g, const PackedInput& d, cudaStream_t st);

// The interface every pre-FFT stage offers its kernels (stages.cuh): prepare() once per thread,
// for a thread whose samples s0, s0 + sstep, ... all belong to one (antenna, window, band), then
// at() per sample, in that order.

// gpudifx's per-sample decode, as mpifxcorr's kernels do it (--decode direct): every sample finds
// its frame (a division), checks validFrames, and reads its byte, with the number of bits, the
// sideband flags and the absolute sample number all evaluated per sample.
struct PreDecode {
    static constexpr bool packedInput = true;
    static constexpr bool withAutocorr = true;
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

    struct State {};
    __device__ __forceinline__ State prepare(const Geometry&, int, int, int, int, int) const { return {}; }
    __device__ __forceinline__ cuFloatComplex at(State&, const Geometry& g, int ant, int win, int band,
                                                 int s) const {
        return (*this)(g, ant, win, band, s);
    }
};

// The same result as PreDecode, bit for bit, with everything that depends only on the window, the
// band or the thread done once per thread (prepare), the number of bits fixed at compile time
// and no branches per sample (--decode lean, the default).  A thread's samples are s0, s0 + sstep,
// s0 + 2 sstep, ... with sstep even (both kernels; a single sample is fine too), and the kernels
// call at() for them in that order.  Per sample:
//   - the byte and bit from the sample's bit position counted from the window's first frame,
//     plus the header bytes of the frames before it (the frame by a multiply-high), and one byte
//     load (to a safe address when the frame is invalid, so that a thread's loads can all be
//     issued together);
//   - validity from the window's bit mask;
//   - the level from selects between the thread's own signed levels: with an even sstep all of
//     a thread's samples have the same sideband sign, which is folded in;
//   - the window weight;
//   - the fringe rotation: the fixed-point phase advanced by sstep slopes (a 64-bit add, exact
//     modulo one turn, so the phase is the one PreDecode computes), then one sincos.
template<int NBIT>
struct PreDecodeLean {
    static_assert(NBIT == 1 || NBIT == 2 || NBIT == 4 || NBIT == 8, "1, 2, 4 or 8 bits");
    static constexpr bool packedInput = true;
    // the autocorrelation kernels are compiled for 2 bits only, to save build time
    static constexpr bool withAutocorr = NBIT == 2;
    PackedInput d;
    const float* window;     // fftchannels weights, or nullptr

    bool direct() const { return false; }

    struct State {
        const unsigned char* data;   // the antenna's packed data
        unsigned long long turns;    // fringe phase of the next sample, 2^-64 turns
        unsigned long long dturns;   // its step over sstep samples
        int frameOff;                // WinDesc x
        unsigned sif0;               // WinDesc y
        unsigned mask;               // WinDesc z
        unsigned bitBase;            // bit position of the band in sample time 0 of the first frame
        unsigned bitsPerTime;        // nband * NBIT
        unsigned skip;               // framebytes - payload bytes: header bytes between payloads
        bool windowed;               // window != nullptr
        float lo, hi;                // 1, 2 bits: the levels of codes 2 and 3 (the others are
                                     // their negatives); 4, 8 bits: lo is the scale. Signed.
    };

    __device__ __forceinline__ State prepare(const Geometry& g, int ant, int win, int band, int s0, int sstep) const {
        const int4 w = d.winDesc[(size_t)ant * g.nwin + win];
        const unsigned char flags = d.decodeFlags[ant * g.nband + band];
        const ulonglong2 ph = d.fringePhase[((size_t)ant * g.nwin + win) * g.nband + band];
        State st;
        st.data = d.packed + (size_t)ant * d.bytesPerAnt;
        st.turns = ph.x + (unsigned long long)s0 * ph.y;
        st.dturns = (unsigned long long)sstep * ph.y;
        st.frameOff = w.x;
        st.sif0 = (unsigned)w.y;
        st.mask = (unsigned)w.z;
        st.bitsPerTime = (unsigned)g.nband * NBIT;
        st.bitBase = st.sif0 * st.bitsPerTime + (unsigned)band * NBIT;
        st.skip = (unsigned)d.ms.framebytes - (unsigned)d.ms.framesamples * st.bitsPerTime / 8;
        st.windowed = window != nullptr;
        // ALTSIGN: the sign of sample s is - where (parity + s) is odd, the same for all of the
        // thread's samples (sstep even)
        const bool alt = flags & GPUDECODE_BAND_ALTSIGN;
        const int parity = (flags & GPUDECODE_BAND_ALTSIGN_WINDOW) ? 0 : w.w;
        const float sgn = alt && ((parity + s0) & 1) ? -1.f : 1.f;
        if constexpr (NBIT <= 2) {
            st.lo = sgn;                                                  // code 2 (1 bit: code 1)
            st.hi = (NBIT == 1 ? 1.f : (float)OPTIMAL_2BIT_HIGH) * sgn;   // code 3
        } else {
            const float quant_factor = NBIT == 4 ? 1.0 / FourBit1sigma : 1.0 / 3.3;
            st.lo = st.hi = quant_factor * sgn;
        }
        return st;
    }

    // NBIT-bit code c to its signed level, as gpudifx decodes VDIF (mark5access's tables): 1 bit
    // -1, +1; 2 bits -HiMag, -1, +1, +HiMag; 4 and 8 bits offset binary scaled as multibitsread_gpu
    __device__ __forceinline__ static float level(const State& st, unsigned c) {
        if constexpr (NBIT == 1) {
            return c ? st.lo : -st.lo;
        } else if constexpr (NBIT == 2) {
            const float m = ((c + 1) & 2) ? st.lo : st.hi;          // codes 1, 2: 1; 0, 3: HiMag
            return (c & 2) ? m : -m;
        } else {
            return ((float)c - (NBIT == 4 ? 7.5f : 128.f)) * st.lo;
        }
    }

    __device__ __forceinline__ cuFloatComplex at(State& st, const Geometry&, int, int, int, int s) const {
        const unsigned q = st.sif0 + (unsigned)s;                  // samples from the first frame's start
        const unsigned k = __umulhi(q, d.fsMagic);                  // frame of the window
        const bool ok = (st.mask >> k) & 1u;
        const unsigned bit = (unsigned)s * st.bitsPerTime + st.bitBase;   // bit, counting payloads only
        const unsigned off = ok ? (unsigned)st.frameOff + k * st.skip + (bit >> 3) : 0u;   // >= 0 if ok
        const unsigned c = ((unsigned)__ldg(st.data + off) >> (bit & 7)) & ((1u << NBIT) - 1);
        const float w = st.windowed ? __ldg(window + s) : 1.f;
        const float x = ok ? level(st, c) * w : 0.f;
        const float t = (float)(int)(st.turns >> 32) * 2.3283064365386963e-10f;     // [-0.5, 0.5) turns
        st.turns += st.dturns;
        cuFloatComplex cr;
        __sincosf(-(float)TWO_PI * t, &cr.y, &cr.x);
        return make_cuFloatComplex(x * cr.x, x * cr.y);
    }

    __device__ __forceinline__ cuFloatComplex operator()(const Geometry& g, int ant, int win, int band, int s) const {
        State st = prepare(g, ant, win, band, s, 2);
        return at(st, g, ant, win, band, s);
    }
};
