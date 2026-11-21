// twostage1.cu - two-stage channelisation, stage 1: the polyphase filter bank (twostage.cuh).
//
// This software contains source code provided by NVIDIA Corporation (cuFFTDx usage).
#include <algorithm>
#include <cufftdx.hpp>
#include "twostage.cuh"

#ifndef FUSE_SM
#error "compile with -DFUSE_SM=<arch, e.g. 750 for sm_75>"
#endif

using namespace cufftdx;

// L1-point FFTs at 8 elements a thread (4, 8, 16 threads an FFT at L1 = 32, 64, 128); FPB coarse
// windows a block, 256 threads.  The decoded span ((FPB - 1) H + P L1 samples) and the folded
// windows (FPB rows of ROW) are FP16 (__half2): in FP32 they took 35 KB a block, one block (8 warps)
// an SM on the 2070, and the kernel was latency-bound; in FP16, 3 blocks an SM at 16 taps and
// O = 4/3.  Each FP16 rounding is ~3e-4 rms of the signal (the check bounds the whole chain).
// The fold does not use the FFT's register layout: there a warp spans several windows, H samples
// apart in the span, whose reads fell in the same banks.  It is done thread per (window, element),
// a warp reading 32 consecutive samples of one window, into rows padded so that the FFT's own loads
// from them (a warp: 32 / (L1 / 8) windows x L1 / 8 elements) fall in different banks too.
template<unsigned L1>
struct Stage1FFT {
    using Base = decltype(Size<L1>() + Precision<float>() + Type<fft_type::c2c>() +
                          Direction<fft_direction::forward>() + SM<FUSE_SM>() + Block());
    static constexpr unsigned E = 8;
    static constexpr unsigned FPB = 256 / (L1 / E);
    static constexpr unsigned ROW = L1 == 32 ? 36 : L1 == 64 ? 72 : 144;   // __half2 a folded row
    using FFT = decltype(Base() + ElementsPerThread<E>() + FFTsPerBlock<FPB>());
};
static constexpr int SPAN_SLACK = 16;          // samples beyond (FPB - 1) H + P L1: integer-shift steps

// the NBIT-bit code c as gpudifx decodes it (mark5access's levels)
template<int NBIT>
__device__ __forceinline__ float level(unsigned c) {
    if constexpr (NBIT == 1) return c ? 1.f : -1.f;
    else if constexpr (NBIT == 2) {
        const float m = ((c + 1) & 2) ? 1.f : (float)OPTIMAL_2BIT_HIGH;   // codes 1, 2: 1; 0, 3: HiMag
        return (c & 2) ? m : -m;
    } else return ((float)c - 7.5f) * (float)(1.0 / FourBit1sigma);
}

struct Stage1Block {            // per-block constants, computed by one thread (FP64)
    long base, frame0;          // window n0's first sample; its frame
    unsigned sif0, parity0;
    unsigned long long A, B;    // fringe slope and phase at base, 2^-64 turns
    unsigned long long slopeClk, offClk;   // the clock terms of the fractional-sample phase, 2^-64 turns
    float f, r, c2;             // the delay at window n0's centre less its integer shift, the rate and
                                // d2 there (samples): window n0 + k's delay is f + r kH + c2 (kH)^2 more
    float offPerFrac;           // the phase offset per sample of fractional delay (fracFreqOffset / fs), turns
};

// Window n0 + k's first sample (from the block's base) and fractional delay (samples).  Explicit
// round-to-nearest intrinsics, so every thread that computes it gets the same bits (no FMA
// contraction differing between call sites): the fold's shift and the correction's fraction agree.
__device__ __forceinline__ void windowDelay(const Stage1Block& b, int k, int H, int& off, float& frac) {
    const float kh = (float)(k * H);
    const float x = __fmaf_rn(kh, __fmaf_rn(kh, b.c2, b.r), b.f);
    const float sh = rintf(x);
    off = k * H + (int)sh;
    frac = __fsub_rn(x, sh);
}

__device__ __forceinline__ unsigned long long turnsToFixed(double x) {    // [0, 1) turns -> 2^-64 turns
    x -= floor(x);
    return x < 1.0 ? __double2ull_rz(x * 18446744073709551616.0) : 0ull;
}

// Block: FPB consecutive coarse windows n0 .. of one (antenna, band), 256 threads.
//  0. The delay model (TwoStageModel: a polynomial per antenna and subint) is evaluated in FP64 once,
//     at window n0's centre, by thread 0 (with the fringe polynomial at the block's first sample and
//     the clock terms).  Each window's integer shift and fractional delay follow in FP32 from that
//     value and the local rate: across a block (at most ~3500 samples) the delay changes by
//     rate x 3500 samples (0.035 at 10 us/s), so FP32 holds it to ~1e-9 samples.  Nothing is read
//     per coarse window.
//  1. The block's span of samples (from window n0's first to the last window's last) is decoded
//     once into shared memory: level, ALTSIGN by the absolute sample's parity, the fringe rotation
//     e^{-2 pi i (A s + B)} of window n0 (s from its first sample; with a continuous fringe model
//     every window's phase is the same function of the absolute sample).  Each sample is decoded
//     once a block, not once for each of the P x O windows it is in.
//  2. The fold: element t of window n is the sum over p < P of tap[p L1 + t] x span[its first
//     sample + p L1 + t], thread per (window, element), into the folded rows.
//  3. The L1-point FFT from the rows; bins c <= Nc get the fractional-sample correction at the coarse
//     channel's centre and the derotation e^{-2 pi i c n H / L1}, added in fixed point (one sincos),
//     into a tile [c][window], written out as (Nc + 1) runs of FPB consecutive coarse samples.
template<class FFT, int NBIT, unsigned ROW>
__launch_bounds__(FFT::max_threads_per_block)
__global__ void stage1_kernel(Geometry g1, PackedInput d, const float* __restrict__ taps, FracParams fp,
                              TwoStageModel model, int Nc, int P, int H, int spanMax, __half2* coarse, float scale) {
    using complex_type = typename FFT::value_type;
    constexpr unsigned L1 = size_of<FFT>::value, FPB = FFT::ffts_per_block, EPT = FFT::input_ept;
    static_assert(FFT::stride * EPT == L1, "a thread's elements cover the FFT");
    extern __shared__ __align__(alignof(float4)) unsigned char smem[];
    constexpr unsigned LOG2L1 = L1 == 32 ? 5 : L1 == 64 ? 6 : 7;
    __shared__ Stage1Block blk;
    const int band = blockIdx.y, ant = blockIdx.z, nwin = g1.nwin, nband = g1.nband;
    const int n0 = blockIdx.x * FPB, nLast = min(n0 + (int)FPB, nwin) - 1;
    const unsigned nthreads = blockDim.x * blockDim.y, tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid == 0) {
        const double* dm = model.delay + 3 * ant;
        const double xc = (double)n0 * H + 0.5 * (P * (int)L1 - 1);       // window n0's centre
        const double tau = dm[0] + xc * (dm[1] + xc * dm[2]);
        const double I = rint(tau);
        const long base = (long)n0 * H + (long)I, fs = d.ms.framesamples;
        const long frame0 = base >= 0 ? base / fs : -((-base + fs - 1) / fs);
        const double* fm = model.fringe + 3 * ((size_t)ant * nband + band);
        const double xb = (double)base;
        const int ab = ant * nband + band;
        const double clk = -fp.clockOffsets[ab] + fp.clockOffsetsDelta[ab] / 2;     // us
        blk = {base, frame0, (unsigned)(base - frame0 * fs), (unsigned)((d.datasamples + base) & 1),
               turnsToFixed(fm[1] + 2 * fm[2] * xb), turnsToFixed(fm[0] + xb * (fm[1] + xb * fm[2])),
               turnsToFixed(clk * fp.recordedBandwidth / Nc), turnsToFixed(clk * fp.fracFreqOffset[ab]),
               (float)(tau - I), (float)(dm[1] + 2 * dm[2] * xc), (float)dm[2],
               (float)(fp.fracFreqOffset[ab] / (2 * fp.recordedBandwidth))};
    }
    __syncthreads();
    int len;
    {
        float fr;
        windowDelay(blk, nLast - n0, H, len, fr);
        len += P * (int)L1;
    }
    if (len > spanMax) {                                             // the whole block
        if (tid == 0) printf("stage 1: span %d > %d samples\n", len, spanMax);
        return;
    }
    __half2* span = reinterpret_cast<__half2*>(smem);
    __half2* rows = span + ((spanMax + 3) & ~3);                      // [FPB][ROW]

    // 1. decode the span
    {
        const unsigned fs = (unsigned)d.ms.framesamples;
        const unsigned char* data = d.packed + (size_t)ant * d.bytesPerAnt + d.ms.payloadoffset;
        const bool* validFrames = d.validFrames + (size_t)ant * d.nframes;
        const bool alt = d.decodeFlags[ant * nband + band] & GPUDECODE_BAND_ALTSIGN;
        for (unsigned i = tid; i < (unsigned)len; i += nthreads) {
            const unsigned q = blk.sif0 + i;
            const unsigned k = __umulhi(q, d.fsMagic);                // frames from frame0
            const long frame = blk.frame0 + k;
            const unsigned sif = q - k * fs;
            float x = 0.f;
            if (frame >= 0 && frame < d.nframes && validFrames[frame]) {
                const unsigned bit = (sif * (unsigned)nband + (unsigned)band) * NBIT;
                const unsigned c = ((unsigned)__ldg(data + (size_t)frame * d.ms.framebytes + (bit >> 3)) >> (bit & 7)) &
                                   ((1u << NBIT) - 1);
                x = level<NBIT>(c);
                if (alt && ((blk.parity0 + i) & 1u)) x = -x;
            }
            const unsigned long long turns = blk.B + (unsigned long long)i * blk.A;
            const float t = (float)(int)(turns >> 32) * 2.3283064365386963e-10f;   // [-0.5, 0.5) turns
            float s, c;
            __sincosf(-(float)TWO_PI * t, &s, &c);
            span[i] = __floats2half2_rn(x * c, x * s);
        }
    }
    __syncthreads();

    // 2. the fold: thread (lane, t) folds element t of windows lane, lane + LANES, ... (WPT of them),
    // so that each tap is loaded once for WPT windows; a warp reads 32 consecutive samples of a window
    {
        constexpr unsigned LANES = FFT::max_threads_per_block / L1, WPT = FPB / LANES;
        static_assert(LANES * L1 == FFT::max_threads_per_block && WPT * LANES == FPB, "whole fold tasks");
        const unsigned t = tid % L1, lane = tid / L1;
        const __half2* w[WPT];
#pragma unroll
        for (unsigned k = 0; k < WPT; k++) {
            const int y = (int)(lane + k * LANES);
            int off = 0;
            float fr;
            if (n0 + y <= nLast) windowDelay(blk, y, H, off, fr);
            w[k] = span + off + t;
        }
        float2 acc[WPT];
#pragma unroll
        for (unsigned k = 0; k < WPT; k++) acc[k] = make_float2(0.f, 0.f);
        for (int p = 0; p < P; p++) {
            const float tap = __ldg(taps + p * L1 + t);
#pragma unroll
            for (unsigned k = 0; k < WPT; k++) {
                const float2 v = __half22float2(w[k][p * L1]);
                acc[k].x += tap * v.x;
                acc[k].y += tap * v.y;
            }
        }
#pragma unroll
        for (unsigned k = 0; k < WPT; k++) rows[(lane + k * LANES) * ROW + t] = __floats2half2_rn(acc[k].x, acc[k].y);
    }
    __syncthreads();

    // 3. the FFT and the coarse corrections
    const int n = n0 + (int)threadIdx.y;
    const bool active = n <= nLast;
    complex_type td[FFT::storage_size];
#pragma unroll
    for (unsigned i = 0; i < EPT; i++) {
        const float2 v = active ? __half22float2(rows[threadIdx.y * ROW + threadIdx.x + i * FFT::stride])
                                : make_float2(0.f, 0.f);
        td[i] = complex_type{v.x, v.y};
    }
    __syncthreads();                                                   // the FFT reuses shared memory
    FFT().execute(td, smem);
    __syncthreads();
    __half2* tile = reinterpret_cast<__half2*>(smem);                // [Nc + 1][FPB]
    // window validity: invalid frames decode to zero (a per-(antenna, subint) flag for the delay model's
    // validity would go here)
    const bool valid = active;
    ulonglong2 ph;
    {
        // FracRotateFixed's (offset, slope) with bigA = fractional delay (us) less the clock: the slope
        // fse bw / Nc = frac / L1 turns a channel; frac / L1 x 2^64 in FP32 (2^64 / L1 a power of two)
        int off;
        float fr;
        windowDelay(blk, (int)threadIdx.y, H, off, fr);
        float v = fr * blk.offPerFrac;
        v -= floorf(v);
        ph = make_ulonglong2(blk.offClk + (v < 1.f ? __float2ull_rz(v * 18446744073709551616.f) : 0ull),
                             blk.slopeClk + (unsigned long long)__float2ll_rn(fr * (float)(1ull << (64 - LOG2L1))));
    }
    const bool conjDC = fp.bandFlags[ant * nband + band] & FUSE_BAND_CONJ_DC;
#pragma unroll
    for (unsigned i = 0; i < FFT::output_ept; i++) {
        const unsigned c = threadIdx.x + i * FFT::stride;
        if (active && c <= (unsigned)Nc) {
            cuFloatComplex v = make_cuFloatComplex(td[i].x, td[i].y);
            if (valid) {
                // FracRotateFixed's phase (offset + c slope) less c n H / L1 turns (c n H mod L1 exactly:
                // L1 a power of two divides 2^32), in 2^-64 turns: one sincos
                const unsigned r = (c * (unsigned)n * (unsigned)H) & (L1 - 1);
                const unsigned long long turns = ph.x + (unsigned long long)c * ph.y -
                                                 ((unsigned long long)r << (64 - LOG2L1));
                const float tt = (float)(int)(turns >> 32) * 2.3283064365386963e-10f;
                cuFloatComplex cr;
                __sincosf((float)TWO_PI * tt, &cr.y, &cr.x);
                if (c == 0 && conjDC) v.y = -v.y;
                v = cuCmulf(v, cr);
            } else {
                v = make_cuFloatComplex(0.f, 0.f);
            }
            tile[c * FPB + threadIdx.y] = __floats2half2_rn(v.x * scale, v.y * scale);
        }
    }
    __syncthreads();
    const size_t out0 = ((size_t)ant * nband + band) * (Nc + 1);
    for (unsigned u = tid; u < (unsigned)(Nc + 1) * FPB; u += nthreads) {
        const unsigned c = u / FPB, y = u % FPB;
        if (n0 + (int)y <= nLast) coarse[(out0 + c) * nwin + n0 + y] = tile[c * FPB + y];
    }
}

template<unsigned L1, int NBIT>
static void launch(const TwoStagePlan& t, const Geometry& g1, const PreVariant& pre, const FracParams& p,
                   const TwoStageModel& m, __half2* coarse, float scale, cudaStream_t st) {
    using C = Stage1FFT<L1>;
    using FFT = typename C::FFT;
    constexpr unsigned FPB = C::FPB;
    const int spanMax = (int)(FPB - 1) * t.H + t.P * (int)L1 + SPAN_SLACK;
    const size_t work = (size_t)((spanMax + 3) & ~3) * sizeof(__half2) + (size_t)FPB * C::ROW * sizeof(__half2);
    const size_t smem = std::max({work, (size_t)FFT::shared_memory_size, (size_t)(t.Nc + 1) * FPB * sizeof(__half2)});
    auto kernel = stage1_kernel<FFT, NBIT, C::ROW>;
    static size_t set = 0;
    if (smem > set) {
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        set = smem;
    }
    const dim3 grid((g1.nwin + FPB - 1) / FPB, g1.nband, g1.nant);
    kernel<<<grid, FFT::block_dim, smem, st>>>(g1, pre.pk, pre.window, p, m, t.Nc, t.P, t.H, spanMax, coarse, scale);
    CHECK_CUDA(cudaGetLastError());
}

template<unsigned L1>
static void byBits(const TwoStagePlan& t, const Geometry& g1, const PreVariant& pre, const FracParams& p,
                   const TwoStageModel& m, __half2* coarse, float scale, cudaStream_t st) {
    switch (pre.pk.ms.nbit) {
    case 1: launch<L1, 1>(t, g1, pre, p, m, coarse, scale, st); break;
    case 2: launch<L1, 2>(t, g1, pre, p, m, coarse, scale, st); break;
    case 4: launch<L1, 4>(t, g1, pre, p, m, coarse, scale, st); break;
    default: fprintf(stderr, "two-stage: 1, 2 or 4 bits\n"); exit(2);
    }
}

bool twoStage1Supported(int L1, int nbit) {
    return (L1 == 32 || L1 == 64 || L1 == 128) && (nbit == 1 || nbit == 2 || nbit == 4);
}

void launchStage1(const TwoStagePlan& t, const Geometry& g1, const PreVariant& pre, const FracParams& p,
                  const TwoStageModel& m, __half2* coarse, float scale1, cudaStream_t st) {
    switch (t.L1) {
    case 32: byBits<32>(t, g1, pre, p, m, coarse, scale1, st); break;
    case 64: byBits<64>(t, g1, pre, p, m, coarse, scale1, st); break;
    case 128: byBits<128>(t, g1, pre, p, m, coarse, scale1, st); break;
    default: fprintf(stderr, "two-stage: stage-1 FFT of 32, 64 or 128 points, not %d\n", t.L1); exit(2);
    }
}
