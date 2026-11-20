// twostage2.cu - two-stage channelisation, stage 2: an FFT of each coarse channel's series
// (twostage.cuh).
//
// This software contains source code provided by NVIDIA Corporation (cuFFTDx usage).
#include <cufftdx.hpp>
#include "twostage.cuh"

#ifndef FUSE_SM
#error "compile with -DFUSE_SM=<arch, e.g. 750 for sm_75>"
#endif

using namespace cufftdx;

template<unsigned N2>
struct Stage2FFT {
    using Base = decltype(Size<N2>() + Precision<float>() + Type<fft_type::c2c>() +
                          Direction<fft_direction::forward>() + SM<FUSE_SM>() + Block());
    static constexpr unsigned E = Base::elements_per_thread;
    static constexpr unsigned FPB = (256 / (N2 / E)) > 0 ? 256 / (N2 / E) : 1;
    using FFT = decltype(Base() + ElementsPerThread<E>() + FFTsPerBlock<FPB>());
};

// FFT f of the launch is (antenna, band, coarse channel c, stage-2 window m), m fastest: its input,
// coarse samples m N2 .. m N2 + N2 - 1 of (ant, band, c), is the f-th run of N2 in the coarse
// buffer.  Thread x holds elements x + i stride before and after the FFT, so the loads and the
// stores of the kept bins are coalesced.
template<class FFT>
__launch_bounds__(FFT::max_threads_per_block)
__global__ void stage2_kernel(int nband, int nwin2, int Nc, int K, int nchan, unsigned long long nfft,
                              const __half2* coarse, __half2* out, float scale) {
    using complex_type = typename FFT::value_type;
    constexpr unsigned N2 = size_of<FFT>::value;
    static_assert(FFT::stride * FFT::input_ept == N2, "a thread's elements cover the FFT");
    extern __shared__ __align__(alignof(float4)) unsigned char smem[];
    const unsigned long long f = (unsigned long long)blockIdx.x * FFT::ffts_per_block + threadIdx.y;
    const bool active = f < nfft;
    complex_type td[FFT::storage_size];
    if (active) {
        const __half2* src = coarse + f * N2;
#pragma unroll
        for (unsigned i = 0; i < FFT::input_ept; i++) {
            const float2 v = __half22float2(src[threadIdx.x + i * FFT::stride]);
            td[i] = complex_type{v.x, v.y};
        }
    } else {
#pragma unroll
        for (unsigned i = 0; i < FFT::input_ept; i++) td[i] = complex_type{0.f, 0.f};
    }

    FFT().execute(td, smem);

    if (!active) return;
    const int m = (int)(f % nwin2);
    const unsigned long long abc = f / nwin2;                         // (ant, band, c)
    const int c = (int)(abc % (Nc + 1));
    const unsigned long long ab = abc / (Nc + 1);
    const int band = (int)(ab % nband), ant = (int)(ab / nband);
    __half2* dst = out + (((size_t)ant * nwin2 + m) * nband + band) * nchan;
#pragma unroll
    for (unsigned i = 0; i < FFT::output_ept; i++) {
        const int b = (int)(threadIdx.x + i * FFT::stride);
        const int k = b < (int)N2 / 2 ? b : b - (int)N2;            // fine bin, from the coarse centre
        const int j = c * K + k;
        if (k >= -K / 2 && k < K / 2 && j >= 0 && j < nchan)
            dst[j] = __floats2half2_rn(td[i].x * scale, td[i].y * scale);
    }
}

template<unsigned N2>
static void launch(const TwoStagePlan& t, const Geometry& g1, int nwin2, const __half2* coarse, __half2* out,
                   float scale, cudaStream_t st) {
    using FFT = typename Stage2FFT<N2>::FFT;
    auto kernel = stage2_kernel<FFT>;
    static bool set = false;
    if (!set) {
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)FFT::shared_memory_size));
        set = true;
    }
    const unsigned long long nfft = (unsigned long long)g1.nant * g1.nband * (t.Nc + 1) * nwin2;
    const unsigned blocks = (unsigned)((nfft + FFT::ffts_per_block - 1) / FFT::ffts_per_block);
    kernel<<<blocks, FFT::block_dim, FFT::shared_memory_size, st>>>(g1.nband, nwin2, t.Nc, t.K, t.nchan(), nfft,
                                                                    coarse, out, scale);
    CHECK_CUDA(cudaGetLastError());
}

bool twoStage2Supported(int N2) { return N2 == 128 || N2 == 256 || N2 == 512 || N2 == 1024 || N2 == 2048 || N2 == 4096 || N2 == 8192; }

void launchStage2(const TwoStagePlan& t, const Geometry& g1, int nwin2, const __half2* coarse, __half2* out,
                  float scale2, cudaStream_t st) {
    if (g1.nwin != nwin2 * t.N2) { fprintf(stderr, "two-stage: %d coarse samples, not %d x %d\n", g1.nwin, nwin2, t.N2); exit(2); }
    switch (t.N2) {
    case 128: launch<128>(t, g1, nwin2, coarse, out, scale2, st); break;
    case 256: launch<256>(t, g1, nwin2, coarse, out, scale2, st); break;
    case 512: launch<512>(t, g1, nwin2, coarse, out, scale2, st); break;
    case 1024: launch<1024>(t, g1, nwin2, coarse, out, scale2, st); break;
    case 2048: launch<2048>(t, g1, nwin2, coarse, out, scale2, st); break;
    case 4096: launch<4096>(t, g1, nwin2, coarse, out, scale2, st); break;
    case 8192: launch<8192>(t, g1, nwin2, coarse, out, scale2, st); break;
    default: fprintf(stderr, "two-stage: stage-2 FFT of 128 to 8192 points, not %d\n", t.N2); exit(2);
    }
}
