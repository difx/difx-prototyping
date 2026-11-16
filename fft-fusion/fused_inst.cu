// fused_inst.cu - the fused cuFFTDx kernel: pre-FFT stage -> FFT -> fractional-sample correction.
//
// This software contains source code provided by NVIDIA Corporation (cuFFTDx usage).
//
// Compiled once per FFT size (the Makefile passes -DFUSE_N=<size> -DFUSE_SM=<arch*10>), because
// cuFFTDx fixes the size at compile time. Each compiled size registers its variants (FFTs per
// block) in fusedRegistry() when the program starts.
#include <cufftdx.hpp>
#include "pipelines.cuh"

#ifndef FUSE_N
#error "compile with -DFUSE_N=<FFT size>"
#endif
#ifndef FUSE_SM
#error "compile with -DFUSE_SM=<arch, e.g. 750 for sm_75>"
#endif

using namespace cufftdx;

using FFTBase = decltype(Size<FUSE_N>() + Precision<float>() + Type<fft_type::c2c>() +
                         Direction<fft_direction::forward>() + SM<FUSE_SM>() + Block());
template<unsigned FPB>
using FFTn = decltype(FFTBase() + ElementsPerThread<FFTBase::elements_per_thread>() + FFTsPerBlock<FPB>());

// One block does FFT::ffts_per_block spectra; spectrum f of the launch is (ant0 + f / (nwin *
// nband), window, band) with band fastest, so a block's spectra are contiguous in memory.
// Thread (x, y) holds elements x, x + stride, x + 2 stride, ... of spectrum y, before and after
// the FFT (natural order), which keeps the global loads and stores coalesced.
// Pre is the pre-FFT stage (PreLoad: FP32 or FP16 storage, optional window and overlap), converted
// to FP32 on load; the FFT is FP32.  Post is the post-FFT stage (FracRotateFixed, stages.cuh).
//
// With CROSSPOL the cross-pol autocorrelations are formed here too: bands 2f and 2f+1 of a window
// are spectra y and y+1 of the same block (band fastest, even nband, even FFTs per block), and
// thread (x, y) and (x, y+1) hold the same channels, so they swap values through shared memory.
template<class FFT, class Pre, class Post, bool CROSSPOL>
__launch_bounds__(FFT::max_threads_per_block)
__global__ void fused_fft_frac(Geometry g, Pre pre, Post post, cuFloatComplex* out,
                               int ant0, unsigned long long nfft) {
    using complex_type = typename FFT::value_type;
    extern __shared__ __align__(alignof(float4)) unsigned char smem[];
    constexpr unsigned N = size_of<FFT>::value;

    const unsigned long long f = (unsigned long long)blockIdx.x * FFT::ffts_per_block + threadIdx.y;
    const bool active = f < nfft;     // the last block may be partly empty; all threads must
    int ant = ant0, win = 0, band = 0; // still take part in execute(), which synchronises
    if (active) {
        band = f % g.nband;
        const unsigned long long t = f / g.nband;
        win = t % g.nwin;
        ant = ant0 + (int)(t / g.nwin);
    }

    complex_type td[FFT::storage_size];
#pragma unroll
    for (unsigned i = 0; i < FFT::input_ept; i++) {
        const unsigned s = threadIdx.x + i * FFT::stride;
        if (active && s < N) {
            const cuFloatComplex v = pre(g, ant, win, band, s);
            td[i] = complex_type{v.x, v.y};
        }
    }

    FFT().execute(td, smem);

    const bool valid = active && post.windowValid(g, ant, win);
#pragma unroll
    for (unsigned i = 0; i < FFT::output_ept; i++) {
        const unsigned k = threadIdx.x + i * FFT::stride;
        if (active && k < (unsigned)g.nchan) {
            cuFloatComplex v = make_cuFloatComplex(td[i].x, td[i].y);
            if (valid)
                v = post.apply(g, ant, win, band, k, v);
            out[g.index(ant, win, band, k)] = v;
            if (CROSSPOL)
                td[i] = complex_type{v.x, v.y};
        }
    }

    if constexpr (CROSSPOL) {
        static_assert(FFT::ffts_per_block % 2 == 0, "cross-pol in the block needs pairs of spectra");
        cuFloatComplex* xs = reinterpret_cast<cuFloatComplex*>(smem);
        const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
        const unsigned partner = (threadIdx.y ^ 1u) * blockDim.x + threadIdx.x;
        __syncthreads();                    // the FFT is done with shared memory
#pragma unroll
        for (unsigned i = 0; i < FFT::output_ept; i++) {
            const unsigned k = threadIdx.x + i * FFT::stride;
            xs[tid] = make_cuFloatComplex(td[i].x, td[i].y);
            __syncthreads();
            if (valid && (threadIdx.y & 1u) == 0 && k < (unsigned)g.nchan)
                post.crossPolPair(g, ant, band, band + 1, k, make_cuFloatComplex(td[i].x, td[i].y), xs[partner]);
            __syncthreads();
        }
    }
}

template<class FFT, class Pre, class Post, bool CROSSPOL>
static void launchKernel(const Geometry& g, const Pre& pre, const FracParams& p, cuFloatComplex* out,
                         int ant0, int nant, cudaStream_t st) {
    auto kernel = fused_fft_frac<FFT, Pre, Post, CROSSPOL>;
    constexpr unsigned xsBytes = CROSSPOL ? FFT::max_threads_per_block * sizeof(cuFloatComplex) : 0;
    constexpr unsigned smemBytes = FFT::shared_memory_size > xsBytes ? FFT::shared_memory_size : xsBytes;
    static bool attributeSet = false;
    if (!attributeSet) {
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
        attributeSet = true;
    }
    const unsigned long long nfft = (unsigned long long)nant * g.nwin * g.nband;
    const unsigned long long blocks = (nfft + FFT::ffts_per_block - 1) / FFT::ffts_per_block;
    Post post;
    post.p = p;
    kernel<<<(unsigned)blocks, FFT::block_dim, smemBytes, st>>>(g, pre, post, out, ant0, nfft);
    CHECK_CUDA(cudaGetLastError());
}

template<class FFT, class Pre, class Post>
static bool launchOne(const Geometry& g, const Pre& pre, const FracParams& p, cuFloatComplex* out,
                      int ant0, int nant, cudaStream_t st) {
    if constexpr (Post::doAutocorr && FFT::ffts_per_block % 2 == 0) {
        if (p.crossPol && p.crossPolInBlock) {
            launchKernel<FFT, Pre, Post, true>(g, pre, p, out, ant0, nant, st);
            return true;
        }
    }
    launchKernel<FFT, Pre, Post, false>(g, pre, p, out, ant0, nant, st);
    return false;
}

// The fused kernel always uses the fixed-point rotator (no FP64 per element).
template<class FFT>
static bool launchVariant(const Geometry& g, const PreVariant& pre, const FracParams& p, bool doAutocorr,
                          cuFloatComplex* out, int ant0, int nant, cudaStream_t st) {
    bool crossDone = false;
    withPre(pre, [&](auto stage) {
        using Pre = decltype(stage);
        if (doAutocorr) crossDone = launchOne<FFT, Pre, FracRotateFixed<true>>(g, stage, p, out, ant0, nant, st);
        else            crossDone = launchOne<FFT, Pre, FracRotateFixed<false>>(g, stage, p, out, ant0, nant, st);
    });
    return crossDone;
}

template<unsigned FPB>
static void addVariant(std::vector<FusedVariant>& v) {
    // only FFTs-per-block counts that fit in a block of 1024 threads
    if constexpr (FFTn<1>::block_dim.x * FPB <= 1024) {
        using FFT = FFTn<FPB>;
        static_assert(!FFT::requires_workspace, "FFT size needs a workspace; only powers of two are supported");
        for (const auto& e : v)
            if (e.fpb == FPB) return;
        v.push_back({FPB, FFT::elements_per_thread, FFT::max_threads_per_block, FFT::shared_memory_size,
                     FPB == FFTBase::suggested_ffts_per_block, &launchVariant<FFT>});
    }
}

static const int registered = [] {
    auto& v = fusedRegistry()[FUSE_N];
    addVariant<FFTBase::suggested_ffts_per_block>(v);
    addVariant<1>(v); addVariant<2>(v); addVariant<4>(v); addVariant<8>(v);
    addVariant<16>(v); addVariant<32>(v);
    return 0;
}();
