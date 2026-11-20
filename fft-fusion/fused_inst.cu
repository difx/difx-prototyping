// fused_inst.cu - the fused cuFFTDx kernel: pre-FFT stage -> FFT -> fractional-sample correction.
//
// This software contains source code provided by NVIDIA Corporation (cuFFTDx usage).
//
// Compiled once per FFT size (the Makefile passes -DFUSE_N=<size> -DFUSE_SM=<arch*10>), because
// cuFFTDx fixes the size at compile time. Each compiled size registers its variants (FFTs per
// block) in fusedRegistry() when the program starts.
#include <algorithm>
#include <type_traits>
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
// E elements per thread (cuFFTDx's suggestion E0, or E0 / 2: measured faster for these kernels,
// see README.md) and FPB FFTs per block
constexpr unsigned E0 = FFTBase::elements_per_thread;
template<unsigned E, unsigned FPB>
using FFTn = decltype(FFTBase() + ElementsPerThread<E>() + FFTsPerBlock<FPB>());

// One block does FFT::ffts_per_block spectra; spectrum f of the launch is (ant0 + f / (nwin *
// nband), window, band) with band fastest, so a block's spectra are contiguous in memory.
// Thread (x, y) holds elements x, x + stride, x + 2 stride, ... of spectrum y, before and after
// the FFT (natural order), which keeps the global loads and stores coalesced.
// Pre is the pre-FFT stage (PreLoad: FP32 or FP16 storage, optional window and overlap, converted
// to FP32 on load; PreDecode, PreDecodeLean: from the packed data, decode.cuh), prepared once per
// thread (all of a thread's samples belong to one spectrum); the FFT is FP32.  Post is the
// post-FFT stage (FracRotateFixed, stages.cuh).  The launch has fewer than 2^32 spectra
// (fusebench checks), so the index arithmetic is 32-bit.
//
// Out is where the spectra go: OutF32 (FP32, Geometry::index; an invalid window's spectrum is
// written unrotated, and the XMAC skips it by its flag) or OutHalf (FP16, scaled, in the same
// order with nchan channels a spectrum, XmacShape::halfIndex; invalid windows as zeros), each
// store of consecutive channels by consecutive threads.
//
// With CROSSPOL the cross-pol autocorrelations are formed here too: bands 2f and 2f+1 of a window
// are spectra y and y+1 of the same block (band fastest, even nband, even FFTs per block), and
// thread (x, y) and (x, y+1) hold the same channels, so they swap values through shared memory.
struct OutF32 {
    cuFloatComplex* p;
    __device__ __forceinline__ void put(const Geometry& g, int ant, int win, int band, unsigned k, cuFloatComplex v,
                                        bool) const {
        p[g.index(ant, win, band, k)] = v;
    }
};
struct OutHalf {
    HalfOut o;
    __device__ __forceinline__ void put(const Geometry&, int ant, int win, int band, unsigned k, cuFloatComplex v,
                                        bool valid) const {
        o.h[o.s.halfIndex(ant, win, band, k)] = valid ? __floats2half2_rn(v.x * o.scale, v.y * o.scale)
                                                      : __float2half2_rn(0.f);
    }
};

template<class FFT, class Pre, class Post, bool CROSSPOL, class Out>
__device__ __forceinline__ void fusedBody(const Geometry& g, const Pre& pre, const Post& post,
                                          const Out& out, int ant0, unsigned long long nfft) {
    using complex_type = typename FFT::value_type;
    extern __shared__ __align__(alignof(float4)) unsigned char smem[];
    constexpr unsigned N = size_of<FFT>::value;

    const unsigned f = blockIdx.x * FFT::ffts_per_block + threadIdx.y;
    const bool active = f < nfft;     // the last block may be partly empty; all threads must
    int ant = ant0, win = 0, band = 0; // still take part in execute(), which synchronises
    if (active) {
        band = f % (unsigned)g.nband;
        const unsigned t = f / (unsigned)g.nband;
        win = t % (unsigned)g.nwin;
        ant = ant0 + (int)(t / (unsigned)g.nwin);
    }

    // thread x holds samples x, x + stride, ...: all N of them, one per (x, i), so that only the
    // last block's spare spectra need a test (once per thread, not per sample)
    static_assert(FFT::stride * FFT::input_ept == N, "a thread's samples cover the FFT exactly");
    complex_type td[FFT::storage_size];
    if (active) {
        typename Pre::State st = pre.prepare(g, ant, win, band, threadIdx.x, FFT::stride);
#pragma unroll
        for (unsigned i = 0; i < FFT::input_ept; i++) {
            const cuFloatComplex v = pre.at(st, g, ant, win, band, threadIdx.x + i * FFT::stride);
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
            out.put(g, ant, win, band, k, v, valid);
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

// The same, writing the tensor-core XMAC's layout (TensorOut, pipelines.cuh) instead of FP32
// spectra.  A block takes W consecutive windows of one (antenna, band), FFTs-per-block at a time
// (W / FFTs-per-block passes), so that it writes each channel's run of W windows in each plane as
// whole 32-byte sectors (W = 16) or halves (W = 8) whatever its FFTs per block.  After each pass's
// fractional-sample rotation the threads put their channels' values, scaled, as FP16 into a
// shared-memory tile [plane][channel][window], kept apart from the FFT's shared memory when there
// are several passes; then the block writes the runs with 16-byte stores.  Block b of the launch is
// (antenna, group of W windows, band), band fastest, so that the blocks of a window group's bands,
// which read the same packed bytes, run together (with the band slowest the packed data was read
// from DRAM 6.6 times over).  The tile's rows are padded by one 32-bit word, so that a warp's
// channels fall in different banks (unpadded, 16 windows to a row put 16 channels in 4 banks).
template<class FFT>
struct TensorPlan {
    static constexpr unsigned N = size_of<FFT>::value, FPB = FFT::ffts_per_block;
    // shared memory a block may use: sm_80/87 163 KB, sm_90/100/103 227 KB, sm_75 64 KB, others
    // (sm_86/89, sm_120/121: the GeForce and workstation parts) 99 KB.  Not "sm_90 and up": the
    // RTX PRO 6000 Blackwell (sm_120) has 99 KB, and a tile planned for 227 KB failed to launch.
    static constexpr unsigned maxSmem = FUSE_SM == 900 || FUSE_SM == 1000 || FUSE_SM == 1030 ? 227u * 1024
                                      : FUSE_SM == 800 || FUSE_SM == 870 ? 163u * 1024
                                      : FUSE_SM >= 860 ? 99u * 1024 : 64u * 1024;
    static constexpr unsigned row(unsigned w) { return w >= 2 ? w + 2 : w; }
    // the tile for W windows; the packed data is real, so at most N / 2 channels
    static constexpr unsigned tile(unsigned w) { return 2u * (N / 2) * row(w) * (unsigned)sizeof(__half); }
    static constexpr unsigned fft = (FFT::shared_memory_size + 15) / 16 * 16;
    static constexpr unsigned W = FPB >= 16 ? FPB
                                : fft + tile(16) <= maxSmem ? 16
                                : FPB <= 8 && fft + tile(8) <= maxSmem ? 8 : FPB;
    static constexpr bool separate = W > FPB;               // several passes: the tile apart
    static constexpr unsigned ROW = row(W);
    static constexpr unsigned tileOffset = separate ? fft : 0;
    static constexpr unsigned smem = separate ? fft + tile(W) : (fft > tile(W) ? fft : tile(W));
};

template<class FFT, class Pre, class Post>
__device__ __forceinline__ void tensorBody(const Geometry& g, const Pre& pre, const Post& post, const TensorOut& o,
                                           int ant0) {
    using complex_type = typename FFT::value_type;
    using P = TensorPlan<FFT>;
    extern __shared__ __align__(alignof(float4)) unsigned char smem[];
    constexpr unsigned N = P::N, FPB = P::FPB, W = P::W, ROW = P::ROW;
    static_assert(FFT::stride * FFT::input_ept == N, "a thread's samples cover the FFT exactly");
    const unsigned nwb = (g.nwin + W - 1) / W;
    const int band = blockIdx.x % (unsigned)g.nband;
    const unsigned aw = blockIdx.x / (unsigned)g.nband;
    const unsigned wb = aw % nwb;
    const int ant = ant0 + (int)(aw / nwb);
    __half* tile = reinterpret_cast<__half*>(smem + P::tileOffset);       // [plane][channel][window]

#pragma unroll 1
    for (unsigned pass = 0; pass < W / FPB; pass++) {
        const unsigned wl = pass * FPB + threadIdx.y;                     // window within the block
        const int win = wb * W + wl;
        const bool active = win < g.nwin;
        complex_type td[FFT::storage_size];
        if (active) {
            typename Pre::State st = pre.prepare(g, ant, win, band, threadIdx.x, FFT::stride);
#pragma unroll
            for (unsigned i = 0; i < FFT::input_ept; i++) {
                const cuFloatComplex v = pre.at(st, g, ant, win, band, threadIdx.x + i * FFT::stride);
                td[i] = complex_type{v.x, v.y};
            }
        }
        FFT().execute(td, smem);
        const bool valid = active && post.windowValid(g, ant, win);
        if constexpr (!P::separate)
            __syncthreads();                                   // the FFT is done with shared memory
#pragma unroll
        for (unsigned i = 0; i < FFT::output_ept; i++) {
            const unsigned k = threadIdx.x + i * FFT::stride;
            if (k < (unsigned)g.nchan) {
                cuFloatComplex v = make_cuFloatComplex(0.f, 0.f);   // invalid windows, those past the end
                if (valid)
                    v = post.apply(g, ant, win, band, k, make_cuFloatComplex(td[i].x, td[i].y));
                tile[k * ROW + wl] = __float2half_rn(v.x * o.scale);
                tile[((unsigned)g.nchan + k) * ROW + wl] = __float2half_rn(v.y * o.scale);
            }
        }
    }
    __syncthreads();
    constexpr unsigned G = W < 8 ? W : 8, NC = W / G;         // G windows a store, NC stores a run
    const unsigned nthreads = blockDim.x * blockDim.y, tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int f = band >> 1, r = 2 * ant + (band & 1);
    for (unsigned u = tid; u < 2u * g.nchan * NC; u += nthreads) {
        const unsigned chunk = u % NC, pk = u / NC;             // pk = plane * nchan + channel
        const unsigned plane = pk / g.nchan, k = pk % g.nchan;
        const unsigned w0 = wb * W + chunk * G;
        if (w0 >= (unsigned)o.L.ntb * 16) continue;             // past the padded subint
        const __half* src = tile + pk * ROW + chunk * G;
        // read as 32-bit words (the rows are padded), stored as one vector (__stcg: without it the
        // compiler split the 16-byte store into four 4-byte ones, 25% slower); cached in L2 only
        const unsigned* w32 = reinterpret_cast<const unsigned*>(src);
        __half* dst = o.h + o.L.index(o.s, f, k, w0 / 16, plane, r, w0 % 16);
        if constexpr (G == 1) *dst = *src;
        else if constexpr (G == 2) *reinterpret_cast<__half2*>(dst) = *reinterpret_cast<const __half2*>(src);
        else if constexpr (G == 4) __stcg(reinterpret_cast<uint2*>(dst), make_uint2(w32[0], w32[1]));
        else __stcg(reinterpret_cast<uint4*>(dst), make_uint4(w32[0], w32[1], w32[2], w32[3]));
    }
}

template<class FFT, class Pre, class Post>
__launch_bounds__(FFT::max_threads_per_block)
__global__ void fused_fft_tensor(Geometry g, Pre pre, Post post, TensorOut o, int ant0) {
    tensorBody<FFT, Pre, Post>(g, pre, post, o, ant0);
}

// The pre-FFT stages the XMAC outputs (tensor layout, FP16 spectra) are compiled for: the lean decode
// of 1-, 2- and 4-bit data (each one more kernel per variant and FFT length, so not 8 bits).
template<class Pre> constexpr bool xmacDecode = std::is_same<Pre, PreDecodeLean<1>>::value ||
                                                std::is_same<Pre, PreDecodeLean<2>>::value ||
                                                std::is_same<Pre, PreDecodeLean<4>>::value;

template<class FFT>
static void launchTensorVariant(const Geometry& g, const PreVariant& pre, const FracParams& p, const TensorOut& o,
                                int ant0, int nant, cudaStream_t st) {
    withPre(pre, [&](auto stage) {
        using Pre = decltype(stage);
        if constexpr (xmacDecode<Pre>) {
            using Post = FracRotateFixed<false>;
            auto kernel = fused_fft_tensor<FFT, Pre, Post>;
            using TP = TensorPlan<FFT>;
            constexpr unsigned smemBytes = TP::smem;
            if (2 * g.nchan > (int)TP::N) { fprintf(stderr, "tensor-layout output: real data only\n"); exit(2); }
            static int regs = -1;
            if (regs < 0) {
                CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
                cudaFuncAttributes attr;
                CHECK_CUDA(cudaFuncGetAttributes(&attr, kernel));
                regs = attr.numRegs;
            }
            fusedLastRegisters = regs;
            Post post;
            post.p = p;
            const unsigned nwb = (g.nwin + TP::W - 1) / TP::W;
            kernel<<<(unsigned)nant * g.nband * nwb, FFT::block_dim, smemBytes, st>>>(g, stage, post, o, ant0);
            CHECK_CUDA(cudaGetLastError());
        } else {
            fprintf(stderr, "the tensor-layout output is compiled for the 1-, 2- and 4-bit lean decode only\n");
            exit(2);
        }
    });
}

// The kernel: __launch_bounds__ with the block size only, so the compiler chooses the registers.
// Not __launch_bounds__(threads, 1): with an explicit minimum of one block per SM the compiler
// used more registers (sm_80: 86-96 instead of 64-72 at cuFFTDx's elements per thread, 54-56
// instead of 48 at half; sm_75: 92-96 instead of 64-80), which lowered occupancy and made some
// variants up to 56% slower (README.md).
template<class FFT, class Pre, class Post, bool CROSSPOL>
__launch_bounds__(FFT::max_threads_per_block)
__global__ void fused_fft_frac(Geometry g, Pre pre, Post post, cuFloatComplex* out,
                               int ant0, unsigned long long nfft) {
    fusedBody<FFT, Pre, Post, CROSSPOL>(g, pre, post, OutF32{out}, ant0, nfft);
}

// The same writing FP16 spectra (HalfOut) for the staged tensor XMAC.
template<class FFT, class Pre, class Post>
__launch_bounds__(FFT::max_threads_per_block)
__global__ void fused_fft_half(Geometry g, Pre pre, Post post, OutHalf out, int ant0, unsigned long long nfft) {
    fusedBody<FFT, Pre, Post, false>(g, pre, post, out, ant0, nfft);
}

// The same with the registers capped (--regcap): REGCAP is a number of threads per SM to make
// room for, so __launch_bounds__ asks for at least REGCAP / block threads blocks per SM.  The
// request is clamped to what an SM holds (sm_80/87, sm_90/100/103: 2048 threads in 32 blocks;
// sm_75: 1024 in 16; others, sm_86/89 and sm_120/121 among them: taken as 1536 in 16), so where
// the SM holds fewer threads than REGCAP (sm_75) the cap asks for no more than the SM's own limit.
constexpr bool BIG_SM = FUSE_SM == 800 || FUSE_SM == 870 || FUSE_SM == 900 || FUSE_SM == 1000 || FUSE_SM == 1030;
constexpr unsigned MAX_BLOCKS_PER_SM = BIG_SM ? 32 : 16;
constexpr unsigned MAX_THREADS_PER_SM = BIG_SM ? 2048 : FUSE_SM == 750 ? 1024 : 1536;
constexpr unsigned minBlocks(unsigned regcap, unsigned threads) {
    const unsigned want = (regcap + threads - 1) / threads;
    const unsigned fit = MAX_THREADS_PER_SM / threads < MAX_BLOCKS_PER_SM ? MAX_THREADS_PER_SM / threads : MAX_BLOCKS_PER_SM;
    return want < fit ? want : (fit ? fit : 1);
}
template<class FFT, class Pre, class Post, bool CROSSPOL, unsigned REGCAP>
__launch_bounds__(FFT::max_threads_per_block, minBlocks(REGCAP, FFT::max_threads_per_block))
__global__ void fused_fft_frac_capped(Geometry g, Pre pre, Post post, cuFloatComplex* out,
                                      int ant0, unsigned long long nfft) {
    fusedBody<FFT, Pre, Post, CROSSPOL>(g, pre, post, OutF32{out}, ant0, nfft);
}

template<class FFT, class Pre, class Post, bool CROSSPOL, unsigned REGCAP>
static void launchKernel(const Geometry& g, const Pre& pre, const FracParams& p, cuFloatComplex* out,
                         int ant0, int nant, cudaStream_t st) {
    auto kernel = [] {
        if constexpr (REGCAP == 0) return fused_fft_frac<FFT, Pre, Post, CROSSPOL>;
        else                       return fused_fft_frac_capped<FFT, Pre, Post, CROSSPOL, REGCAP>;
    }();
    constexpr unsigned xsBytes = CROSSPOL ? FFT::max_threads_per_block * sizeof(cuFloatComplex) : 0;
    constexpr unsigned smemBytes = FFT::shared_memory_size > xsBytes ? FFT::shared_memory_size : xsBytes;
    static int regs = -1;
    if (regs < 0) {
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
        cudaFuncAttributes attr;
        CHECK_CUDA(cudaFuncGetAttributes(&attr, kernel));
        regs = attr.numRegs;
    }
    fusedLastRegisters = regs;
    const unsigned long long nfft = (unsigned long long)nant * g.nwin * g.nband;
    const unsigned long long blocks = (nfft + FFT::ffts_per_block - 1) / FFT::ffts_per_block;
    Post post;
    post.p = p;
    kernel<<<(unsigned)blocks, FFT::block_dim, smemBytes, st>>>(g, pre, post, out, ant0, nfft);
    CHECK_CUDA(cudaGetLastError());
}

template<class FFT, class Pre, class Post, unsigned REGCAP>
static bool launchOne(const Geometry& g, const Pre& pre, const FracParams& p, cuFloatComplex* out,
                      int ant0, int nant, cudaStream_t st) {
    if constexpr (Post::doAutocorr && FFT::ffts_per_block % 2 == 0) {
        if (p.crossPol && p.crossPolInBlock) {
            launchKernel<FFT, Pre, Post, true, REGCAP>(g, pre, p, out, ant0, nant, st);
            return true;
        }
    }
    launchKernel<FFT, Pre, Post, false, REGCAP>(g, pre, p, out, ant0, nant, st);
    return false;
}

// The fused kernel always uses the fixed-point rotator (no FP64 per element).
template<class FFT, unsigned REGCAP>
static bool launchVariant(const Geometry& g, const PreVariant& pre, const FracParams& p, bool doAutocorr,
                          cuFloatComplex* out, int ant0, int nant, cudaStream_t st) {
    bool crossDone = false;
    withPre(pre, [&](auto stage) {
        using Pre = decltype(stage);
        if (!doAutocorr)
            crossDone = launchOne<FFT, Pre, FracRotateFixed<false>, REGCAP>(g, stage, p, out, ant0, nant, st);
        else if constexpr (Pre::withAutocorr)
            crossDone = launchOne<FFT, Pre, FracRotateFixed<true>, REGCAP>(g, stage, p, out, ant0, nant, st);
        else {
            fprintf(stderr, "this pre-FFT stage is not compiled with the autocorrelations\n");
            exit(2);
        }
    });
    return crossDone;
}

template<class FFT>
static void launchHalfVariant(const Geometry& g, const PreVariant& pre, const FracParams& p, const HalfOut& o,
                              int ant0, int nant, cudaStream_t st) {
    withPre(pre, [&](auto stage) {
        using Pre = decltype(stage);
        if constexpr (xmacDecode<Pre>) {
            using Post = FracRotateFixed<false>;
            auto kernel = fused_fft_half<FFT, Pre, Post>;
            constexpr unsigned smemBytes = FFT::shared_memory_size;
            static int regs = -1;
            if (regs < 0) {
                CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
                cudaFuncAttributes attr;
                CHECK_CUDA(cudaFuncGetAttributes(&attr, kernel));
                regs = attr.numRegs;
            }
            fusedLastRegisters = regs;
            Post post;
            post.p = p;
            const unsigned long long nfft = (unsigned long long)nant * g.nwin * g.nband;
            const unsigned long long blocks = (nfft + FFT::ffts_per_block - 1) / FFT::ffts_per_block;
            kernel<<<(unsigned)blocks, FFT::block_dim, smemBytes, st>>>(g, stage, post, OutHalf{o}, ant0, nfft);
            CHECK_CUDA(cudaGetLastError());
        } else {
            fprintf(stderr, "the FP16 output is compiled for the 1-, 2- and 4-bit lean decode only\n");
            exit(2);
        }
    });
}

template<unsigned E, unsigned FPB, unsigned REGCAP = 0>
static void addVariant(std::vector<FusedVariant>& v) {
    // only FFTs-per-block counts that fit in a block of 1024 threads
    if constexpr (FFTn<E, 1>::block_dim.x * FPB <= 1024) {
        using FFT = FFTn<E, FPB>;
        static_assert(!FFT::requires_workspace, "FFT size needs a workspace; only powers of two are supported");
        for (const auto& e : v)
            if (e.fpb == FPB && e.ept == E && e.regcap == REGCAP) return;
        void (*tensor)(const Geometry&, const PreVariant&, const FracParams&, const TensorOut&, int, int,
                       cudaStream_t) = nullptr;
        void (*half)(const Geometry&, const PreVariant&, const FracParams&, const HalfOut&, int, int,
                     cudaStream_t) = nullptr;
        if constexpr (REGCAP == 0) { tensor = &launchTensorVariant<FFT>; half = &launchHalfVariant<FFT>; }
        v.push_back({FPB, E, FFT::max_threads_per_block, FFT::shared_memory_size,
                     E == E0 && FPB == FFTBase::suggested_ffts_per_block && REGCAP == 0, REGCAP,
                     &launchVariant<FFT, REGCAP>, tensor, half, REGCAP == 0 ? TensorPlan<FFT>::smem : 0u});
    }
}

template<unsigned E>
static void addVariants(std::vector<FusedVariant>& v) {
    addVariant<E, 1>(v); addVariant<E, 2>(v); addVariant<E, 4>(v); addVariant<E, 8>(v);
    addVariant<E, 16>(v); addVariant<E, 32>(v);
    // registers capped for 1024 and 1536 threads per SM (an A100 SM holds 2048 threads and 64K
    // registers: 64 and 42 registers a thread), at the FFTs-per-block counts that do best here
    addVariant<E, 1, 1024>(v); addVariant<E, 2, 1024>(v); addVariant<E, 4, 1024>(v);
    addVariant<E, 1, 1536>(v); addVariant<E, 2, 1536>(v); addVariant<E, 4, 1536>(v);
}

static const int registered = [] {
    auto& v = fusedRegistry()[FUSE_N];
    addVariant<E0, FFTBase::suggested_ffts_per_block>(v);
    addVariants<E0>(v);
    using Half = decltype(Size<FUSE_N>() + Precision<float>() + Type<fft_type::c2c>() +
                          Direction<fft_direction::forward>() + Block() + ElementsPerThread<(E0 / 2 >= 2 ? E0 / 2 : 2)>());
    if constexpr (E0 / 2 >= 2 && is_supported_v<Half, FUSE_SM>)
        addVariants<E0 / 2>(v);
    return 0;
}();
