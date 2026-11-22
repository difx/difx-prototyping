// xmac.cu - the three XMAC versions of xmac.cuh.
#include <algorithm>
#include <cstdint>
#include <vector>
#include "xmac.cuh"
#include "xmac_copy.cuh"

// ---- gpudifx ---------------------------------------------------------------------------------
template<class T> static T* upload(const std::vector<T>& v) {
    T* p;
    CHECK_CUDA(cudaMalloc(&p, v.size() * sizeof(T)));
    CHECK_CUDA(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

void XmacGpudifx::init(const XmacShape& shape, const cuFloatComplex* spectra, const bool* valid, int multiprocessors) {
    s = shape;
    sms = multiprocessors;
    const int nbl = s.nbaseline();
    std::vector<const cuFloatComplex*> hm1(nbl), hm2(nbl);
    std::vector<const bool*> hv1(nbl), hv2(nbl);
    std::vector<int> hb1((size_t)s.nfreq * nbl * 4), hb2(hb1.size()), hoff(hb1.size());
    std::vector<int> hbs(nbl, s.fftchannels), hws(nbl, s.nband() * s.fftchannels);
    const size_t perAnt = (size_t)s.nwin * s.nband() * s.fftchannels;
    for (int a = 0; a < s.nant; a++)
        for (int b = a + 1; b < s.nant; b++) {
            const int bl = baselineIndex(a, b, s.nant);
            hm1[bl] = spectra + a * perAnt;
            hm2[bl] = spectra + b * perAnt;
            hv1[bl] = valid + (size_t)a * s.nwin;
            hv2[bl] = valid + (size_t)b * s.nwin;
            for (int f = 0; f < s.nfreq; f++)
                for (int p = 0; p < 4; p++) {
                    const int pol1 = p < 2 ? p : p - 2, pol2 = p < 2 ? p : 3 - p;
                    const size_t i = ((size_t)f * nbl + bl) * 4 + p;
                    hb1[i] = 2 * f + pol1;
                    hb2[i] = 2 * f + pol2;
                    hoff[i] = (int)s.visIndex(f, bl, p, 0);
                }
        }
    m1 = upload(hm1); m2 = upload(hm2); v1 = upload(hv1); v2 = upload(hv2);
    b1 = upload(hb1); b2 = upload(hb2); offsets = upload(hoff);
    bandStride = upload(hbs); windowStride = upload(hws);
    bandStride2 = upload(hbs); windowStride2 = upload(hws);
}

void XmacGpudifx::init(const XmacShape& shape, const Tables& t, int multiprocessors) {
    s = shape;
    sms = multiprocessors;
    m1 = upload(t.m1); m2 = upload(t.m2); v1 = upload(t.v1); v2 = upload(t.v2);
    b1 = upload(t.b1); b2 = upload(t.b2); offsets = upload(t.offsets);
    bandStride = upload(t.bandStride); windowStride = upload(t.windowStride);
    bandStride2 = upload(t.bandStride2); windowStride2 = upload(t.windowStride2);
}

void XmacGpudifx::run(cuFloatComplex* vis, cudaStream_t st) const {
    CHECK_CUDA(cudaMemsetAsync(vis, 0, s.visCount() * sizeof(cuFloatComplex), st));
    const int nbl = s.nbaseline();
    const XmacLaunch L = xmacGpudifxLaunch(nbl, 4, s.nchan, s.nwin, sms);
    for (int f = 0; f < s.nfreq; f++) {
        const size_t o = (size_t)f * nbl * 4;
        gpu_fuse_xmac_and_average<<<L.blocks, L.threads, 0, st>>>(m1, m2, v1, v2, b1 + o, b2 + o, offsets + o, vis,
            nbl, 4, s.nwin, L.fftsPerChunk, s.nchan, 1, bandStride, windowStride, bandStride2, windowStride2);
    }
    CHECK_CUDA(cudaGetLastError());
}

void XmacGpudifx::destroy() {
    for (void* p : {(void*)m1, (void*)m2, (void*)v1, (void*)v2, (void*)b1, (void*)b2, (void*)offsets,
                    (void*)bandStride, (void*)windowStride, (void*)bandStride2, (void*)windowStride2})
        cudaFree(p);
}

// ---- shared ----------------------------------------------------------------------------------
// Pairs (I, J), I <= J, of blocks of receivers, on the device.
static const int2* tilePairs(int ntiles, int& npairs) {
    static std::vector<std::pair<int, const int2*>> cache;
    for (auto& c : cache)
        if (c.first == ntiles) { npairs = ntiles * (ntiles + 1) / 2; return c.second; }
    std::vector<int2> v;
    for (int i = 0; i < ntiles; i++)
        for (int j = i; j < ntiles; j++) v.push_back(make_int2(i, j));
    npairs = (int)v.size();
    const int2* d = upload(v);
    cache.push_back({ntiles, d});
    return d;
}

// How many blocks to split the windows over: enough threads for twice what the GPU holds, with
// at least minPerChunk windows a chunk.
static int windowChunks(long long launchThreads, int nwin, int minPerChunk, int sms) {
    const long long target = (long long)sms * 2048 * 2;
    int n = (int)std::max(1LL, (target + launchThreads - 1) / launchThreads);
    return std::max(1, std::min(n, nwin / minPerChunk));
}

__device__ __forceinline__ void addVis(cuFloatComplex* vis, size_t i, float re, float im, bool atomic) {
    if (atomic) { atomicAdd(&vis[i].x, re); atomicAdd(&vis[i].y, im); }
    else vis[i] = make_cuFloatComplex(re, im);
}

// ---- staged tensor XMAC -----------------------------------------------------------------------
// The tensor-core XMAC from FP16 spectra in the station kernel's own order (halfIndex, or wherever
// the map says), each block rearranging what it needs in shared memory: no transpose kernel, and no
// layout the station kernel must write (the tensor layout, the transpose and the tensor XMAC from
// them, the tiled FP32 XMAC and a WMMA version of this one were retired after the comparisons in
// xmac/docs/, xmac-paths.md and the rest).
//
// The receivers are taken in blocks of 16 (the tensor cores' rows), and the blocks in super-blocks
// of KS.  A thread block takes CH consecutive channels of one frequency and one pair (A, B), A <= B,
// of super-blocks, and stages only those receivers: A's 16 KS at rows 0 .. 16 KS - 1 of the stage
// and, when B != A, B's at rows 16 KS .. 32 KS - 1.  Its work items are (channel, pair (i, j) of
// 16-blocks, i of A and j of B, i <= j when A == B), their sums in registers over the whole window
// loop.  stagedPlan chooses:
//   up to 32 receivers (16 stations)  one super-block of all of them (KS 1 or 2), a warp per
//                                     channel: the block stages every receiver, as DiFX's usual
//                                     arrays need, and reads each one once.  8 channels, but 16 at
//                                     KS 1 with more than 8 receivers (64-byte reads of a
//                                     receiver's window instead of 32: on the 2070, 1.2-1.35x
//                                     faster at 6-8 stations, where the 32-byte reads fetched 1.45x
//                                     the data from DRAM; 1.05-1.1x slower at 2-4)
//   more                              KS = 4 (64 receivers), 4 channels, 16 warps (4 a channel):
//                                     33 KB of shared memory a stage whatever the number of
//                                     stations.  Each receiver is read by every pair it is in (the
//                                     number of super-blocks), the repeats partly from L2
// (--staged-tile in xmacbench forces KS 1 or 2, or 4.  On the 2070, KS 4 is 1.1-2.0x faster than
// KS 2 with 8 channels from 24 stations up; KS 1 was 1.6-2.1x slower than KS 2.)
// The channel groups are fastest in the launch, so that the blocks reading neighbouring parts of a
// window's spectra run together (with pair fastest instead: 1.3-1.5x slower from 32 stations up).
//
//  - The stage holds the spectra as the station kernel wrote them, (re, im) interleaved: a row (one
//    receiver, one channel) is 16 windows x (re, im) FP16, 64 bytes, nothing rearranged.  With the
//    pairs along the sum dimension (32 values: re_0, im_0, re_1, ...), one product over a row gives
//    sum(re1 re2 + im1 im2), the real part of X1 X2^*; the imaginary part, sum(im1 re2 - re1 im2),
//    is the product with A's pairs turned into (im, -re), one byte permute and one sign flip per
//    register.  Four tensor-core multiplies per 16 windows.
//  - PTX: ldmatrix and mma.sync.m16n8k16 (on sm_75, two m16n8k8), whose register layouts are
//    documented: each 32-bit register of a fragment holds one (re, im) pair.  Each warp loads each
//    of its A and B fragments once per 8 windows and uses it for every item that needs it, and its
//    sums go to vis straight from the accumulator registers.  (A WMMA version, with real and
//    imaginary planes in the stage, was 1.05-1.4x slower.)
//  - The copy into the stage: a task is (channel quad, window quad, row): four 16-byte loads from
//    global memory, 4 channels of each of 4 windows, transposed in registers into four 16-byte
//    stores, one a channel, each exactly one chunk of the row (4 windows).  The next 16 windows are
//    loaded into registers while the tensor cores work on these, and stored after: into the other
//    of two stages where two cost no blocks an SM (stagedPlan; one barrier per 16 windows), else
//    into the one stage (two).
//    (Tried: a task of one window, a 16-byte load and four 4-byte stores; and on sm_80 the same as
//    four 4-byte cp.async into two stages, with no registers in between, which was 1.3-1.6x slower
//    on the A100 from 24 stations up: each cp.async touches the 32 lanes' 32 sectors for 4 bytes of
//    each.)
// A row's four 16-byte chunks (4 windows each) are XOR-swizzled by its row (chunk q at q ^ ((row >> 1)
// & 3)), so that ldmatrix's 8 rows of a phase fall in 8 different bank groups without padding; the
// channels' stages are 16 bytes apart beyond a multiple of 128, so that the copy's stores (lanes:
// channel quad fastest, then window quad, then row) fall in different banks too.
constexpr int SROW = 64;                                                // bytes a stage row
static __host__ __device__ int stagedChannelBytes(int srows) { return srows * SROW + 16; }
__device__ __forceinline__ int stagedChunk(int row, int q) { return row * SROW + ((q ^ ((row >> 1) & 3)) << 4); }

__device__ __forceinline__ void ldsm4(uint32_t r[4], const unsigned char* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
// d (16 x 8, FP32) += a (16 x 16, FP16, row-major) b (16 x 8, FP16, column-major)
__device__ __forceinline__ void mma16816(float d[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(b0));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[2]), "r"(a[3]), "r"(b1));
#endif
}
template<int KS, int CH>
struct MmaCfg {
    static constexpr int NW = CH == 8 ? 8 : 16;          // warps a block
    static constexpr int WPC = NW / CH;                  // warps a channel: 1, or KS (each its own j)
    static constexpr int NJ = WPC == 1 ? KS : 1;         // J blocks a warp
    static constexpr int MINB = (KS == 1 ? 2 : 1) * (NW == 8 ? 2 : 1);   // blocks an SM, for the register budget
    static constexpr int NQ = CH / 4;                    // channel quads a block
    static constexpr int RSTEP = NW * 32 / (NQ * 4);     // copy: rows between a thread's tasks
    static constexpr int MAXR = 32 * KS / RSTEP;         // copy: tasks a thread, at most
    static_assert(WPC == 1 || WPC == KS, "a warp per channel, or one per J block");
    static_assert(32 * KS % RSTEP == 0, "whole copy tasks");
};

// WT: weighted (xmacTensorStagedWeighted): the copy scales each value by sqrt|w| of its (window,
// channel)'s bin, the A fragments take negative weights' signs, and (W.skip) the block runs only the
// 16-window blocks with some non-zero weight in its channels (a list made first, when they fit MAXTL).
constexpr int MAXWBINS = 256, MAXTL = 1024;
constexpr size_t WEIGHTED_SMEM = MAXWBINS * 4 + MAXTL * 2 + MAXTL + 2 * 16 * 4 + 16;
__device__ __forceinline__ uint32_t scaleHalf2(uint32_t w, __half2 sc) {    // FP16 product, rounded
    const __half2 h = __hmul2(*reinterpret_cast<const __half2*>(&w), sc);
    return *reinterpret_cast<const uint32_t*>(&h);
}
// MODE 0 plain, 1 weighted (WT), 2 sorted (xmacTensorStagedSorted: the windows are slots sorted by bin
// per channel, channel-major (a receiver's channel c at off + c W.cstride, its slots contiguous: the
// copy loads 4 slots of a channel, no transpose); blockIdx.z is a segment of the block's channel
// group, (bin, first and last + 1 slot block, the bin's segment count), instead of a chunk of the
// windows; the sums go to that bin), 3 sorted and gathered (the same, the copy reading each slot's window
// from channel-major spectra in time order through W.slotWin: no permuted copy)
template<int KS, int CH, int MODE>
__global__ void __launch_bounds__(MmaCfg<KS, CH>::NW * 32, MmaCfg<KS, CH>::MINB)
xmac_staged_kernel(XmacShape s, XmacMap m, const __half2* __restrict__ spec, float2* vis,
                   const int2* __restrict__ superPairs, int nsp, int srows, int nbuf, int nchunk, XmacWeights W) {
    constexpr bool WT = MODE == 1;
    using C = MmaCfg<KS, CH>;
    constexpr int NJ = C::NJ;
    extern __shared__ __align__(32) unsigned char smraw[];
    const int CSb = stagedChannelBytes(srows), BUFb = CH * CSb;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int ncg = (s.nchan + CH - 1) / CH;
    const int sp = blockIdx.x / ncg, c0 = (blockIdx.x % ncg) * CH;      // channel group fastest
    const int f = blockIdx.y;
    const int R = m.nrecv[f], nb = (R + 15) / 16;                       // this frequency's receivers
    const int2 AB = superPairs[sp];
    if (AB.y * 16 * KS >= R) return;           // the pair is past this frequency's receivers (whole block)
    const long long* moff = m.off + (size_t)f * m.rmax;
    const int* mws = m.wstride + (size_t)f * m.rmax;
    const int* mst = m.station + (size_t)f * (m.rmax / 2);
    const bool diag = AB.x == AB.y;
    const int used = diag ? 16 * KS : 32 * KS;                          // stage rows in use
    const int ntb = (s.nwin + 15) / 16;
    int tb0, tb1, segBin = 0;
    bool atomicOut = nchunk > 1;
    if constexpr (MODE >= 2) {
        const int4 sg = W.segs[((size_t)f * ncg + blockIdx.x % ncg) * W.maxSeg + blockIdx.z];
        if (sg.x < 0) return;                                           // past the group's segments
        segBin = sg.x; tb0 = sg.y; tb1 = sg.z; atomicOut = sg.w > 1 || W.avg > 1 || W.accumulate;
    } else {
        const int per = (ntb + nchunk - 1) / nchunk;
        tb0 = blockIdx.z * per; tb1 = min(ntb, tb0 + per);
        if constexpr (MODE == 1) atomicOut |= W.avg > 1 || W.accumulate;
    }
    auto recvOf = [&](int row) {                                        // stage row -> receiver, or -1
        const int r = row < 16 * KS ? AB.x * 16 * KS + row : AB.y * 16 * KS + row - 16 * KS;
        return r < R ? r : -1;
    };

    // the rows past the last receiver: zeroed once, in every stage and channel (the copy never writes them)
    for (int u = threadIdx.x; u < nbuf * CH * used * 16; u += blockDim.x) {
        const int word = u % 16, row = (u / 16) % used, bc = u / (16 * used);
        if (recvOf(row) < 0) *reinterpret_cast<uint32_t*>(smraw + bc * CSb + row * SROW + 4 * word) = 0u;
    }

    // weighted: the table (sqrt|w| with w's sign); per 16-window block a mode, 0 skip (every weight 0),
    // 1 plain (every weight 1: copied as is), 2 scaled, 3 scaled with negative weights (their signs to
    // the A operand), and the list of blocks to run with their modes (W.skip; else every block, mode 2
    // or 3); the signs of the stage's windows, per channel 4 nibbles.  In the dynamic shared memory past
    // the stages (WEIGHTED_SMEM more; the unweighted kernel has none of it).
    float* swt = reinterpret_cast<float*>(smraw + nbuf * BUFb);
    short* tlist = reinterpret_cast<short*>(swt + MAXWBINS);
    unsigned char* tflag = reinterpret_cast<unsigned char*>(tlist + MAXTL);
    unsigned char* sgnb = tflag + MAXTL;                                // [buf][channel][window quad]
    int& tcount = *reinterpret_cast<int*>(sgnb + 2 * 16 * 4);
    bool useList = false;
    int defMode = 2;
    if constexpr (WT) {
        defMode = W.negative ? 3 : 2;
        for (int i = threadIdx.x; i < W.nbins; i += blockDim.x) {
            const float w = W.weights[i];
            swt[i] = copysignf(sqrtf(fabsf(w)), w);
        }
        __syncthreads();
        if (W.skip && tb1 - tb0 <= MAXTL) {
            useList = true;
            // flags per 16-window block: 1 some weight non-zero, 2 some weight not 1, 4 some negative.
            // A half-warp a block's 16 windows, each lane its window's CH channels
            const int nwi = (tb1 - tb0) * 16;
            for (int base = 0; base < nwi; base += blockDim.x) {
                const int i = base + threadIdx.x, wg = tb0 * 16 + i;
                unsigned fl = 0;
                if (i < nwi && wg < s.nwin) {
                    const uint8_t* b = W.bins + ((size_t)wg * s.nfreq + f) * s.nchan + c0;
                    for (int qq = 0; qq < CH / 4 && c0 + 4 * qq < s.nchan; qq++) {
                        const unsigned b4 = __ldg(reinterpret_cast<const unsigned*>(b + 4 * qq));
#pragma unroll
                        for (int e = 0; e < 4; e++) {
                            const float w = swt[(b4 >> (8 * e)) & 255];
                            fl |= (w != 0.f ? 1u : 0u) | (w != 1.f ? 2u : 0u) | (w < 0.f ? 4u : 0u);
                        }
                    }
                }
#pragma unroll
                for (int o = 8; o > 0; o >>= 1) fl |= __shfl_xor_sync(0xffffffffu, fl, o);
                if ((threadIdx.x & 15) == 0 && i < nwi) tflag[i >> 4] = (unsigned char)fl;
            }
            __syncthreads();
            if (threadIdx.x < 32) {                                     // the list: (block, mode) of the non-zero
                int n = 0;
                for (int base = 0; base < tb1 - tb0; base += 32) {
                    const int t = base + (int)threadIdx.x;
                    const unsigned fl = t < tb1 - tb0 ? tflag[t] : 0u;
                    const bool on = fl & 1u;
                    const unsigned bal = __ballot_sync(0xffffffffu, on);
                    const int mode = !(fl & 2u) ? 1 : (fl & 4u) ? 3 : 2;
                    if (on) tlist[n + __popc(bal & ((1u << threadIdx.x) - 1))] = (short)(t | mode << 12);
                    n += __popc(bal);
                }
                if (threadIdx.x == 0) tcount = n;
            }
            __syncthreads();
        }
    }

    // the copy tasks: (channel quad q, window quad wq, row), q fastest then wq; a thread's q and wq are
    // fixed, its rows row0, row0 + RSTEP, ...: their spectra (this block's channels, window 0), or none
    const int q = threadIdx.x % C::NQ, wq = (threadIdx.x / C::NQ) % 4, row0 = threadIdx.x / (C::NQ * 4);
    // src: where each task's spectra start (window 0), as a 32-bit offset in __half2 from spec, or NONE;
    // wstride: __half2 a window.  (Rows may have different window strides.  As 64-bit pointers, with a
    // stride each, the KS 2 kernel spilled at its 128 registers; the host checks the offsets fit.)
    constexpr unsigned NONE = 0xffffffffu;
    unsigned src[C::MAXR];
    int dofs[C::MAXR], wstride[C::MAXR];
#pragma unroll
    for (int k = 0; k < C::MAXR; k++) {
        const int row = row0 + k * C::RSTEP;
        const int r = row < used ? recvOf(row) : -1;
        const bool any = r >= 0 && c0 + 4 * q < s.nchan;
        src[k] = any ? (unsigned)moff[r] + (MODE >= 2 ? (unsigned)(c0 + 4 * q) * (unsigned)W.cstride : c0 + 4 * q) : NONE;
        wstride[k] = any ? mws[r] : 0;
        dofs[k] = 4 * q * CSb + stagedChunk(row, wq);
    }
    // windows 16 tb + 4 wq .. + 3 of each task into registers (zero past the last window) ...
    uint4 v[C::MAXR][4];
    uint32_t bq[WT ? 4 : 1];                    // weighted: the 4 channels' bins of each of the 4 windows
    auto fetch = [&](int tb, int mode = 1) {
        if constexpr (MODE == 2) {            // channel-major slots: v[k][cq] is channel cq's 4 slots
#pragma unroll
            for (int k = 0; k < C::MAXR; k++)
#pragma unroll
                for (int cq = 0; cq < 4; cq++)
                    v[k][cq] = src[k] != NONE
                             ? __ldg(reinterpret_cast<const uint4*>(spec + (src[k] + (unsigned)cq * (unsigned)W.cstride
                                                                            + (unsigned)(tb * 16 + 4 * wq))))
                             : make_uint4(0, 0, 0, 0);
            return;
        }
        if constexpr (MODE == 3) {            // gathered: channel cq's 4 slots' windows (-1: padding)
            const bool anyq = c0 + 4 * q < s.nchan;
#pragma unroll
            for (int cq = 0; cq < 4; cq++) {
                const int4 wi = anyq ? __ldg(reinterpret_cast<const int4*>(W.slotWin + ((size_t)f * s.nchan + c0 + 4 * q + cq) * W.cap
                                                                           + tb * 16 + 4 * wq))
                                     : make_int4(-1, -1, -1, -1);
#pragma unroll
                for (int k = 0; k < C::MAXR; k++) {
                    if (src[k] == NONE) { v[k][cq] = make_uint4(0, 0, 0, 0); continue; }
                    const unsigned* b = reinterpret_cast<const unsigned*>(spec + src[k] + (unsigned)cq * (unsigned)W.cstride);
                    v[k][cq] = make_uint4(wi.x >= 0 ? __ldg(b + wi.x) : 0u, wi.y >= 0 ? __ldg(b + wi.y) : 0u,
                                          wi.z >= 0 ? __ldg(b + wi.z) : 0u, wi.w >= 0 ? __ldg(b + wi.w) : 0u);
                }
            }
            return;
        }
        if (WT && mode >= 2) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int wg = tb * 16 + 4 * wq + e;
                bq[e] = wg < s.nwin && c0 + 4 * q < s.nchan
                      ? __ldg(reinterpret_cast<const unsigned*>(W.bins + ((size_t)wg * s.nfreq + f) * s.nchan + c0 + 4 * q))
                      : 0u;
            }
        }
#pragma unroll
        for (int k = 0; k < C::MAXR; k++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int wg = tb * 16 + 4 * wq + e;
                v[k][e] = src[k] != NONE && wg < s.nwin
                        ? __ldg(reinterpret_cast<const uint4*>(spec + (src[k] + (unsigned)wg * (unsigned)wstride[k])))
                        : make_uint4(0, 0, 0, 0);
            }
    };
    // ... and from there into a stage: channel cq's 4 windows are word cq of each, one chunk
    auto put = [&](int buf, int mode = 1) {
        if (WT && mode >= 2) {                  // weighted: channel cq's 4 windows times their sqrt|w|
#pragma unroll
            for (int cq = 0; cq < 4; cq++) {
                __half2 sc[4];
                unsigned neg = 0;
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const float w = swt[(bq[e] >> (8 * cq)) & 255];
                    sc[e] = __float2half2_rn(fabsf(w));
                    neg |= (w < 0.f ? 1u : 0u) << e;
                }
                if (mode == 3 && row0 == 0) sgnb[(buf * 16 + 4 * q + cq) * 4 + wq] = (unsigned char)neg;
#pragma unroll
                for (int k = 0; k < C::MAXR; k++) {
                    if (src[k] == NONE) continue;
                    *reinterpret_cast<uint4*>(smraw + buf * BUFb + dofs[k] + cq * CSb) = make_uint4(
                        scaleHalf2((&v[k][0].x)[cq], sc[0]), scaleHalf2((&v[k][1].x)[cq], sc[1]),
                        scaleHalf2((&v[k][2].x)[cq], sc[2]), scaleHalf2((&v[k][3].x)[cq], sc[3]));
                }
            }
        } else if constexpr (MODE >= 2) {
#pragma unroll
            for (int k = 0; k < C::MAXR; k++) {
                if (src[k] == NONE) continue;
#pragma unroll
                for (int cq = 0; cq < 4; cq++) *reinterpret_cast<uint4*>(smraw + buf * BUFb + dofs[k] + cq * CSb) = v[k][cq];
            }
        } else {
#pragma unroll
        for (int k = 0; k < C::MAXR; k++) {
            if (src[k] == NONE) continue;
            unsigned char* d = smraw + buf * BUFb + dofs[k];
#pragma unroll
            for (int cq = 0; cq < 4; cq++)
                *reinterpret_cast<uint4*>(d + cq * CSb) = make_uint4((&v[k][0].x)[cq], (&v[k][1].x)[cq],
                                                                     (&v[k][2].x)[cq], (&v[k][3].x)[cq]);
        }
        }
    };

    // the warp's items: its channel; I blocks i = 0 .. KS - 1 of super-block A, J blocks j of B (every
    // j with a warp per channel, else j = sub); i <= j when A == B
    const int cw = warp % CH, sub = warp / CH;
    const int c = c0 + cw;
    bool on[KS][NJ], anyI[KS], anyJ[NJ];
#pragma unroll
    for (int jj = 0; jj < NJ; jj++) anyJ[jj] = false;
#pragma unroll
    for (int i = 0; i < KS; i++) {
        anyI[i] = false;
#pragma unroll
        for (int jj = 0; jj < NJ; jj++) {
            const int j = NJ == KS ? jj : sub;
            on[i][jj] = c < s.nchan && (!diag || i <= j) && AB.x * KS + i < nb && AB.y * KS + j < nb;
            anyI[i] |= on[i][jj];
            anyJ[jj] |= on[i][jj];
        }
    }
    float acc[KS][NJ][2][2][4];                                         // [i][jj][re, im][n half][4]
#pragma unroll
    for (int i = 0; i < KS; i++)
#pragma unroll
        for (int jj = 0; jj < NJ; jj++)
#pragma unroll
            for (int q = 0; q < 16; q++) (&acc[i][jj][0][0][0])[q] = 0.f;
    // ldmatrix lane addresses within a 16-row block: A's four 8 x 8 matrices are (rows 0-7, 8-15) x
    // (first, second 8 values), B's (rows 0-7: first, second 8 values), then rows 8-15
    const int aRow = lane & 15, aHalf = lane >> 4;
    const int bRow = (lane & 7) + ((lane >> 4) << 3), bHalf = (lane >> 3) & 1;
    auto compute = [&](int buf, int mode = 1) {
        const unsigned char* base = smraw + buf * BUFb + cw * CSb;
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {                                // windows 8 ks .. 8 ks + 7
            // weighted, negative weights: A's registers 0, 1 hold window 8 ks + t, 2, 3 window 8 ks + 4 + t
            // (ldmatrix: t = lane & 3), whose sign they take (window quads 2 ks and 2 ks + 1 of the stage)
            uint32_t sgn[2] = {0u, 0u};
            if (WT && mode == 3) {
#pragma unroll
                for (int h = 0; h < 2; h++)
                    if ((sgnb[(buf * 16 + cw) * 4 + 2 * ks + h] >> (lane & 3)) & 1u) sgn[h] = 0x80008000u;
            }
            uint32_t Bf[NJ][4];
#pragma unroll
            for (int jj = 0; jj < NJ; jj++) {
                if (!anyJ[jj]) continue;
                const int j = NJ == KS ? jj : sub;
                const int row = (diag ? 0 : 16 * KS) + 16 * j + bRow;
                ldsm4(Bf[jj], base + stagedChunk(row, 2 * ks + bHalf));
            }
#pragma unroll
            for (int i = 0; i < KS; i++) {
                if (!anyI[i]) continue;
                uint32_t Af[4];
                ldsm4(Af, base + stagedChunk(16 * i + aRow, 2 * ks + aHalf));
                if constexpr (WT) {
                    Af[0] ^= sgn[0]; Af[1] ^= sgn[0]; Af[2] ^= sgn[1]; Af[3] ^= sgn[1];
                }
#pragma unroll
                for (int jj = 0; jj < NJ; jj++) {                       // real parts
                    if (!on[i][jj]) continue;
                    mma16816(acc[i][jj][0][0], Af, Bf[jj][0], Bf[jj][1]);
                    mma16816(acc[i][jj][0][1], Af, Bf[jj][2], Bf[jj][3]);
                }
#pragma unroll
                for (int e = 0; e < 4; e++) Af[e] = __byte_perm(Af[e], 0, 0x1032) ^ 0x80000000u;   // (re, im) -> (im, -re)
#pragma unroll
                for (int jj = 0; jj < NJ; jj++) {                       // imaginary parts
                    if (!on[i][jj]) continue;
                    mma16816(acc[i][jj][1][0], Af, Bf[jj][0], Bf[jj][1]);
                    mma16816(acc[i][jj][1][1], Af, Bf[jj][2], Bf[jj][3]);
                }
            }
        }
    };

    // the sums out, from the accumulators, to dst ([freq][baseline][product][channel]): register e of n
    // half h is row g + 8 (e >> 1), column 8 h + 2 t + (e & 1) of the 16 x 16 tile
    auto store = [&](float2* dst) {
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < KS; i++)
#pragma unroll
        for (int jj = 0; jj < NJ; jj++) {
            if (!on[i][jj]) continue;
            const int I = AB.x * KS + i, J = AB.y * KS + (NJ == KS ? jj : sub);
#pragma unroll
            for (int h = 0; h < 2; h++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int r1 = I * 16 + g + 8 * (e >> 1), r2 = J * 16 + 8 * h + 2 * t + (e & 1);
                    if (r1 >= R || r2 >= R || (r1 >> 1) >= (r2 >> 1)) continue;
                    const int bl = baselineIndex(mst[r1 >> 1], mst[r2 >> 1], s.nant);
                    if constexpr (MODE == 0)
                        addVis((cuFloatComplex*)dst, s.visIndex(f, bl, productIndex(r1 & 1, r2 & 1), c), acc[i][jj][0][h][e],
                               acc[i][jj][1][h][e], atomicOut);
                    else                                // binned: W.avg channels summed into each output channel
                        addVis((cuFloatComplex*)dst, (((size_t)f * s.nbaseline() + bl) * 4 + productIndex(r1 & 1, r2 & 1)) *
                                                     (s.nchan / W.avg) + c / W.avg,
                               acc[i][jj][0][h][e], acc[i][jj][1][h][e], atomicOut);
                }
        }
    };

    if constexpr (MODE != 1) {                  // plain, and sorted (its segment's slots)
    if (tb0 < tb1) { fetch(tb0); put(0); }
    __syncthreads();
    if (nbuf == 2) {
        for (int tb = tb0; tb < tb1; tb++) {
            const int buf = (tb - tb0) & 1;
            const bool more = tb + 1 < tb1;
            if (more) fetch(tb + 1);             // in flight during the products
            compute(buf);
            if (more) put(buf ^ 1);              // every warp was done with that stage at the last barrier
            __syncthreads();
        }
    } else {
        for (int tb = tb0; tb < tb1; tb++) {
            const bool more = tb + 1 < tb1;
            if (more) fetch(tb + 1);
            compute(0);
            __syncthreads();                     // done with the stage
            if (more) put(0);
            __syncthreads();
        }
    }
    } else {
    // weighted: the same over tb0 .. tb1 - 1 (mode defMode), or (skipping) the list's blocks and modes
    const int nt = useList ? tcount : tb1 - tb0;
    auto entry = [&](int i) { return useList ? (int)tlist[i] : (i | defMode << 12); };
    if (nt > 0) { const int e = entry(0); fetch(tb0 + (e & 0xfff), e >> 12); put(0, e >> 12); }
    __syncthreads();
    if (nbuf == 2) {
        for (int i = 0; i < nt; i++) {
            const int buf = i & 1;
            const bool more = i + 1 < nt;
            const int e = entry(i), en = more ? entry(i + 1) : 0;
            if (more) fetch(tb0 + (en & 0xfff), en >> 12);
            compute(buf, e >> 12);
            if (more) put(buf ^ 1, en >> 12);
            __syncthreads();
        }
    } else {
        for (int i = 0; i < nt; i++) {
            const bool more = i + 1 < nt;
            const int e = entry(i), en = more ? entry(i + 1) : 0;
            if (more) fetch(tb0 + (en & 0xfff), en >> 12);
            compute(0, e >> 12);
            __syncthreads();
            if (more) put(0, en >> 12);
            __syncthreads();
        }
    }
    }

    // binned, averaging at least the block's channels: the warps' sums (one channel each) added in shared
    // memory first (the stages, done with), and only each J's first warp stores them, so a block writes
    // one channel's worth instead of CH (when it fits; else every warp adds its own)
    bool storeHere = true;
    if constexpr (MODE >= 1) {
        constexpr int NE = KS * NJ * 16;                                // accumulators a lane
        if (W.avg >= CH && C::WPC * 32 * NE * 4 <= nbuf * BUFb) {
            float* red = reinterpret_cast<float*>(smraw);              // [sub][NE][lane]: lanes in different banks
            __syncthreads();
            for (int u = threadIdx.x; u < C::WPC * 32 * NE; u += blockDim.x) red[u] = 0.f;
            __syncthreads();
            float* mine = red + (size_t)sub * NE * 32 + lane;
#pragma unroll
            for (int i = 0; i < KS; i++)
#pragma unroll
                for (int jj = 0; jj < NJ; jj++)
                    if (on[i][jj])
#pragma unroll
                        for (int u = 0; u < 16; u++) atomicAdd(mine + ((i * NJ + jj) * 16 + u) * 32, (&acc[i][jj][0][0][0])[u]);
            __syncthreads();
            storeHere = cw == 0;
            if (storeHere)
#pragma unroll
                for (int i = 0; i < KS; i++)
#pragma unroll
                    for (int jj = 0; jj < NJ; jj++)
#pragma unroll
                        for (int u = 0; u < 16; u++) (&acc[i][jj][0][0][0])[u] = mine[((i * NJ + jj) * 16 + u) * 32];
        }
    }
    if (!storeHere) return;
    if constexpr (MODE >= 2) store(vis + (size_t)segBin * (s.visCount() / W.avg));
    else store(vis);
}

// The plan for a shape: KS, channels a block, the super-blocks, the stage's rows, the warps, and the
// shared memory a block (the version's own).
struct StagedPlan { int ks, ch, nsuper, srows, nw, nbuf; size_t smem; };
static int stagedForcedKS = 0;                 // xmacTensorStagedSetTile; 0: stagedPlan chooses
void xmacTensorStagedSetTile(int ks) { stagedForcedKS = ks; }

static int deviceAttr(cudaDeviceAttr a) {
    int dev, v;
    CHECK_CUDA(cudaGetDevice(&dev));
    CHECK_CUDA(cudaDeviceGetAttribute(&v, a, dev));
    return v;
}
// Blocks an SM of the mma.sync version with KS and this much shared memory a block (the occupancy
// calculator: registers, warps and shared memory).  On the 2070 (64 KB an SM) two stages halve the
// blocks at KS 2 and cost a quarter at KS 1, and those were 1.1-1.3x slower than one; on the A100
// (164 KB) the registers limit the blocks first.
template<int KS, int CH> static int stagedBlocks(size_t smem) {
    auto k = xmac_staged_kernel<KS, CH, 0>;
    static bool set = false;
    if (!set) {
        CHECK_CUDA(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        deviceAttr(cudaDevAttrMaxSharedMemoryPerBlockOptin)));
        set = true;
    }
    int n = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, k, MmaCfg<KS, CH>::NW * 32, smem));
    return n;
}
static int stagedBlocksPerSM(int ks, int ch, size_t smem) {
    return ks == 1 ? (ch == 16 ? stagedBlocks<1, 16>(smem) : stagedBlocks<1, 8>(smem))
         : ks == 2 ? stagedBlocks<2, 8>(smem) : stagedBlocks<4, 4>(smem);
}

// rmax: receivers a frequency, at most (the map's; s.nrecv() for the regular layout)
static StagedPlan stagedPlan(const XmacShape& s, int rmax) {
    const int nb = (rmax + 15) / 16;
    static int maxSmem = 0;
    if (!maxSmem) maxSmem = deviceAttr(cudaDevAttrMaxSharedMemoryPerBlockOptin);
    auto plan = [&](int ks) {
        StagedPlan p;
        p.ks = ks;
        p.ch = ks == 4 ? 4 : ks == 1 && rmax > 8 ? 16 : 8;
        p.nw = p.ch == 8 ? 8 : 16;
        p.nsuper = (nb + ks - 1) / ks;
        p.srows = (p.nsuper == 1 ? 16 : 32) * ks;
        const size_t one = (size_t)p.ch * stagedChannelBytes(p.srows);  // two stages where they cost no
        p.nbuf = 2 * one <= (size_t)maxSmem && stagedBlocksPerSM(ks, p.ch, 2 * one) >= stagedBlocksPerSM(ks, p.ch, one)
               ? 2 : 1;                                                // blocks an SM
        p.smem = p.nbuf * one;
        return p;
    };
    if (stagedForcedKS) return plan(stagedForcedKS);
    if (nb <= 2) return plan(nb);                                       // one super-block: every receiver
    return plan(4);                                                     // else 64-receiver super-blocks, 4 channels
}

static StagedPlan stagedPlan(const XmacShape& s) { return stagedPlan(s, s.nrecv()); }
int xmacTensorStagedTile(const XmacShape& s) { return stagedPlan(s).ks; }
int xmacTensorStagedStages(const XmacShape& s) { return stagedPlan(s).nbuf; }
bool xmacTensorStagedFits(const XmacShape& s) {
    static int maxSmem = 0;
    if (!maxSmem) maxSmem = deviceAttr(cudaDevAttrMaxSharedMemoryPerBlockOptin);
    return stagedPlan(s).smem <= (size_t)maxSmem;
}

template<int KS, int CH, int MODE = 0>
static void launchStaged(const XmacShape& s, const XmacMap& m, const StagedPlan& P, const __half2* spec,
                         cuFloatComplex* vis, int sms, cudaStream_t st, const XmacWeights& W = XmacWeights{}) {
    static size_t smemSet = 0;
    auto kernel = xmac_staged_kernel<KS, CH, MODE>;
    const size_t smem = P.smem + (MODE == 1 ? WEIGHTED_SMEM : 0);
    if (smem > smemSet) {
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        smemSet = smem;
    }
    int nsp;
    const int2* superPairs = tilePairs(P.nsuper, nsp);
    const int threads = P.nw * 32;
    const unsigned nx = (unsigned)((s.nchan + CH - 1) / CH * nsp);
    const int ntb = (s.nwin + 15) / 16;
    const int nchunk = MODE >= 2 ? W.maxSeg : windowChunks((long long)nx * s.nfreq * threads, ntb, 4, sms);
    const size_t vc = MODE >= 1 ? s.visCount() / W.avg : s.visCount();     // binned: averaged channels
    if (MODE >= 1 && W.accumulate) {}     // binned, accumulating: added to what is there
    else if (MODE >= 2)                   // sorted: every bin's (a channel may lack some bins)
        CHECK_CUDA(cudaMemsetAsync(vis, 0, (size_t)W.nbins * vc * sizeof(cuFloatComplex), st));
    else if (nchunk > 1 || !m.complete || (MODE == 1 && W.avg > 1))   // else every block stores its sums, one each
        CHECK_CUDA(cudaMemsetAsync(vis, 0, vc * sizeof(cuFloatComplex), st));
    kernel<<<dim3(nx, s.nfreq, nchunk), threads, smem, st>>>(s, m, spec, (float2*)vis, superPairs, nsp, P.srows,
                                                              P.nbuf, nchunk, W);
    CHECK_CUDA(cudaGetLastError());
}

XmacMapHost xmacRegularMap(const XmacShape& s) {
    XmacMapHost h;
    h.rmax = s.nrecv();
    h.nrecv.assign(s.nfreq, h.rmax);
    h.off.resize((size_t)s.nfreq * h.rmax);
    h.wstride.assign(h.off.size(), s.nband() * s.nchan);
    h.station.resize((size_t)s.nfreq * s.nant);
    for (int f = 0; f < s.nfreq; f++) {
        for (int r = 0; r < h.rmax; r++) h.off[(size_t)f * h.rmax + r] = (long long)s.halfIndex(r >> 1, 0, 2 * f + (r & 1), 0);
        for (int a = 0; a < s.nant; a++) h.station[(size_t)f * s.nant + a] = a;
    }
    return h;
}

XmacMap xmacUploadMap(const XmacMapHost& h) {
    for (size_t i = 0; i < h.off.size(); i++)
        if (h.off[i] % 4 || h.wstride[i] % 4) {
            fprintf(stderr, "staged tensor XMAC: map offsets and window strides must be multiples of 4 __half2\n");
            exit(2);
        }
    XmacMap m;
    m.rmax = h.rmax;
    m.complete = std::all_of(h.nrecv.begin(), h.nrecv.end(), [&](int n) { return n == h.rmax; });
    for (size_t i = 0; i < h.off.size(); i++) {
        m.maxOff = std::max(m.maxOff, h.off[i]);
        m.maxStride = std::max(m.maxStride, (long long)h.wstride[i]);
    }
    m.nrecv = upload(h.nrecv);
    m.off = upload(h.off);
    m.wstride = upload(h.wstride);
    m.station = upload(h.station);
    return m;
}

void xmacFreeMap(XmacMap& m) {
    for (const void* p : {(const void*)m.nrecv, (const void*)m.off, (const void*)m.wstride, (const void*)m.station})
        cudaFree(const_cast<void*>(p));
    m = XmacMap{};
}

template<int MODE>
static void stagedDispatch(const XmacShape& s, const XmacMap& m, const __half2* spec, cuFloatComplex* vis, int sms,
                           cudaStream_t st, const XmacWeights& W) {
    if (s.nchan % 4) { fprintf(stderr, "staged tensor XMAC: the channel count must be a multiple of 4\n"); exit(2); }
    if (m.maxOff + (long long)(s.nwin - 1) * m.maxStride + s.nchan > 0xfffffffeLL) {
        fprintf(stderr, "staged tensor XMAC: the spectra span more than 2^32 __half2 (16 GiB): use fewer windows\n");
        exit(2);
    }
    const StagedPlan P = stagedPlan(s, m.rmax);
    static int maxSmem = 0;
    if (!maxSmem) maxSmem = deviceAttr(cudaDevAttrMaxSharedMemoryPerBlockOptin);
    if (P.smem + (MODE == 1 ? WEIGHTED_SMEM : 0) > (size_t)maxSmem) {
        fprintf(stderr, "staged tensor XMAC: %zu bytes of shared memory a block (KS %d), more than a block may have\n",
                P.smem + (MODE == 1 ? WEIGHTED_SMEM : 0), P.ks);
        exit(2);
    }
    if (P.ks == 1 && P.ch == 16) launchStaged<1, 16, MODE>(s, m, P, spec, vis, sms, st, W);
    else if (P.ks == 1) launchStaged<1, 8, MODE>(s, m, P, spec, vis, sms, st, W);
    else if (P.ks == 2) launchStaged<2, 8, MODE>(s, m, P, spec, vis, sms, st, W);
    else if (P.ks == 4) launchStaged<4, 4, MODE>(s, m, P, spec, vis, sms, st, W);
    else { fprintf(stderr, "staged tensor XMAC: KS must be 1, 2 or 4, not %d\n", P.ks); exit(2); }
}
void xmacTensorStagedMapped(const XmacShape& s, const XmacMap& m, const __half2* spec, cuFloatComplex* vis, int sms,
                            cudaStream_t st) {
    stagedDispatch<0>(s, m, spec, vis, sms, st, XmacWeights{});
}
static void checkAvg(const XmacShape& s, int avg) {
    if (avg < 1 || (avg & (avg - 1)) || s.nchan % avg) {
        fprintf(stderr, "staged tensor XMAC, binned: the channels averaged a power of 2 dividing the channels\n");
        exit(2);
    }
}
void xmacTensorStagedWeighted(const XmacShape& s, const XmacMap& m, const __half2* spec, cuFloatComplex* vis,
                              const XmacWeights& w, int sms, cudaStream_t st) {
    if (w.nbins < 1 || w.nbins > MAXWBINS || !w.bins || !w.weights) {
        fprintf(stderr, "staged tensor XMAC, weighted: 1 to %d bins, with their bins and weights\n", MAXWBINS);
        exit(2);
    }
    checkAvg(s, w.avg);
    stagedDispatch<1>(s, m, spec, vis, sms, st, w);
}
void xmacTensorStagedSorted(const XmacShape& s, const XmacMap& m, const __half2* spec, cuFloatComplex* vis,
                            const int4* segs, int maxSeg, int nbins, int cstride, int sms, cudaStream_t st,
                            const int* slotWin, int avg, bool accumulate) {
    checkAvg(s, avg);
    if (nbins < 1 || !segs || maxSeg < 1 || (!slotWin && (cstride % 4 || cstride < s.nwin)) || s.nwin % 4 ||
        m.maxOff + (long long)s.nchan * cstride > 0xfffffffeLL) {
        fprintf(stderr, "staged tensor XMAC, sorted: bins and segments needed; the channel stride a multiple of 4, at "
                        "least the slots, and the spectra within 2^32 __half2\n");
        exit(2);
    }
    XmacWeights w;
    w.cstride = cstride;
    w.avg = avg;
    w.accumulate = accumulate;
    if (slotWin) { w.slotWin = slotWin; w.cap = s.nwin; }
    w.segs = segs;
    w.maxSeg = maxSeg;
    w.nbins = nbins;
    if (slotWin) stagedDispatch<3>(s, m, spec, vis, sms, st, w);
    else stagedDispatch<2>(s, m, spec, vis, sms, st, w);
}
int xmacTensorStagedChannels(const XmacShape& s) { return stagedPlan(s).ch; }

// The regular layout (halfIndex): its map, made and uploaded once per shape.
void xmacTensorStaged(const XmacShape& s, const __half2* spec, cuFloatComplex* vis, int sms, cudaStream_t st) {
    static XmacShape mapped{};
    static XmacMap map{};
    if (map.rmax == 0 || mapped.nant != s.nant || mapped.nfreq != s.nfreq || mapped.nchan != s.nchan ||
        mapped.nwin != s.nwin) {
        if (map.rmax) xmacFreeMap(map);
        map = xmacUploadMap(xmacRegularMap(s));
        mapped = s;
    }
    xmacTensorStagedMapped(s, map, spec, vis, sms, st);
}
