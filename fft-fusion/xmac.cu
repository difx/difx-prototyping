// xmac.cu - the three XMAC versions of xmac.cuh.
#include <algorithm>
#include <cstdint>
#include <vector>
#include <mma.h>
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
}

void XmacGpudifx::run(cuFloatComplex* vis, cudaStream_t st) const {
    CHECK_CUDA(cudaMemsetAsync(vis, 0, s.visCount() * sizeof(cuFloatComplex), st));
    const int nbl = s.nbaseline();
    const XmacLaunch L = xmacGpudifxLaunch(nbl, 4, s.nchan, s.nwin, sms);
    for (int f = 0; f < s.nfreq; f++) {
        const size_t o = (size_t)f * nbl * 4;
        gpu_fuse_xmac_and_average<<<L.blocks, L.threads, 0, st>>>(m1, m2, v1, v2, b1 + o, b2 + o, offsets + o, vis,
            nbl, 4, s.nwin, L.fftsPerChunk, s.nchan, 1, bandStride, windowStride, bandStride, windowStride);
    }
    CHECK_CUDA(cudaGetLastError());
}

void XmacGpudifx::destroy() {
    for (void* p : {(void*)m1, (void*)m2, (void*)v1, (void*)v2, (void*)b1, (void*)b2, (void*)offsets,
                    (void*)bandStride, (void*)windowStride})
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

// ---- tiled FP32 ------------------------------------------------------------------------------
// Thread (channel, pair of receiver blocks I <= J of T receivers): the T x T products X_r1 conj(X_r2)
// of r1 in block I, r2 in block J, summed over its chunk of windows.  Block: 32 channels x 8
// pairs; grid: channel groups x pair groups x (freq, window chunk).
constexpr int XT = 4;
__global__ void __launch_bounds__(256) xmac_tiled_kernel(XmacShape s, const float2* __restrict__ spec, float2* vis,
                                                         const int2* __restrict__ pairs, int npairs, int nchunk) {
    const int c = blockIdx.x * 32 + threadIdx.x;
    const int tp = blockIdx.y * blockDim.y + threadIdx.y;
    const int f = blockIdx.z / nchunk, chunk = blockIdx.z % nchunk;
    if (c >= s.nchan || tp >= npairs) return;
    const int2 IJ = pairs[tp];
    const int R = s.nrecv();
    const int per = (s.nwin + nchunk - 1) / nchunk;
    const int w0 = chunk * per, w1 = min(s.nwin, w0 + per);
    const size_t wstride = (size_t)s.nband() * s.fftchannels;
    size_t pr[XT], pc[XT];
    bool okr[XT], okc[XT];
#pragma unroll
    for (int i = 0; i < XT; i++) {
        const int r1 = IJ.x * XT + i, r2 = IJ.y * XT + i;
        okr[i] = r1 < R; okc[i] = r2 < R;
        const int q1 = okr[i] ? r1 : 0, q2 = okc[i] ? r2 : 0;
        pr[i] = s.specIndex(q1 >> 1, w0, 2 * f + (q1 & 1), c);
        pc[i] = s.specIndex(q2 >> 1, w0, 2 * f + (q2 & 1), c);
    }
    float ar[XT][XT] = {}, ai[XT][XT] = {};
#pragma unroll 2
    for (int w = w0; w < w1; w++) {
        float2 x[XT], y[XT];
#pragma unroll
        for (int i = 0; i < XT; i++) {
            x[i] = spec[pr[i]]; pr[i] += wstride;
            y[i] = spec[pc[i]]; pc[i] += wstride;
        }
#pragma unroll
        for (int i = 0; i < XT; i++)
#pragma unroll
            for (int j = 0; j < XT; j++) {          // x conj(y)
                ar[i][j] += x[i].x * y[j].x + x[i].y * y[j].y;
                ai[i][j] += x[i].y * y[j].x - x[i].x * y[j].y;
            }
    }
#pragma unroll
    for (int i = 0; i < XT; i++)
#pragma unroll
        for (int j = 0; j < XT; j++) {
            const int r1 = IJ.x * XT + i, r2 = IJ.y * XT + j;
            if (!okr[i] || !okc[j] || (r1 >> 1) >= (r2 >> 1)) continue;   // padding; same station
            const int bl = baselineIndex(r1 >> 1, r2 >> 1, s.nant);
            addVis(vis, s.visIndex(f, bl, productIndex(r1 & 1, r2 & 1), c), ar[i][j], ai[i][j], nchunk > 1);
        }
}

void xmacTiled(const XmacShape& s, const cuFloatComplex* spectra, cuFloatComplex* vis, int sms, cudaStream_t st) {
    int npairs;
    const int2* pairs = tilePairs((s.nrecv() + XT - 1) / XT, npairs);
    const dim3 threads(32, 8);
    const unsigned cg = (s.nchan + 31) / 32, pg = (npairs + 7) / 8;
    const int nchunk = windowChunks((long long)cg * pg * 256 * s.nfreq, s.nwin, 64, sms);
    CHECK_CUDA(cudaMemsetAsync(vis, 0, s.visCount() * sizeof(cuFloatComplex), st));
    xmac_tiled_kernel<<<dim3(cg, pg, s.nfreq * nchunk), threads, 0, st>>>(s, (const float2*)spectra, (float2*)vis,
                                                                          pairs, npairs, nchunk);
    CHECK_CUDA(cudaGetLastError());
}

// ---- tensor ----------------------------------------------------------------------------------
// The transpose: block (32 channels, 8 receivers) of one (freq, block of 16 windows); grid
// channel groups x window blocks x (freq, group of 8 receivers).  Reads 32 consecutive channels
// at a time; writes, per plane, the 16 windows (32 bytes) of each of the 8 receivers of a channel,
// 256 contiguous bytes, 16 bytes a thread.
__global__ void __launch_bounds__(256) xmac_transpose_kernel(XmacShape s, TensorLayout L,
                                                             const float2* __restrict__ spec, __half* h) {
    __shared__ float2 sm[8][16][33];
    const int c = blockIdx.x * 32 + threadIdx.x;
    const int tb = blockIdx.y;
    const int groups = (L.rows + 7) / 8;
    const int f = blockIdx.z / groups, r0 = (blockIdx.z % groups) * 8;
    const int r = r0 + threadIdx.y;
#pragma unroll 4
    for (int t = 0; t < 16; t++) {
        const int w = tb * 16 + t;
        float2 v = make_float2(0.f, 0.f);
        if (r < L.rows && w < s.nwin && c < s.nchan) v = spec[s.specIndex(r >> 1, w, 2 * f + (r & 1), c)];
        sm[threadIdx.y][t][threadIdx.x] = v;
    }
    __syncthreads();
    const int k0 = threadIdx.y * 32 + threadIdx.x;
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int k = k0 + 256 * i;                     // (channel, receiver, half), half fastest
        const int hh = k & 1, rr = (k >> 1) & 7, cc = k >> 4;
        const int ch = blockIdx.x * 32 + cc;
        if (ch >= s.nchan || r0 + rr >= L.rows) continue;
        __align__(16) __half2 re[4], im[4];
#pragma unroll
        for (int t2 = 0; t2 < 4; t2++) {
            const float2 a = sm[rr][hh * 8 + 2 * t2][cc], b = sm[rr][hh * 8 + 2 * t2 + 1][cc];
            re[t2] = __floats2half2_rn(a.x, b.x);
            im[t2] = __floats2half2_rn(a.y, b.y);
        }
        *(uint4*)(h + L.index(s, f, ch, tb, 0, r0 + rr, hh * 8)) = *(uint4*)re;
        *(uint4*)(h + L.index(s, f, ch, tb, 1, r0 + rr, hh * 8)) = *(uint4*)im;
    }
}

void xmacTranspose(const XmacShape& s, const TensorLayout& L, const cuFloatComplex* spectra, __half* h,
                   cudaStream_t st) {
    xmac_transpose_kernel<<<dim3((s.nchan + 31) / 32, L.ntb, s.nfreq * ((L.rows + 7) / 8)), dim3(32, 8), 0, st>>>(
        s, L, (const float2*)spectra, h);
    CHECK_CUDA(cudaGetLastError());
}

// The tensor-core XMAC: a warp per (channel, pair of 16-receiver blocks I <= J), 4 warps a block
// (the pairs of one channel together, so that they share its data in L1); grid: (channel, pair)
// groups x freq x window chunks.  Per block of 16 windows, with A the receivers of I (rows) by
// window, B those of J: Re V += Ar Br^T + Ai Bi^T, Im V += Ai Br^T - Ar Bi^T (B loaded column-major
// from the same row-major tile gives the transpose).
__global__ void __launch_bounds__(128) xmac_tensor_kernel(XmacShape s, TensorLayout L, const __half* __restrict__ h,
                                                          float2* vis, const int2* __restrict__ pairs, int npairs,
                                                          int nchunk) {
    using namespace nvcuda;
    __shared__ float sm[4][2][256];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = blockIdx.x * 4 + warp;               // (channel, pair), pair fastest
    const int f = blockIdx.y, chunk = blockIdx.z;
    if (g >= s.nchan * npairs) return;
    const int c = g / npairs;
    const int2 IJ = pairs[g % npairs];
    const int per = (L.ntb + nchunk - 1) / nchunk;
    const int tb0 = chunk * per, tb1 = min(L.ntb, tb0 + per);
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> ar, ai;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> br, bi;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> vr, vi;
    wmma::fill_fragment(vr, 0.f);
    wmma::fill_fragment(vi, 0.f);
    const size_t step = (size_t)2 * L.rows * 16;        // one block of 16 windows
    const __half* p = h + L.index(s, f, c, tb0, 0, 0, 0);
    for (int tb = tb0; tb < tb1; tb++, p += step) {
        wmma::load_matrix_sync(ar, p + IJ.x * 256, 16);
        wmma::load_matrix_sync(ai, p + L.rows * 16 + IJ.x * 256, 16);
        wmma::load_matrix_sync(br, p + IJ.y * 256, 16);
        wmma::load_matrix_sync(bi, p + L.rows * 16 + IJ.y * 256, 16);
        wmma::mma_sync(vr, ar, br, vr);
        wmma::mma_sync(vr, ai, bi, vr);
        wmma::mma_sync(vi, ai, br, vi);
#pragma unroll
        for (int e = 0; e < ar.num_elements; e++) ar.x[e] = __hneg(ar.x[e]);
        wmma::mma_sync(vi, ar, bi, vi);
    }
    wmma::store_matrix_sync(sm[warp][0], vr, 16, wmma::mem_row_major);
    wmma::store_matrix_sync(sm[warp][1], vi, 16, wmma::mem_row_major);
    __syncwarp();
    const int R = s.nrecv();
    for (int e = lane; e < 256; e += 32) {
        const int r1 = IJ.x * 16 + (e >> 4), r2 = IJ.y * 16 + (e & 15);
        if (r1 >= R || r2 >= R || (r1 >> 1) >= (r2 >> 1)) continue;
        const int bl = baselineIndex(r1 >> 1, r2 >> 1, s.nant);
        addVis((cuFloatComplex*)vis, s.visIndex(f, bl, productIndex(r1 & 1, r2 & 1), c), sm[warp][0][e],
               sm[warp][1][e], nchunk > 1);
    }
}

void xmacTensor(const XmacShape& s, const TensorLayout& L, const __half* h, cuFloatComplex* vis, int sms,
                cudaStream_t st) {
    int npairs;
    const int2* pairs = tilePairs((L.rows + 15) / 16, npairs);
    const unsigned nb = (unsigned)((s.nchan * npairs + 3) / 4);
    const int nchunk = windowChunks((long long)nb * 128 * s.nfreq, L.ntb, 4, sms);
    CHECK_CUDA(cudaMemsetAsync(vis, 0, s.visCount() * sizeof(cuFloatComplex), st));
    xmac_tensor_kernel<<<dim3(nb, s.nfreq, nchunk), 128, 0, st>>>(s, L, h, (float2*)vis, pairs, npairs, nchunk);
    CHECK_CUDA(cudaGetLastError());
}

// ---- staged tensor ---------------------------------------------------------------------------
// The tensor-core XMAC from FP16 spectra in the station kernel's order (halfIndex), each block
// doing the reshuffle into the tensor layout itself, in shared memory.
//
// The receivers are taken in blocks of 16 (the tensor cores' rows), and the blocks in super-blocks
// of KS.  A thread block takes CH consecutive channels of one frequency and one pair (A, B), A <= B,
// of super-blocks, and stages only those receivers: A's 16 KS at rows 0 .. 16 KS - 1 of the stage
// and, when B != A, B's at rows 16 KS .. 32 KS - 1.  Its work items are (channel, pair (i, j) of
// 16-blocks, i of A and j of B, i <= j when A == B), their sums in registers over the whole window
// loop.  stagedPlan chooses:
//   up to 32 receivers (16 stations)  one super-block of all of them (KS 1 or 2), 8 channels, a warp
//                                     per channel: the block stages every receiver, as DiFX's usual
//                                     arrays need, and reads each one once
//   more                              KS = 4 (64 receivers), 4 channels, 16 warps (4 a channel):
//                                     at most 49 KB of shared memory a block whatever the number
//                                     of stations.  Each receiver is read by every pair it is in
//                                     (the number of super-blocks), the repeats partly from L2
// (--staged-tile in xmacbench forces KS 1 or 2 with 8 channels, or 4.  On the 2070, KS 4 is 1.1-2.0x
// faster than KS 2 with 8 channels from 24 stations up; KS 1 was 1.6-2.1x slower than KS 2.)
// The channel groups are fastest in the launch, so that the blocks reading neighbouring parts of a
// window's spectra run together (with pair fastest instead: 1.3-1.5x slower from 32 stations up).
//
// Per block of 16 windows the block loads its channels for its receivers into the stage,
// [channel][plane][row][window] (TensorLayout for one channel; rows past the last receiver stay
// zero), then each warp runs its items' WMMA products from there.  A load task is (channel quad,
// pair of windows, row), quad fastest: 16-byte loads, 4 channels of window w and of w + 1, so that
// with 8 channels two neighbouring lanes read one 32-byte sector in the same instruction (a lane
// loading both quads was 1.1-1.2x slower at 10-16 stations); every load of a block of windows is
// issued before the stores into the stage, written as __half2 (real or imaginary parts of the two
// windows, per channel).  Load, barrier, products, barrier.  (Loading the next block of windows into
// registers during the products, with two stages, needed more registers than the 128 a thread
// allowed: it spilled, and was slower.)
// The stage's rows are LD = 24 halves apart (48 bytes): with 16 (32 bytes) WMMA's loads from it had
// 4-way bank conflicts.
constexpr int LD = 24;
static __host__ __device__ int stagedChannelStride(int srows) { return 2 * srows * LD + 16; }     // halves

// KS: 16-receiver blocks per super-block; CH: channels per block (8 or 4).  8 channels: 8 warps, a warp
// per channel; 4 channels: 16 warps, 4 a channel, each taking every 4th of the channel's items.
template<int KS, int CH>
struct StagedCfg {
    static constexpr int NW = CH == 8 ? 8 : 16;          // warps a block
    static constexpr int WPC = NW / CH;                  // warps a channel
    static constexpr int NI = (KS * KS + WPC - 1) / WPC; // items a warp, at most
    static constexpr int NQ = CH / 4;                    // channel quads a block
    static constexpr int MAXT = NQ * 8 * KS / NW;        // load tasks a thread: NQ x 8 window pairs x 32 KS rows
    static constexpr int MINB = NW == 8 ? 2 : 1;         // blocks an SM, for the register budget (128 a thread)
    static_assert(NQ * 8 * 32 * KS % (NW * 32) == 0, "whole load tasks");
};

template<int KS, int CH>
__global__ void __launch_bounds__(StagedCfg<KS, CH>::NW * 32, StagedCfg<KS, CH>::MINB)
xmac_staged_wmma_kernel(XmacShape s, const __half2* __restrict__ spec, float2* vis, const int2* __restrict__ superPairs,
                   int nsp, int srows, int nchunk) {
    using namespace nvcuda;
    using C = StagedCfg<KS, CH>;
    constexpr int NI = C::NI, MAXT = C::MAXT, NQ = C::NQ;
    extern __shared__ __align__(32) unsigned char smraw[];
    const int R = s.nrecv(), nb = (R + 15) / 16, CS = stagedChannelStride(srows);
    __half* stage = reinterpret_cast<__half*>(smraw);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int ncg = (s.nchan + CH - 1) / CH;
    const int sp = blockIdx.x / ncg, c0 = (blockIdx.x % ncg) * CH;      // channel group fastest
    const int f = blockIdx.y;
    const int2 AB = superPairs[sp];
    const bool diag = AB.x == AB.y;
    const int used = diag ? 16 * KS : 32 * KS;                          // stage rows in use
    const int ntb = (s.nwin + 15) / 16;
    const int per = (ntb + nchunk - 1) / nchunk;
    const int tb0 = blockIdx.z * per, tb1 = min(ntb, tb0 + per);
    // stage row -> receiver, or -1 past the last one
    auto recvOf = [&](int row) {
        const int r = row < 16 * KS ? AB.x * 16 * KS + row : AB.y * 16 * KS + row - 16 * KS;
        return r < R ? r : -1;
    };

    // the rows past the last receiver, every channel and plane: zero once (the loads never write them)
    for (int u = threadIdx.x; u < CH * 2 * used * 16; u += blockDim.x) {
        const int t = u % 16, row = (u / 16) % used, cp = u / (16 * used);
        if (recvOf(row) < 0) stage[(cp >> 1) * CS + (cp & 1) * srows * LD + row * LD + t] = __float2half(0.f);
    }

    // the load tasks: (channel quad q, window pair wp, row), q fastest, so that with 8 channels two
    // neighbouring lanes read the two halves of one 32-byte sector in the same instruction (with 4,
    // the neighbouring channel group, next in the launch, reads the other half); a thread's tasks are
    // fixed
    const size_t wstride = (size_t)s.nband() * s.nchan / 4;            // uint4 from one window to the next
    const uint4* src[MAXT];
    int dst[MAXT], wofs[MAXT];
    bool live[MAXT];
#pragma unroll
    for (int i = 0; i < MAXT; i++) {
        const int t = threadIdx.x + i * C::NW * 32;
        const int q = t % NQ, wp = (t / NQ) % 8, row = t / (NQ * 8);
        const int r = row < used ? recvOf(row) : -1;
        live[i] = r >= 0 && c0 + 4 * q < s.nchan;
        wofs[i] = 2 * wp;
        src[i] = reinterpret_cast<const uint4*>(spec + s.halfIndex(r >= 0 ? r >> 1 : 0, 0, 2 * f + (r >= 0 ? r & 1 : 0),
                                                                   c0)) + 2 * wp * wstride + q;
        dst[i] = row * LD + 2 * wp + 4 * q * CS;
    }
    // one block of 16 windows into the stage
    auto load = [&](int tb) {
        const uint4 z = make_uint4(0, 0, 0, 0);
        uint4 va[MAXT], vb[MAXT];
#pragma unroll
        for (int i = 0; i < MAXT; i++) {                   // every load first, then the stores
            const int w = tb * 16 + wofs[i];
            const uint4* p = src[i] + (size_t)tb * 16 * wstride;
            va[i] = live[i] && w < s.nwin ? __ldg(p) : z;
            vb[i] = live[i] && w + 1 < s.nwin ? __ldg(p + wstride) : z;
        }
#pragma unroll
        for (int i = 0; i < MAXT; i++) {
            if (!live[i]) continue;
            const __half2* a = reinterpret_cast<const __half2*>(&va[i]);
            const __half2* b = reinterpret_cast<const __half2*>(&vb[i]);
#pragma unroll
            for (int k = 0; k < 4; k++) {
                __half* d = stage + dst[i] + k * CS;
                *reinterpret_cast<__half2*>(d) = __lows2half2(a[k], b[k]);                  // real parts
                *reinterpret_cast<__half2*>(d + srows * LD) = __highs2half2(a[k], b[k]);    // imaginary parts
            }
        }
    };

    // the warp's items: its channel, every WPC-th 16-block pair (i, j) of (A, B)
    const int cw = warp % CH, sub = warp / CH;
    const int c = c0 + cw;
    int2 IJ[NI];
    int rowI[NI], rowJ[NI];
    bool on[NI];
#pragma unroll
    for (int n = 0; n < NI; n++) {
        const int pidx = sub + C::WPC * n, i = pidx / KS, j = pidx % KS;
        IJ[n] = make_int2(AB.x * KS + i, AB.y * KS + j);
        rowI[n] = 16 * i;
        rowJ[n] = diag ? 16 * j : 16 * KS + 16 * j;
        on[n] = pidx < KS * KS && c < s.nchan && (!diag || i <= j) && IJ[n].x < nb && IJ[n].y < nb;
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> vr[NI], vi[NI];
#pragma unroll
    for (int n = 0; n < NI; n++) { wmma::fill_fragment(vr[n], 0.f); wmma::fill_fragment(vi[n], 0.f); }
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> ar, ai;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> br, bi;

    const __half* ps = stage + cw * CS;
    for (int tb = tb0; tb < tb1; tb++) {
        load(tb);
        __syncthreads();
#pragma unroll
        for (int n = 0; n < NI; n++) {
            if (!on[n]) continue;
            wmma::load_matrix_sync(ar, ps + rowI[n] * LD, LD);
            wmma::load_matrix_sync(ai, ps + (srows + rowI[n]) * LD, LD);
            wmma::load_matrix_sync(br, ps + rowJ[n] * LD, LD);
            wmma::load_matrix_sync(bi, ps + (srows + rowJ[n]) * LD, LD);
            wmma::mma_sync(vr[n], ar, br, vr[n]);
            wmma::mma_sync(vr[n], ai, bi, vr[n]);
            wmma::mma_sync(vi[n], ai, br, vi[n]);
#pragma unroll
            for (int e = 0; e < ar.num_elements; e++) ar.x[e] = __hneg(ar.x[e]);
            wmma::mma_sync(vi[n], ar, bi, vi[n]);
        }
        __syncthreads();                                    // done with the stage
    }

    // the sums out, through a 2 KB scratch per warp (the stage is free once all warps are here)
    __syncthreads();
    float* sm = reinterpret_cast<float*>(smraw) + warp * 512;
#pragma unroll
    for (int n = 0; n < NI; n++) {
        if (!on[n]) continue;
        wmma::store_matrix_sync(sm, vr[n], 16, wmma::mem_row_major);
        wmma::store_matrix_sync(sm + 256, vi[n], 16, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 256; e += 32) {
            const int r1 = IJ[n].x * 16 + (e >> 4), r2 = IJ[n].y * 16 + (e & 15);
            if (r1 >= R || r2 >= R || (r1 >> 1) >= (r2 >> 1)) continue;
            const int bl = baselineIndex(r1 >> 1, r2 >> 1, s.nant);
            addVis((cuFloatComplex*)vis, s.visIndex(f, bl, productIndex(r1 & 1, r2 & 1), c), sm[e], sm[256 + e],
                   nchunk > 1);     // the pairs and channels are each one block's: only window chunks share sums
        }
        __syncwarp();
    }
}

// ---- staged tensor, mma.sync: the default ------------------------------------------------------
// The same blocks, super-blocks and items as the WMMA version above, with three changes:
//  - The stage holds the spectra as the station kernel wrote them, (re, im) interleaved: a row (one
//    receiver, one channel) is 16 windows x (re, im) FP16, 64 bytes, nothing rearranged.  With the
//    pairs along the sum dimension (32 values: re_0, im_0, re_1, ...), one product over a row gives
//    sum(re1 re2 + im1 im2), the real part of X1 X2^*; the imaginary part, sum(im1 re2 - re1 im2),
//    is the product with A's pairs turned into (im, -re), one byte permute and one sign flip per
//    register.  The same four tensor-core multiplies per 16 windows as before.
//  - PTX: ldmatrix and mma.sync.m16n8k16 (on sm_75, two m16n8k8), whose register layouts are
//    documented: each 32-bit register of a fragment holds one (re, im) pair.  Each warp loads each
//    of its A and B fragments once per 8 windows and uses it for every item that needs it, and its
//    sums go to vis straight from the accumulator registers.
//  - The copy into the stage: a task is (channel quad, window, row), 16 bytes from global memory, 4
//    channels of one window, going to 4 channels' stages, 4 bytes each.  On sm_80 and later 4
//    cp.async a task (zero-filled past the last window), with no registers in between, into two
//    stages: the next 16 windows load while the tensor cores work on these, one barrier per 16
//    windows.  On sm_75 (no cp.async) a 16-byte load and 4 stores, one stage, two barriers.  (4 bytes
//    a task, its own address arithmetic each, ran 2.3x the instructions and was 1.1-1.6x slower on the
//    2070.)
// A row's four 16-byte chunks (4 windows each) are XOR-swizzled by its row (chunk q at q ^ ((row >> 1)
// & 3)), so that ldmatrix's 8 rows of a phase fall in 8 different bank groups without padding; the
// channels' stages are 16 bytes apart beyond a multiple of 128, so that the copy's stores (lanes:
// quad fastest, then window) fall in different banks too.
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
#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cpAsync4(unsigned char* dst, const void* src, bool valid) {   // zero-fill if !valid
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                 :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src), "r"(valid ? 4 : 0));
}
__device__ __forceinline__ void cpCommit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cpWaitAll() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
#endif

template<int KS, int CH>
struct MmaCfg {
    static constexpr int NW = CH == 8 ? 8 : 16;          // warps a block
    static constexpr int WPC = NW / CH;                  // warps a channel: 1, or KS (each its own j)
    static constexpr int NJ = WPC == 1 ? KS : 1;         // J blocks a warp
    static constexpr int MINB = NW == 8 ? 2 : 1;         // blocks an SM, for the register budget
    static constexpr int NQ = CH / 4;                    // channel quads a block
    static constexpr int RSTEP = NW * 32 / (NQ * 16);    // copy: rows between a thread's tasks
    static constexpr int MAXR = 32 * KS / RSTEP;         // copy: tasks a thread, at most
    static_assert(WPC == 1 || WPC == KS, "a warp per channel, or one per J block");
};

template<int KS, int CH>
__global__ void __launch_bounds__(MmaCfg<KS, CH>::NW * 32, MmaCfg<KS, CH>::MINB)
xmac_staged_kernel(XmacShape s, const __half2* __restrict__ spec, float2* vis, const int2* __restrict__ superPairs,
                   int nsp, int srows, int nbuf, int nchunk) {
    using C = MmaCfg<KS, CH>;
    constexpr int NJ = C::NJ;
    extern __shared__ __align__(32) unsigned char smraw[];
    const int CSb = stagedChannelBytes(srows), BUFb = CH * CSb;
    const __half2** rowSrc = reinterpret_cast<const __half2**>(smraw + nbuf * BUFb);
    const int R = s.nrecv(), nb = (R + 15) / 16;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int ncg = (s.nchan + CH - 1) / CH;
    const int sp = blockIdx.x / ncg, c0 = (blockIdx.x % ncg) * CH;      // channel group fastest
    const int f = blockIdx.y;
    const int2 AB = superPairs[sp];
    const bool diag = AB.x == AB.y;
    const int used = diag ? 16 * KS : 32 * KS;                          // stage rows in use
    const int ntb = (s.nwin + 15) / 16;
    const int per = (ntb + nchunk - 1) / nchunk;
    const int tb0 = blockIdx.z * per, tb1 = min(ntb, tb0 + per);
    auto recvOf = [&](int row) {                                        // stage row -> receiver, or -1
        const int r = row < 16 * KS ? AB.x * 16 * KS + row : AB.y * 16 * KS + row - 16 * KS;
        return r < R ? r : -1;
    };

    // each stage row's spectra (this block's channels, window 0), or none past the last receiver; those
    // rows are zeroed once, in every stage and channel
    for (int row = threadIdx.x; row < used; row += blockDim.x) {
        const int r = recvOf(row);
        rowSrc[row] = r >= 0 ? spec + s.halfIndex(r >> 1, 0, 2 * f + (r & 1), c0) : nullptr;
    }
    for (int u = threadIdx.x; u < nbuf * CH * used * 16; u += blockDim.x) {
        const int word = u % 16, row = (u / 16) % used, bc = u / (16 * used);
        if (recvOf(row) < 0) *reinterpret_cast<uint32_t*>(smraw + bc * CSb + row * SROW + 4 * word) = 0u;
    }
    __syncthreads();

    // the copy tasks: (channel quad q, window w, row), q fastest then w; a thread's q and w are fixed,
    // its rows row0, row0 + RSTEP, ...
    const int q = threadIdx.x % C::NQ, w = (threadIdx.x / C::NQ) % 16, row0 = threadIdx.x / (C::NQ * 16);
    const size_t wstride = (size_t)s.nband() * s.nchan;                // __half2 from one window to the next
    const bool quadOk = c0 + 4 * q < s.nchan;                          // (nchan is a multiple of 4)
    auto copy = [&](int tb, int buf) {
        const int wg = tb * 16 + w;
        const bool ok = quadOk && wg < s.nwin;
        const size_t off = (size_t)wg * wstride + 4 * q;
        unsigned char* d0 = smraw + buf * BUFb + 4 * q * CSb + ((w & 3) << 2);
#if __CUDA_ARCH__ >= 800
#pragma unroll
        for (int k = 0; k < C::MAXR; k++) {
            const int row = row0 + k * C::RSTEP;
            if (row >= used) break;
            const __half2* p = rowSrc[row];
            if (!p) continue;
            const __half2* src = ok ? p + off : spec;
            unsigned char* d = d0 + stagedChunk(row, w >> 2);
#pragma unroll
            for (int cq = 0; cq < 4; cq++) cpAsync4(d + cq * CSb, src + (ok ? cq : 0), ok);
        }
#else
        constexpr int NB = C::MAXR < 4 ? C::MAXR : 4;                  // 16-byte loads in flight, then their stores
#pragma unroll
        for (int k0 = 0; k0 < C::MAXR; k0 += NB) {
            uint4 v[NB];
#pragma unroll
            for (int k = 0; k < NB; k++) {
                const int row = row0 + (k0 + k) * C::RSTEP;
                const __half2* p = row < used ? rowSrc[row] : nullptr;
                v[k] = p && ok ? __ldg(reinterpret_cast<const uint4*>(p + off)) : make_uint4(0, 0, 0, 0);
            }
#pragma unroll
            for (int k = 0; k < NB; k++) {
                const int row = row0 + (k0 + k) * C::RSTEP;
                if (row >= used || !rowSrc[row]) continue;
                unsigned char* d = d0 + stagedChunk(row, w >> 2);
                *reinterpret_cast<uint32_t*>(d) = v[k].x;
                *reinterpret_cast<uint32_t*>(d + CSb) = v[k].y;
                *reinterpret_cast<uint32_t*>(d + 2 * CSb) = v[k].z;
                *reinterpret_cast<uint32_t*>(d + 3 * CSb) = v[k].w;
            }
        }
#endif
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
    auto compute = [&](int buf) {
        const unsigned char* base = smraw + buf * BUFb + cw * CSb;
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {                                // windows 8 ks .. 8 ks + 7
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

#if __CUDA_ARCH__ >= 800
    if (tb0 < tb1) { copy(tb0, 0); cpCommit(); }
    for (int tb = tb0; tb < tb1; tb++) {
        const int buf = (tb - tb0) & 1;
        cpWaitAll();
        __syncthreads();                         // this stage is in; every warp is done with the other one
        if (tb + 1 < tb1) { copy(tb + 1, buf ^ 1); cpCommit(); }
        compute(buf);
    }
#else
    for (int tb = tb0; tb < tb1; tb++) {
        copy(tb, 0);
        __syncthreads();
        compute(0);
        __syncthreads();                         // done with the stage
    }
#endif

    // the sums out, from the accumulators: register e of n half h is row g + 8 (e >> 1), column
    // 8 h + 2 t + (e & 1) of the 16 x 16 tile
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
                    const int bl = baselineIndex(r1 >> 1, r2 >> 1, s.nant);
                    addVis((cuFloatComplex*)vis, s.visIndex(f, bl, productIndex(r1 & 1, r2 & 1), c), acc[i][jj][0][h][e],
                           acc[i][jj][1][h][e], nchunk > 1);
                }
        }
}

// The plan for a shape: KS, channels a block, the super-blocks, the stage's rows, the warps, and the
// shared memory a block (the version's own).
struct StagedPlan { int ks, ch, nsuper, srows, nw, nbuf; size_t smem; };
static int stagedForcedKS = 0;                 // xmacTensorStagedSetTile; 0: stagedPlan chooses
static int stagedImpl = 0;                     // xmacTensorStagedSetImpl: 0 mma.sync, 1 WMMA
void xmacTensorStagedSetTile(int ks) { stagedForcedKS = ks; }
void xmacTensorStagedSetImpl(int impl) { stagedImpl = impl; }

static int deviceAttr(cudaDeviceAttr a) {
    int dev, v;
    CHECK_CUDA(cudaGetDevice(&dev));
    CHECK_CUDA(cudaDeviceGetAttribute(&v, a, dev));
    return v;
}
static StagedPlan stagedPlan(const XmacShape& s) {
    const int nb = (s.nrecv() + 15) / 16;
    static int major = 0;
    if (!major) major = deviceAttr(cudaDevAttrComputeCapabilityMajor);
    auto plan = [&](int ks) {
        StagedPlan p;
        p.ks = ks;
        p.ch = ks == 4 ? 4 : 8;
        p.nw = p.ch == 8 ? 8 : 16;
        p.nsuper = (nb + ks - 1) / ks;
        p.srows = (p.nsuper == 1 ? 16 : 32) * ks;
        if (stagedImpl == 1) {                                          // WMMA: one stage, planes, padded rows
            p.nbuf = 1;
            p.smem = std::max((size_t)p.ch * stagedChannelStride(p.srows) * sizeof(__half),
                              (size_t)p.nw * 512 * sizeof(float));
        } else {                                                        // mma.sync: two stages from sm_80
            p.nbuf = major >= 8 ? 2 : 1;
            p.smem = (size_t)p.nbuf * p.ch * stagedChannelBytes(p.srows) + (size_t)p.srows * sizeof(void*);
        }
        return p;
    };
    if (stagedForcedKS) return plan(stagedForcedKS);
    if (nb <= 2) return plan(nb);                                       // one super-block: every receiver
    return plan(4);                                                     // else 64-receiver super-blocks, 4 channels
}

int xmacTensorStagedTile(const XmacShape& s) { return stagedPlan(s).ks; }
bool xmacTensorStagedFits(const XmacShape& s) {
    static int maxSmem = 0;
    if (!maxSmem) maxSmem = deviceAttr(cudaDevAttrMaxSharedMemoryPerBlockOptin);
    return stagedPlan(s).smem <= (size_t)maxSmem;
}

template<int KS, int CH>
static void launchStaged(const XmacShape& s, const StagedPlan& P, const __half2* spec, cuFloatComplex* vis, int sms,
                         cudaStream_t st) {
    static size_t smemSet[2] = {0, 0};
    auto kernel = stagedImpl == 1 ? nullptr : xmac_staged_kernel<KS, CH>;
    auto kernelW = xmac_staged_wmma_kernel<KS, CH>;
    size_t& set = smemSet[stagedImpl == 1];
    if (P.smem > set) {
        if (kernel) CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)P.smem));
        else CHECK_CUDA(cudaFuncSetAttribute(kernelW, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)P.smem));
        set = P.smem;
    }
    int nsp;
    const int2* superPairs = tilePairs(P.nsuper, nsp);
    const int threads = P.nw * 32;
    const unsigned nx = (unsigned)((s.nchan + CH - 1) / CH * nsp);
    const int ntb = (s.nwin + 15) / 16;
    const int nchunk = windowChunks((long long)nx * s.nfreq * threads, ntb, 4, sms);
    CHECK_CUDA(cudaMemsetAsync(vis, 0, s.visCount() * sizeof(cuFloatComplex), st));
    if (kernel)
        kernel<<<dim3(nx, s.nfreq, nchunk), threads, P.smem, st>>>(s, spec, (float2*)vis, superPairs, nsp, P.srows,
                                                                    P.nbuf, nchunk);
    else
        kernelW<<<dim3(nx, s.nfreq, nchunk), threads, P.smem, st>>>(s, spec, (float2*)vis, superPairs, nsp, P.srows,
                                                                     nchunk);
    CHECK_CUDA(cudaGetLastError());
}

void xmacTensorStaged(const XmacShape& s, const __half2* spec, cuFloatComplex* vis, int sms, cudaStream_t st) {
    if (s.nchan % 4) { fprintf(stderr, "staged tensor XMAC: the channel count must be a multiple of 4\n"); exit(2); }
    const StagedPlan P = stagedPlan(s);
    if (!xmacTensorStagedFits(s)) {
        fprintf(stderr, "staged tensor XMAC: %zu bytes of shared memory a block (KS %d), more than a block may have\n",
                P.smem, P.ks);
        exit(2);
    }
    if (P.ks == 1) launchStaged<1, 8>(s, P, spec, vis, sms, st);
    else if (P.ks == 2) launchStaged<2, 8>(s, P, spec, vis, sms, st);
    else if (P.ks == 4) launchStaged<4, 4>(s, P, spec, vis, sms, st);
    else { fprintf(stderr, "staged tensor XMAC: KS must be 1, 2 or 4, not %d\n", P.ks); exit(2); }
}
