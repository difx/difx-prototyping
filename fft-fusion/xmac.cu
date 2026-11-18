// xmac.cu - the three XMAC versions of xmac.cuh.
#include <algorithm>
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
// doing the reshuffle into the tensor layout itself, in shared memory.  Block: SK = 8 consecutive
// channels of one frequency, 8 warps; its work items are (channel, pair of 16-receiver blocks
// I <= J), pair fastest, up to SIPW a warp (warp w takes items w, w + 8, ...), their sums kept in
// registers over the whole window loop.  When a channel group has more items than one block takes,
// they are split over several blocks (grid y), each loading only the channel quads its items use.
//
// Per block of 16 windows the block loads its channels for every receiver into a stage in shared
// memory, [channel][plane][receiver][window] (TensorLayout for one channel; the rows up to the
// next multiple of 16 stay zero, so the spare products are of zeros), then each warp runs its
// items' WMMA products from there.  A load task is (pair of windows, receiver): for each of the
// block's channel quads (one or two) two 16-byte loads, 4 channels of window w and of w + 1, so a
// lane reads whole 32-byte sectors; written as __half2 (real or imaginary parts of the two
// windows, per channel).  Lanes take the 8 window pairs fastest, then 4 receivers, so a warp's
// stores fall in 32 different banks.  Load, barrier, products, barrier: the other block on the SM
// (at most 128 registers a thread for two) runs while this one waits.  (Loading the next block of
// windows into registers during the products, with two stages, needed more registers than that:
// it spilled, and was slower.)
// The stage's rows are LD = 24 halves apart (48 bytes): with 16 (32 bytes) WMMA's loads from it had
// 4-way bank conflicts.
constexpr int SK = 8, SW = 8, SIPW = 4, LD = 24;
static __host__ __device__ int stagedRows(int R) { return (R + 15) / 16 * 16; }
static __host__ __device__ int stagedChannelStride(int rows) { return 2 * rows * LD + 16; }     // halves

template<int MAXT>
__global__ void __launch_bounds__(SW * 32, 2) xmac_staged_kernel(XmacShape s, const __half2* __restrict__ spec,
                                                              float2* vis, const int2* __restrict__ pairs,
                                                              int npairs, int ngroup, int nchunk) {
    using namespace nvcuda;
    extern __shared__ __align__(32) unsigned char smraw[];
    const int R = s.nrecv(), rows = stagedRows(R), CS = stagedChannelStride(rows);
    __half* stage = reinterpret_cast<__half*>(smraw);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int c0 = blockIdx.x * SK;
    const int f = blockIdx.y / ngroup, group = blockIdx.y % ngroup;
    const int ntb = (s.nwin + 15) / 16;
    const int per = (ntb + nchunk - 1) / nchunk;
    const int tb0 = blockIdx.z * per, tb1 = min(ntb, tb0 + per);
    const int nitems = SK * npairs;
    const int item0 = group * SW * SIPW;
    // the channel quads this block's items use
    const int q0 = (item0 / npairs) / 4, q1 = (min(nitems, item0 + SW * SIPW) - 1) / npairs / 4;

    // the spare rows (R .. rows - 1) of every channel and plane: zero once
    for (int u = threadIdx.x; u < SK * 2 * (rows - R) * 16; u += blockDim.x) {
        const int t = u % 16, r = R + (u / 16) % (rows - R), cp = u / (16 * (rows - R));
        stage[(cp >> 1) * CS + (cp & 1) * rows * LD + r * LD + t] = __float2half(0.f);
    }

    // the load tasks: (window pair wp, receiver r), wp fastest; a thread's tasks are fixed
    const size_t wstride = (size_t)s.nband() * s.nchan / 4;            // uint4 from one window to the next
    bool quad[2];                                                      // the block's quads that exist
#pragma unroll
    for (int q = 0; q < 2; q++) quad[q] = q >= q0 && q <= q1 && c0 + 4 * q < s.nchan;
    const uint4* src[MAXT];
    int dst[MAXT], wofs[MAXT];
    bool live[MAXT];
#pragma unroll
    for (int i = 0; i < MAXT; i++) {
        const int t = threadIdx.x + i * SW * 32;
        const int wp = t % 8, r = t / 8;
        live[i] = r < R;
        wofs[i] = 2 * wp;
        src[i] = reinterpret_cast<const uint4*>(spec + s.halfIndex(live[i] ? r >> 1 : 0, 0,
                                                                   2 * f + (r & 1), c0)) + 2 * wp * wstride;
        dst[i] = r * LD + 2 * wp;
    }
    // one block of 16 windows into the stage
    auto load = [&](int tb) {
        const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int i = 0; i < MAXT; i++) {
            if (!live[i]) continue;
            const int w = tb * 16 + wofs[i];
            const uint4* p = src[i] + (size_t)tb * 16 * wstride;
            const bool okA = w < s.nwin, okB = w + 1 < s.nwin;
#pragma unroll
            for (int q = 0; q < 2; q++) {
                if (!quad[q]) continue;
                const uint4 va = okA ? __ldg(p + q) : z, vb = okB ? __ldg(p + wstride + q) : z;
                const __half2* a = reinterpret_cast<const __half2*>(&va);
                const __half2* b = reinterpret_cast<const __half2*>(&vb);
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    __half* d = stage + dst[i] + (4 * q + k) * CS;
                    *reinterpret_cast<__half2*>(d) = __lows2half2(a[k], b[k]);                 // real parts
                    *reinterpret_cast<__half2*>(d + rows * LD) = __highs2half2(a[k], b[k]);    // imaginary parts
                }
            }
        }
    };

    int cl[SIPW];
    int2 IJ[SIPW];
    bool on[SIPW];
#pragma unroll
    for (int j = 0; j < SIPW; j++) {
        const int it = item0 + warp + SW * j;
        cl[j] = it / npairs;
        on[j] = it < nitems && c0 + cl[j] < s.nchan;
        IJ[j] = on[j] ? pairs[it % npairs] : make_int2(0, 0);
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> vr[SIPW], vi[SIPW];
#pragma unroll
    for (int j = 0; j < SIPW; j++) { wmma::fill_fragment(vr[j], 0.f); wmma::fill_fragment(vi[j], 0.f); }
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> ar, ai;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> br, bi;

    for (int tb = tb0; tb < tb1; tb++) {
        load(tb);
        __syncthreads();
#pragma unroll
        for (int j = 0; j < SIPW; j++) {
            if (!on[j]) continue;
            const __half* p = stage + cl[j] * CS;
            wmma::load_matrix_sync(ar, p + IJ[j].x * 16 * LD, LD);
            wmma::load_matrix_sync(ai, p + rows * LD + IJ[j].x * 16 * LD, LD);
            wmma::load_matrix_sync(br, p + IJ[j].y * 16 * LD, LD);
            wmma::load_matrix_sync(bi, p + rows * LD + IJ[j].y * 16 * LD, LD);
            wmma::mma_sync(vr[j], ar, br, vr[j]);
            wmma::mma_sync(vr[j], ai, bi, vr[j]);
            wmma::mma_sync(vi[j], ai, br, vi[j]);
#pragma unroll
            for (int e = 0; e < ar.num_elements; e++) ar.x[e] = __hneg(ar.x[e]);
            wmma::mma_sync(vi[j], ar, bi, vi[j]);
        }
        __syncthreads();                                    // done with the stage
    }

    // the sums out, through a 2 KB scratch per warp (the stage is free once all warps are here)
    __syncthreads();
    float* sm = reinterpret_cast<float*>(smraw) + warp * 512;
#pragma unroll
    for (int j = 0; j < SIPW; j++) {
        if (!on[j]) continue;
        wmma::store_matrix_sync(sm, vr[j], 16, wmma::mem_row_major);
        wmma::store_matrix_sync(sm + 256, vi[j], 16, wmma::mem_row_major);
        __syncwarp();
        const int c = c0 + cl[j];
        for (int e = lane; e < 256; e += 32) {
            const int r1 = IJ[j].x * 16 + (e >> 4), r2 = IJ[j].y * 16 + (e & 15);
            if (r1 >= R || r2 >= R || (r1 >> 1) >= (r2 >> 1)) continue;
            const int bl = baselineIndex(r1 >> 1, r2 >> 1, s.nant);
            addVis((cuFloatComplex*)vis, s.visIndex(f, bl, productIndex(r1 & 1, r2 & 1), c), sm[e], sm[256 + e],
                   nchunk > 1);     // the groups take different items: only window chunks share sums
        }
        __syncwarp();
    }
}

// the staged XMAC's shared memory per block, and the most a block may have on this device
static size_t stagedSmem(const XmacShape& s) {
    const size_t stageBytes = (size_t)SK * stagedChannelStride(stagedRows(s.nrecv())) * sizeof(__half);
    return std::max(stageBytes, (size_t)SW * 512 * sizeof(float));
}
static int maxBlockSmem() {
    int dev, maxSmem;
    CHECK_CUDA(cudaGetDevice(&dev));
    CHECK_CUDA(cudaDeviceGetAttribute(&maxSmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    return maxSmem;
}

bool xmacTensorStagedFits(const XmacShape& s) { return stagedSmem(s) <= (size_t)maxBlockSmem(); }

template<int MAXT>
static void launchStaged(const XmacShape& s, const __half2* spec, cuFloatComplex* vis, const int2* pairs, int npairs,
                         int ngroup, int sms, cudaStream_t st) {
    static int maxSmem = 0;
    static size_t smemSet = 0;
    if (!maxSmem) maxSmem = maxBlockSmem();
    const size_t smem = stagedSmem(s);
    if (smem > (size_t)maxSmem) {
        fprintf(stderr, "staged tensor XMAC: %zu bytes of shared memory for %d receivers, more than a block's %d\n",
                smem, s.nrecv(), maxSmem);
        exit(2);
    }
    if (smem > smemSet) {
        CHECK_CUDA(cudaFuncSetAttribute(xmac_staged_kernel<MAXT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        smemSet = smem;
    }
    const unsigned cg = (s.nchan + SK - 1) / SK;
    const int ntb = (s.nwin + 15) / 16;
    const int nchunk = windowChunks((long long)cg * s.nfreq * ngroup * SW * 32, ntb, 4, sms);
    CHECK_CUDA(cudaMemsetAsync(vis, 0, s.visCount() * sizeof(cuFloatComplex), st));
    xmac_staged_kernel<MAXT><<<dim3(cg, s.nfreq * ngroup, nchunk), SW * 32, smem, st>>>(
        s, spec, (float2*)vis, pairs, npairs, ngroup, nchunk);
    CHECK_CUDA(cudaGetLastError());
}

void xmacTensorStaged(const XmacShape& s, const __half2* spec, cuFloatComplex* vis, int sms, cudaStream_t st) {
    if (s.nchan % 4) { fprintf(stderr, "staged tensor XMAC: the channel count must be a multiple of 4\n"); exit(2); }
    int npairs;
    const int2* pairs = tilePairs(stagedRows(s.nrecv()) / 16, npairs);
    const int ngroup = (SK * npairs + SW * SIPW - 1) / (SW * SIPW);
    const int tasks = 8 * s.nrecv();                              // (window pair, receiver)
    if (tasks <= SW * 32) launchStaged<1>(s, spec, vis, pairs, npairs, ngroup, sms, st);
    else if (tasks <= 2 * SW * 32) launchStaged<2>(s, spec, vis, pairs, npairs, ngroup, sms, st);
    else if (tasks <= 4 * SW * 32) launchStaged<4>(s, spec, vis, pairs, npairs, ngroup, sms, st);
    else launchStaged<8>(s, spec, vis, pairs, npairs, ngroup, sms, st);
}
