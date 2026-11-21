// xmac.cuh - the cross-multiply-accumulate (XMAC) stage: the station spectra the fused kernel
// writes, cross-multiplied for every baseline and polarisation product and summed over the
// subint's FFT windows.  Two versions (xmac.cu), timed by xmacbench.cu:
//   gpudifx  gpudifx's kernel verbatim (xmac_copy.cuh), launched as GPUCore launches it: a thread
//            per (baseline, product, channel), each reading both stations' FP32 spectra for
//            every window.  The reference.
//   staged   FP16 in, FP32 sums on the tensor cores, from FP16 spectra in the station kernel's own
//            order (halfIndex, or anywhere a map says), each block rearranging its channels of the
//            receivers it multiplies in shared memory.
// (Retired after the comparisons in xmac/docs/ (xmac-paths.md and the rest): a register-tiled FP32 XMAC,
// a transpose into a tensor layout with a WMMA XMAC from it, the station kernel writing that layout
// directly, and a WMMA version of the staged XMAC.)
#pragma once
#include <vector>
#include <cuda_fp16.h>
#include "common.cuh"

// The input: nant stations' spectra, [ant][win][band][fftchannels] FP32 complex, the first nchan
// channels of each used (the fused kernel's output; fftchannels = 2 nchan for real data).  Bands
// 2f and 2f+1 are the two polarisations, A and B, of frequency f.  A window invalid at a station
// (valid[ant][win] false) must add nothing to its baselines: gpudifx's kernel skips it, the staged
// one takes it to be zero, which the station processing would write (it knows the window is
// invalid), so that it need not read the flags per sample.
//
// The output: visibilities [freq][baseline][product][nchan] FP32 complex, summed over the windows.
// Baselines (a, b), a < b, in the order (0,1), (0,2), ..., (0,n-1), (1,2), ...; products
// 0 (A,A), 1 (B,B), 2 (A,B), 3 (B,A), each X_a(pol1) conj(X_b(pol2)).  No autocorrelations yet.
struct XmacShape {
    int nant, nfreq, nchan, nwin, fftchannels;
    __host__ __device__ int nband() const { return 2 * nfreq; }
    __host__ __device__ int nbaseline() const { return nant * (nant - 1) / 2; }
    __host__ __device__ int nrecv() const { return 2 * nant; }        // receivers: (station, pol)
    __host__ __device__ size_t visCount() const { return (size_t)nfreq * nbaseline() * 4 * nchan; }
    __host__ __device__ size_t visIndex(int f, int bl, int p, int c) const {
        return (((size_t)f * nbaseline() + bl) * 4 + p) * nchan + c;
    }
    __host__ __device__ size_t specIndex(int ant, int win, int band, int c) const {
        return (((size_t)ant * nwin + win) * nband() + band) * fftchannels + c;
    }
    // the FP16 spectra for the staged tensor XMAC: [ant][win][band][nchan] __half2 (re, im)
    __host__ __device__ size_t halfIndex(int ant, int win, int band, int c) const {
        return (((size_t)ant * nwin + win) * nband() + band) * nchan + c;
    }
    __host__ __device__ size_t halfCount() const { return (size_t)nant * nwin * nband() * nchan; }
    // complex multiply-accumulates per subint (the work, for throughput figures)
    double cmacs() const { return (double)nfreq * nchan * nwin * nbaseline() * 4; }
};
__host__ __device__ inline int baselineIndex(int a, int b, int nant) {      // a < b
    return a * (2 * nant - a - 1) / 2 + (b - a - 1);
}
__host__ __device__ inline int productIndex(int pol1, int pol2) {
    return pol1 == pol2 ? pol1 : 2 + pol1;
}

// gpudifx's kernel, one launch per frequency as GPUCore makes them; vis is zeroed first.
struct XmacGpudifx {
    XmacShape s{};
    int sms = 0;
    // per-frequency launch metadata, as GPUCore::buildXmacPlans uploads it
    const cuFloatComplex** m1 = nullptr, ** m2 = nullptr;     // [baseline] station spectra
    const bool** v1 = nullptr, ** v2 = nullptr;               // [baseline] station valid flags
    int* b1 = nullptr, * b2 = nullptr, * offsets = nullptr;    // [freq][baseline][product]
    int* bandStride = nullptr, * windowStride = nullptr;      // [baseline], the first station's
    int* bandStride2 = nullptr, * windowStride2 = nullptr;    // [baseline], the second station's
    void init(const XmacShape& shape, const cuFloatComplex* spectra, const bool* valid, int multiprocessors);
    // the tables given directly (heterogeneous stations: xmacbench --messy); each vector as above, the
    // offsets in vis, which needs room for whatever offsets point past visCount()
    struct Tables {
        std::vector<const cuFloatComplex*> m1, m2;
        std::vector<const bool*> v1, v2;
        std::vector<int> b1, b2, offsets, bandStride, windowStride, bandStride2, windowStride2;
    };
    void init(const XmacShape& shape, const Tables& t, int multiprocessors);
    void run(cuFloatComplex* vis, cudaStream_t st) const;
    void destroy();
};

// The staged XMAC splits the windows over several blocks when the baselines and channels alone
// would not fill the GPU (as gpudifx's launch does), and adds the partial sums atomically: vis is
// zeroed first.
// Where the staged XMAC finds each receiver's FP16 spectra, per correlated frequency: the stations
// need not share one layout (different bandwidths and numbers of bands, a frequency some stations
// lack).  Per frequency f, receivers r = 2 k + pol for the k-th station present (stations in
// increasing order), nrecv[f] of them:
//   off[f][r]      __half2 offset of window 0, channel 0 of f in that station's spectra
//   wstride[f][r]  __half2 from one window to the next
//   station[f][k]  the station (for the baseline index of the output, [freq][baseline][product][chan]
//                  as XmacShape::visIndex with all nant stations; baselines a frequency lacks stay zero)
// Offsets and strides must be multiples of 4 (16-byte loads of 4 channels).  The regular layout
// (halfIndex) is the special case xmacRegularMap.
struct XmacMapHost {
    int rmax = 0;                                // receivers a frequency, at most
    std::vector<int> nrecv, wstride, station;    // [freq], [freq][rmax], [freq][rmax / 2]
    std::vector<long long> off;                  // [freq][rmax]
};
struct XmacMap {                                 // the same on the device
    int rmax = 0;
    long long maxOff = 0, maxStride = 0;         // (host) the largest offset and window stride
    const int* nrecv = nullptr, * wstride = nullptr, * station = nullptr;
    const long long* off = nullptr;
};
XmacMapHost xmacRegularMap(const XmacShape& s);
XmacMap xmacUploadMap(const XmacMapHost& h);
void xmacFreeMap(XmacMap& m);

// The staged tensor version: FP16 spectra (halfIndex; windows past the subint need not exist) ->
// vis.  Each block stages a few channels of the receivers it multiplies: every receiver up to 32 (8
// channels), else two super-blocks of 64 receivers (KS 4, 4 channels), so the shared memory a block
// needs (at most 33 KB a stage) does not grow with the number of stations.  xmacTensorStagedTile is
// the KS it uses for a shape, xmacTensorStagedStages its number of stages (1, or 2 where they cost
// no occupancy); xmacTensorStagedSetTile(1, 2 or 4) forces KS (0: choose again), for comparisons;
// xmacTensorStagedFits says whether the plan fits a block on this device.
void xmacTensorStaged(const XmacShape& s, const __half2* spec, cuFloatComplex* vis, int multiprocessors,
                      cudaStream_t st);
// The same with the spectra where a map says (the mma.sync version; spec is the base the offsets are from).
void xmacTensorStagedMapped(const XmacShape& s, const XmacMap& m, const __half2* spec, cuFloatComplex* vis,
                            int multiprocessors, cudaStream_t st);
int xmacTensorStagedTile(const XmacShape& s);
int xmacTensorStagedStages(const XmacShape& s);
void xmacTensorStagedSetTile(int ks);
bool xmacTensorStagedFits(const XmacShape& s);
