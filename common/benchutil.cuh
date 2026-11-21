// benchutil.cuh - what every benchmark here uses (fusebench, xmacbench, twostagebench): the random
// numbers, device buffers, the packed (VDIF) data's format and the host's own decode of it, and the
// timing.  Host code (splitmix also on the device).
#pragma once
#include <algorithm>
#include <cstring>
#include <vector>
#include "common.cuh"
#include "decode.cuh"

// ---- random numbers ----------------------------------------------------------------------------
__host__ __device__ inline unsigned long long splitmix(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

struct Rng {
    unsigned long long s;
    double uniform() { s = splitmix(s); return (s >> 11) * (1.0 / 9007199254740992.0); }
    double range(double a, double b) { return a + (b - a) * uniform(); }
};

// ---- device buffers ----------------------------------------------------------------------------
template<class T> T* devAlloc(size_t n) { T* p; CHECK_CUDA(cudaMalloc(&p, n * sizeof(T))); return p; }
template<class T> T* devCopy(const std::vector<T>& v) {
    T* p = devAlloc<T>(v.size());
    CHECK_CUDA(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

// ---- packed data -------------------------------------------------------------------------------
// Multi-channel VDIF: 32-byte headers, 8000-byte payloads, nband interleaved nbit-bit samples per
// sample time, little-endian bit order, offset-binary levels as mark5access decodes them.
static const int VDIF_HEADER = 32, VDIF_PAYLOAD = 8000;

// The host's own decode of one sample (written from mark5access's level tables, independently of
// the GPU code), for the references.
static inline double hostDecode(const unsigned char* payload, int nbit, int nband, int sampleInFrame, int band) {
    const long bit = ((long)sampleInFrame * nband + band) * nbit;
    if (nbit <= 2) {
        const int v = (payload[bit / 8] >> (bit % 8)) & ((1 << nbit) - 1);
        static const double two[4] = {-OPTIMAL_2BIT_HIGH, -1.0, 1.0, OPTIMAL_2BIT_HIGH};
        return nbit == 1 ? (v ? 1.0 : -1.0) : two[v];
    }
    unsigned int w;
    memcpy(&w, payload + bit / 32 * 4, 4);
    const int v = (w >> (bit % 32)) & ((1u << nbit) - 1);
    return nbit == 4 ? (v - 7.5) / 2.95 : (v - 128.0) / 3.3;
}

// ---- timing ------------------------------------------------------------------------------------
struct Timing { double median, min; };

// reps runs back to back on the stream, an event between each (the GPU never idles between them):
// the median and the minimum, ms
template<class F>
static Timing timeIt(F f, int warmup, int reps, cudaStream_t st) {
    for (int i = 0; i < warmup; i++) f();
    std::vector<cudaEvent_t> ev(reps + 1);
    for (auto& e : ev) CHECK_CUDA(cudaEventCreate(&e));
    CHECK_CUDA(cudaEventRecord(ev[0], st));
    for (int i = 0; i < reps; i++) {
        f();
        CHECK_CUDA(cudaEventRecord(ev[i + 1], st));
    }
    CHECK_CUDA(cudaEventSynchronize(ev[reps]));
    std::vector<double> t(reps);
    for (int i = 0; i < reps; i++) {
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, ev[i], ev[i + 1]));
        t[i] = ms;
    }
    for (auto& e : ev) cudaEventDestroy(e);
    std::sort(t.begin(), t.end());
    return {reps % 2 ? t[reps / 2] : 0.5 * (t[reps / 2 - 1] + t[reps / 2]), t[0]};
}

// each rep on its own, synchronised (xmacbench's: the GPU idles between reps): the median (the
// upper one for an even count) and the minimum, ms
template<class F>
static Timing timeItSync(F f, int warmup, int reps, cudaStream_t st) {
    for (int i = 0; i < warmup; i++) f();
    cudaEvent_t e0, e1;
    CHECK_CUDA(cudaEventCreate(&e0));
    CHECK_CUDA(cudaEventCreate(&e1));
    std::vector<double> t;
    for (int i = 0; i < reps; i++) {
        CHECK_CUDA(cudaEventRecord(e0, st));
        f();
        CHECK_CUDA(cudaEventRecord(e1, st));
        CHECK_CUDA(cudaEventSynchronize(e1));
        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
        t.push_back(ms);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    std::sort(t.begin(), t.end());
    return {t[t.size() / 2], t[0]};
}
