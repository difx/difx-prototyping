// fusebench.cu - is fusing the FFT and the fractional-sample correction with cuFFTDx worth it?
//
// Times, on synthetic data in GPU memory (no host transfers), one subint's
//   unfused:  [pre-FFT stage kernel] + cuFFT + rotation kernel, per antenna (mpifxcorr's shape)
//   fused:    one cuFFTDx kernel per antenna doing the same
// where the input is either the FFT input as mpifxcorr's decode + fringe-rotation kernel leaves it
// (FP32 or FP16), or the packed data itself (--input packed: the pre-FFT stage is the decode and
// fringe rotation, gpudifx's kernel in the unfused pipeline, inside the fused kernel otherwise)
// and checks that both give the same spectra (and autocorrelations, if asked for), and match a
// double-precision DFT of the exact, unrounded input on a sample of spectra. See README.md.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <string>
#include <vector>
#include <cufftdx/cufftdx_version.hpp>
#include "pipelines.cuh"

int fusedLastRegisters = 0;

std::map<int, std::vector<FusedVariant>>& fusedRegistry() {
    static std::map<int, std::vector<FusedVariant>> r;
    return r;
}

// ---- cross-pol pass for the fused pipeline ---------------------------------------------------
__global__ void crosspol_kernel(Geometry g, FracCommon<true> post, const cuFloatComplex* out) {
    const int win = blockIdx.x;
    const int chan = blockIdx.y * blockDim.x + threadIdx.x;
    const int ant = blockIdx.z;
    if (chan >= g.nchan || !post.windowValid(g, ant, win))
        return;
    post.crossPol(g, ant, win, chan, out + ant * g.perAnt());
}

void launchCrossPol(const Geometry& g, const FracParams& p, const cuFloatComplex* out, int maxThreads,
                    cudaStream_t st) {
    const int tpb = std::min(g.nchan, maxThreads);
    const dim3 grid(g.nwin, (g.nchan + tpb - 1) / tpb, g.nant);
    crosspol_kernel<<<grid, tpb, 0, st>>>(g, FracCommon<true>{p}, out);
    CHECK_CUDA(cudaGetLastError());
}

// ---- synthetic data --------------------------------------------------------------------------
__device__ __host__ inline unsigned long long splitmix(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// unit-variance complex Gaussian noise for sample i (Box-Muller on a counter hash), so that any
// sample can be regenerated exactly, before rounding to the storage precision
__device__ inline float2 noise(unsigned long long seed, size_t i) {
    const unsigned long long h = splitmix(seed ^ splitmix(i));
    const float u1 = ((h >> 40) + 1) * (1.0f / 16777217.0f);
    const float u2 = ((h >> 16) & 0xFFFFFF) * (1.0f / 16777216.0f);
    const float r = sqrtf(-logf(u1));
    float s, c;
    sincospif(2.0f * u2, &s, &c);
    return make_float2(r * c, r * s);
}

__device__ inline void store(float2* x, size_t i, float2 v) { x[i] = v; }
__device__ inline void store(__half2* x, size_t i, float2 v) { x[i] = __float22half2_rn(v); }

template<class T>
__global__ void fill_noise(T* x, size_t n, unsigned long long seed) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        store(x, i, noise(seed, i));
}

// The first kernel of the two-kernel pipeline (--input packed): decode + fringe rotation of all
// antennas into a buffer per FFT window, [ant][win][band][fftchannels], in FP32 or FP16.  A block
// per (antenna, window), a warp per band at a time (decodeToThreads: up to 8 warps); a warp's
// lanes take 32 consecutive samples at a time (coalesced stores), U at a time unrolled, so that
// each lane's samples are lane, lane + 32, ... and the stage is prepared once per lane and band,
// as in the fused kernel.  fftchannels is a multiple of 32.  The window weights are left to the
// FFT kernel, which applies them as it loads.
template<int U, class Pre, class T>
__global__ void decode_to(Geometry g, Pre pre, T* out) {
    const unsigned aw = blockIdx.x;                      // ant * nwin + win
    const int ant = aw / g.nwin, win = aw % g.nwin;
    const int lane = threadIdx.x & 31;
    for (int band = threadIdx.x >> 5; band < g.nband; band += blockDim.x >> 5) {
        typename Pre::State st = pre.prepare(g, ant, win, band, lane, 32);
        T* o = out + ((size_t)aw * g.nband + band) * g.fftchannels;
        int s0 = 0;
        for (; s0 + 32 * U <= g.fftchannels; s0 += 32 * U) {
#pragma unroll
            for (int k = 0; k < U; k++) {
                const int s = s0 + lane + 32 * k;
                store(o, s, pre.at(st, g, ant, win, band, s));
            }
        }
        for (int s = s0 + lane; s < g.fftchannels; s += 32)   // the rest (fftchannels < 32 U)
            store(o, s, pre.at(st, g, ant, win, band, s));
    }
}
static unsigned decodeToThreads(const Geometry& g) { return 32 * std::min(g.nband, 8); }

// The exact FFT input of a few chosen spectra (the noise before rounding to the storage
// precision, times the window), for the double-precision reference.
template<class Pre>
__global__ void gather_exact(Geometry g, Pre pre, unsigned long long seed, const int* which, double2* out) {
    const int k = blockIdx.x;
    const int ant = which[3 * k], win = which[3 * k + 1], band = which[3 * k + 2];
    for (int s = threadIdx.x; s < g.fftchannels; s += blockDim.x) {
        const float2 v = noise(seed, pre.sourceIndex(g, ant, win, band, s));
        const double w = pre.weight(s);
        out[(size_t)k * g.fftchannels + s] = make_double2(v.x * w, v.y * w);
    }
}

struct Rng {
    unsigned long long s;
    double uniform() { s = splitmix(s); return (s >> 11) * (1.0 / 9007199254740992.0); }
    double range(double a, double b) { return a + (b - a) * uniform(); }
};

template<class T> T* devAlloc(size_t n) { T* p; CHECK_CUDA(cudaMalloc(&p, n * sizeof(T))); return p; }
template<class T> T* devCopy(const std::vector<T>& v) {
    T* p = devAlloc<T>(v.size());
    CHECK_CUDA(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
static bool* devCopyBool(const std::vector<char>& v) {
    bool* p = devAlloc<bool>(v.size());
    std::vector<unsigned char> b(v.begin(), v.end());
    CHECK_CUDA(cudaMemcpy(p, b.data(), b.size(), cudaMemcpyHostToDevice));
    return p;
}

// Tapered window: tapers of round(taper * n) samples at each end, flat between. Shapes, x in
// (0, 1) across the taper: tukey 0.5 (1 - cos(pi x)); sine sin(pi x / 2); vorbis
// sin(pi/2 * 0.5 (1 - cos(pi x))). Overlapping neighbours by the taper length, Tukey tapers add to
// one in amplitude, sine and Vorbis tapers in power (no loss of sensitivity); Vorbis tapers are
// also smooth at both ends, so their sidelobes fall off nearly as fast as Tukey's.
enum class Taper { TUKEY, SINE, VORBIS };
static std::vector<float> taperedWindow(int n, double taper, Taper shape) {
    std::vector<float> w(n, 1.0f);
    const int L = (int)std::lround(taper * n);
    for (int i = 0; i < L; i++) {
        const double x = (i + 0.5) / L, rc = 0.5 * (1.0 - cos(M_PI * x));
        const float v = (float)(shape == Taper::TUKEY ? rc : shape == Taper::SINE ? sin(0.5 * M_PI * x)
                                                                                  : sin(0.5 * M_PI * rc));
        w[i] = v;
        w[n - 1 - i] = v;
    }
    return w;
}

// ---- packed data -----------------------------------------------------------------------------
// Multi-channel VDIF: 32-byte headers, 8000-byte payloads, nband interleaved nbit-bit samples per
// sample time, little-endian bit order, offset-binary levels as mark5access decodes them.
static const int VDIF_HEADER = 32, VDIF_PAYLOAD = 8000;

// The host's own decode of one sample (written from mark5access's level tables, independently of
// the GPU code), for the reference.
static double hostDecode(const unsigned char* payload, int nbit, int nband, int sampleInFrame, int band) {
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

// ---- comparison ------------------------------------------------------------------------------
// over the nchan used channels of every spectrum of a valid window ([ant][win]; the pipelines need
// not agree on the others): max |a - b|, sum |a - b|^2, sum |b|^2, count
__global__ void compare_kernel(Geometry g, const cuFloatComplex* a, const cuFloatComplex* b, const bool* valid,
                               double* sums, unsigned int* maxbits) {
    const size_t n = g.spectra() * g.nchan;
    double d2 = 0, r2 = 0, cnt = 0;
    float dmax = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const size_t spec = i / g.nchan;
        if (!valid[spec / g.nband]) continue;              // spectra are [ant][win][band]
        cnt += 1;
        const size_t idx = spec * g.fftchannels + i % g.nchan;
        const float dx = a[idx].x - b[idx].x, dy = a[idx].y - b[idx].y;
        const float d = dx * dx + dy * dy;
        d2 += d;
        r2 += (double)b[idx].x * b[idx].x + (double)b[idx].y * b[idx].y;
        dmax = fmaxf(dmax, d);
    }
    atomicAdd(&sums[0], d2);
    atomicAdd(&sums[1], r2);
    atomicAdd(&sums[2], cnt);
    atomicMax(maxbits, __float_as_uint(sqrtf(dmax)));   // non-negative floats order as their bits
}

struct Diff { double rms, max; };   // relative to the rms of the reference

static Diff compareDevice(const Geometry& g, const cuFloatComplex* a, const cuFloatComplex* ref, const bool* valid) {
    double* sums = devAlloc<double>(3);
    unsigned int* mx = devAlloc<unsigned int>(1);
    CHECK_CUDA(cudaMemset(sums, 0, 3 * sizeof(double)));
    CHECK_CUDA(cudaMemset(mx, 0, sizeof(unsigned int)));
    compare_kernel<<<1024, 256>>>(g, a, ref, valid, sums, mx);
    double h[3];
    unsigned int m;
    CHECK_CUDA(cudaMemcpy(h, sums, sizeof h, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&m, mx, sizeof m, cudaMemcpyDeviceToHost));
    cudaFree(sums);
    cudaFree(mx);
    float fm;
    memcpy(&fm, &m, sizeof fm);
    const double rref = sqrt(h[1] / h[2]);
    return {sqrt(h[0] / h[2]) / rref, fm / rref};
}

static Diff compareHost(const std::vector<std::complex<double>>& a, const std::vector<std::complex<double>>& ref) {
    double d2 = 0, r2 = 0, dmax = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        d2 += std::norm(a[i] - ref[i]);
        r2 += std::norm(ref[i]);
        dmax = std::max(dmax, std::abs(a[i] - ref[i]));
    }
    const double rref = sqrt(r2 / ref.size());
    return {sqrt(d2 / ref.size()) / rref, dmax / rref};
}

// ---- timing ----------------------------------------------------------------------------------
struct Timing { double median, min; };

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

// ---- chunks: one antenna, a range of windows -------------------------------------------------
// The parameters and the packed input of antenna ant, windows [w0, w0 + nwin), as a geometry of
// one antenna and nwin windows (offset pointers), so that a pipeline can run a subint a chunk at a
// time, each chunk's intermediate buffer small enough to stay in L2 (--chunk).
struct ChunkView { Geometry g; FracParams p; PreVariant pre; };

static ChunkView chunkView(const Geometry& g, const FracParams& p, const PreVariant& pre, int ant, int w0, int nwin) {
    ChunkView v{g, p, pre};
    v.g.nant = 1;
    v.g.nwin = nwin;
    const size_t aw = (size_t)ant * g.nwin + w0, ab = (size_t)ant * g.nband;
    v.p.fracSampleError += aw;
    v.p.validSamples += aw;
    v.p.clockOffsets += ab;
    v.p.clockOffsetsDelta += ab;
    v.p.fracFreqOffset += ab;
    v.p.bandFlags += ab;
    if (v.p.autocorr) v.p.autocorr += (size_t)ant * 2 * g.nband * g.nchan;
    v.p.fixedPhase += aw * g.nband;
    v.p.numBlocks = nwin;
    if (pre.packed) {
        PackedInput& d = v.pre.pk;
        d.packed += (size_t)ant * d.bytesPerAnt;
        d.validFrames += (size_t)ant * d.nframes;
        d.sampleIndex += aw;
        d.decodeFlags += ab;
        d.bigA += aw * g.nband;
        d.bigBred += aw * g.nband;
        d.fringePhase += aw * g.nband;
        d.winDesc += aw;
    }
    return v;
}

// ---- options ---------------------------------------------------------------------------------
static void usage() {
    printf(
"Usage: fusebench [options]\n"
"Shape of one subint:\n"
"  --nant N        antennas (one GPUMode each), default 10\n"
"  --nband N       recorded bands per antenna, default 8 (dual pol: bands 2k, 2k+1 pair up)\n"
"  --nchan N       output channels per band (recordedbandchannels), default 128\n"
"  --nwin N        subint length in FFT lengths, default 10000: the number of FFT windows per\n"
"                  antenna without overlap (numBufferedFFTs); with --overlap there are more\n"
"  --complex       complex-sampled data: FFT length = nchan (default real: FFT length = 2 nchan)\n"
"  --bw MHz        bandwidth per band, sets the sample time and the real-time figure, default 64\n"
"FFT input (the output of decode + fringe rotation, or the packed data):\n"
"  --input P       fp32 (default) | fp16: the FFT input, stored by an earlier stage; the FFT itself\n"
"                  is FP32 either way | packed: start from the packed data (multi-channel VDIF),\n"
"                  so that the pre-FFT stage is the decode, sideband handling (odd bands are lower\n"
"                  sideband) and fringe rotation; with --rotator double the unfused pipeline runs\n"
"                  gpudifx's decode + fringe kernel (FP64 phases), else fixed-point phases\n"
"  --nbit N        packed: bits per sample, 1, 2 (default), 4 or 8\n"
"  --decode D      packed: lean (default; what depends only on the window and band once per\n"
"                  thread, the number of bits fixed at compile time) | direct (gpudifx's\n"
"                  per-sample arithmetic: frame, validity and bits per sample worked out for\n"
"                  every sample).  Both the fused kernel and the 2-kernel decode use it\n"
"  --bad-frames F  packed: fraction of the VDIF frames flagged invalid (decoded as zeros), default 0\n"
"  --layout L      window (default: one buffer per FFT window, mpifxcorr's layout) | series\n"
"                  (one continuous series per antenna and band)\n"
"  --taper F       window on the FFT input with tapers of F of the FFT length at each end,\n"
"                  default 0 (no window)\n"
"  --window W      taper shape: tukey (default, raised cosine) | sine | vorbis (sin(pi/2 x\n"
"                  raised cosine): with --overlap equal to --taper, sine and vorbis tapers lose\n"
"                  no sensitivity\n"
"  --overlap F     consecutive windows share F of the FFT length (the hop is 1 - F FFT lengths),\n"
"                  default 0; needs --layout series or packed input. With F equal to the\n"
"                  taper, the tapers of neighbouring windows lie on top of each other\n"
"  --wola          short for --layout series --taper 0.125 --overlap 0.125 --window vorbis:\n"
"                  12.5%% / 75%% / 12.5%% taper / flat / taper, each window overlapping each\n"
"                  neighbour over its taper (a --window given as well takes precedence)\n"
"Rotation and autocorrelations:\n"
"  --rotator R     unfused pipeline: fixed (default, as the fused kernel) | double (mpifxcorr's\n"
"                  kernels verbatim, FP64 per element: the fractional-sample rotation and, with\n"
"                  packed input, the decode + fringe rotation)\n"
"  --autocorr M    none (default; autocorrelations in the XMAC) | band (per-band, atomics in the\n"
"                  rotation, as mpifxcorr with DIFX_GPU_XMAC_AUTOCORR unset) | full (plus cross-pol)\n"
"  --singlepol     no band pairs (no cross-pol products)\n"
"  --crosspol-pass fused: form the cross-pol products in a separate pass, not in the kernel\n"
"Fused kernel and run:\n"
"  --fpb N|sweep   FFTs per block: cuFFTDx's suggestion (default), N, or time all\n"
"  --ept E         FFT elements per thread: cuFFTDx's suggestion (default) | half (of it, where\n"
"                  supported) | N | sweep (both)\n"
"  --regcap R      registers capped (__launch_bounds__) to fit R threads per SM: 0 (default, no\n"
"                  cap) | 1024 | 1536 (1, 2 and 4 FFTs per block only) | sweep (all three)\n"
"  --batched       one launch for all antennas (one cuFFT plan; one fused kernel)\n"
"  --chunk W       packed input: also run the end-to-end kernel and the 2-kernel pipelines W\n"
"                  windows at a time, each antenna on its own stream (the antennas run together),\n"
"                  each antenna's part of the 2-kernel buffer reused for every chunk, so that\n"
"                  nant x W windows are in flight (sized to stay in L2; the size is printed)\n"
"  --invalid F     fraction of windows flagged invalid, default 0\n"
"  --reps N        timed repetitions, default 20 (median and minimum reported)\n"
"  --warmup N      untimed repetitions first, default 3\n"
"  --check N       spectra checked against a double-precision DFT, default 16\n"
"  --seed N        random seed, default 1\n"
"  --device N      CUDA device, default 0\n");
}

int main(int argc, char** argv) {
    Geometry g{10, 8, 10000, 0, 128};
    bool complexData = false, batched = false, singlePol = false, sweep = false, crossPolPass = false;
    bool half = false, series = false, packed = false, lean = true;
    int nbit = 2;
    Taper shape = Taper::TUKEY;
    bool wola = false, shapeGiven = false;
    double bw = 64.0, invalidFrac = 0.0, badFrames = 0.0, taper = 0.0, overlap = 0.0;
    AutoMode amode = AutoMode::NONE;
    Rotator rotator = Rotator::FIXED;
    int fpb = 0, ept = 0, regcap = 0, chunk = 0, reps = 20, warmup = 3, ncheck = 16, device = 0;
    bool sweepEpt = false, sweepRegcap = false;
    unsigned long long seed = 1;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(2); }
            return argv[++i];
        };
        auto choose = [&](std::initializer_list<const char*> names) -> int {
            const std::string m = next();
            int k = 0;
            for (const char* n : names) { if (m == n) return k; k++; }
            fprintf(stderr, "%s: unknown value %s\n", a.c_str(), m.c_str());
            exit(2);
        };
        if (a == "--nant") g.nant = std::stoi(next());
        else if (a == "--nband") g.nband = std::stoi(next());
        else if (a == "--nchan") g.nchan = std::stoi(next());
        else if (a == "--nwin") g.nwin = std::stoi(next());
        else if (a == "--complex") complexData = true;
        else if (a == "--bw") bw = std::stod(next());
        else if (a == "--input") { const int k = choose({"fp32", "fp16", "packed"}); half = k == 1; packed = k == 2; }
        else if (a == "--nbit") nbit = std::stoi(next());
        else if (a == "--decode") lean = choose({"lean", "direct"}) == 0;
        else if (a == "--bad-frames") badFrames = std::stod(next());
        else if (a == "--layout") series = choose({"window", "series"}) == 1;
        else if (a == "--taper") taper = std::stod(next());
        else if (a == "--window") { shape = (Taper)choose({"tukey", "sine", "vorbis"}); shapeGiven = true; }
        else if (a == "--overlap") overlap = std::stod(next());
        else if (a == "--wola") { series = true; taper = 0.125; overlap = 0.125; wola = true; }
        else if (a == "--rotator") rotator = choose({"double", "fixed"}) == 0 ? Rotator::DOUBLE : Rotator::FIXED;
        else if (a == "--autocorr") amode = (AutoMode)choose({"none", "band", "full"});
        else if (a == "--singlepol") singlePol = true;
        else if (a == "--crosspol-pass") crossPolPass = true;
        else if (a == "--fpb") { std::string m = next(); if (m == "sweep") sweep = true; else fpb = std::stoi(m); }
        else if (a == "--regcap") {
            std::string m = next();
            if (m == "sweep") sweepRegcap = true; else regcap = std::stoi(m);
        }
        else if (a == "--ept") {
            std::string m = next();
            if (m == "sweep") sweepEpt = true; else if (m == "half") ept = -1; else ept = std::stoi(m);
        }
        else if (a == "--batched") batched = true;
        else if (a == "--chunk") chunk = std::stoi(next());
        else if (a == "--invalid") invalidFrac = std::stod(next());
        else if (a == "--reps") reps = std::stoi(next());
        else if (a == "--warmup") warmup = std::stoi(next());
        else if (a == "--check") ncheck = std::stoi(next());
        else if (a == "--seed") seed = std::stoull(next());
        else if (a == "--device") device = std::stoi(next());
        else { usage(); return a == "--help" || a == "-h" ? 0 : 2; }
    }
    if (wola && !shapeGiven)
        shape = Taper::VORBIS;
    g.fftchannels = complexData ? g.nchan : 2 * g.nchan;
    if (g.nant < 1 || g.nband < 1 || g.nwin < 1 || g.nchan < 1 || reps < 1 || taper < 0 || taper > 0.5 ||
        overlap < 0 || overlap >= 1) { usage(); return 2; }
    if (overlap > 0 && !series && !packed) { fprintf(stderr, "--overlap needs --layout series\n"); return 2; }
    if (packed && complexData) { fprintf(stderr, "--input packed: real data only so far\n"); return 2; }
    if (packed && g.fftchannels % 32) { fprintf(stderr, "--input packed: FFT length a multiple of 32\n"); return 2; }
    if (packed && nbit != 1 && nbit != 2 && nbit != 4 && nbit != 8) { fprintf(stderr, "--nbit: 1, 2, 4 or 8\n"); return 2; }
    if (packed && (VDIF_PAYLOAD * 8) % (g.nband * nbit) != 0) {
        fprintf(stderr, "--input packed: %d bands of %d bits do not divide a %d-byte payload\n", g.nband, nbit, VDIF_PAYLOAD);
        return 2;
    }
    if (chunk < 0 || (chunk > 0 && !packed)) { fprintf(stderr, "--chunk W: W > 0, with --input packed\n"); return 2; }
    if (packed && lean && nbit != 2 && amode != AutoMode::NONE) {
        fprintf(stderr, "--decode lean: the autocorrelations are compiled for 2 bits only (or use --decode direct)\n");
        return 2;
    }
    if (packed && rotator == Rotator::DOUBLE && taper > 0) {
        fprintf(stderr, "--rotator double with packed input runs gpudifx's decode, which has no window\n");
        return 2;
    }
    // the subint is nwin FFT lengths of data; overlapping windows step by hop samples through it
    const int nwinNoOverlap = g.nwin;
    const int hop = g.fftchannels - (int)std::lround(overlap * g.fftchannels);
    const size_t subintSamples = (size_t)nwinNoOverlap * g.fftchannels;
    g.nwin = (int)((subintSamples - g.fftchannels) / hop + 1);
    const int nfreq = singlePol ? g.nband : (g.nband + 1) / 2;

    CHECK_CUDA(cudaSetDevice(device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    int cufftVersion = 0, rtVersion = 0;
    cufftGetVersion(&cufftVersion);
    cudaRuntimeGetVersion(&rtVersion);

    auto reg = fusedRegistry().find(g.fftchannels);
    if (reg == fusedRegistry().end()) {
        fprintf(stderr, "FFT length %d is not compiled in (sizes:", g.fftchannels);
        for (auto& e : fusedRegistry()) fprintf(stderr, " %d", e.first);
        fprintf(stderr, "); add it to SIZES in the Makefile\n");
        return 2;
    }
    std::vector<FusedVariant> variants;
    for (const auto& v : reg->second)
        if (v.smem <= prop.sharedMemPerBlockOptin && v.threads <= (unsigned)prop.maxThreadsPerBlock)
            variants.push_back(v);
    std::sort(variants.begin(), variants.end(), [](const FusedVariant& a, const FusedVariant& b) {
        return a.regcap != b.regcap ? a.regcap < b.regcap : a.ept != b.ept ? a.ept > b.ept : a.fpb < b.fpb; });
    // default: cuFFTDx's suggestion; with --ept alone, its suggested FFTs per block at that EPT
    unsigned eptDefault = 0, fpbDefault = 0;
    for (const auto& v : variants)
        if (v.suggested) { eptDefault = v.ept; fpbDefault = v.fpb; }
    const unsigned eptWanted = ept > 0 ? (unsigned)ept : ept < 0 ? eptDefault / 2 : eptDefault;
    const unsigned fpbWanted = fpb > 0 ? (unsigned)fpb : fpbDefault;
    const FusedVariant* chosen = nullptr;
    for (const auto& v : variants)
        if (v.ept == eptWanted && v.fpb == fpbWanted && v.regcap == (unsigned)regcap)
            chosen = &v;
    if (!chosen) {
        fprintf(stderr, "no fused variant with %u FFTs per block, %u elements per thread and register cap %d fits "
                "this GPU for FFT length %d (available, FFTs per block x elements per thread / cap:", fpbWanted,
                eptWanted, regcap, g.fftchannels);
        for (const auto& v : variants) fprintf(stderr, " %ux%u/%u", v.fpb, v.ept, v.regcap);
        fprintf(stderr, ")\n");
        return 2;
    }

    const double sampleTimeUs = complexData ? 1.0 / bw : 1.0 / (2.0 * bw);
    const double subintMs = subintSamples * sampleTimeUs / 1000.0;
    const size_t nelem = g.perAnt() * g.nant;                      // FFT input/output, per window
    const size_t seriesLen = (size_t)(g.nwin - 1) * hop + g.fftchannels;
    const size_t ninput = series ? (size_t)g.nant * g.nband * seriesLen : nelem;
    // packed: each antenna's data starts a few samples early (an integer delay), then windows of
    // fftchannels samples every hop samples
    const int framesamples = VDIF_PAYLOAD * 8 / (g.nband * nbit);
    const int maxDelay = 64;
    const int nframes = (int)(((size_t)(g.nwin - 1) * hop + g.fftchannels + maxDelay + framesamples - 1) / framesamples);
    const size_t bytesPerAnt = (size_t)nframes * (VDIF_HEADER + VDIF_PAYLOAD);
    if (g.spectra() >= 0x80000000ull) { fprintf(stderr, "more than 2^31 spectra in a subint\n"); return 2; }
    if (packed && lean && !leanDecodeFits(framesamples, g.fftchannels, bytesPerAnt)) {
        fprintf(stderr, "--decode lean: windows of up to 32 frames and less than 2 GB per antenna (use --decode direct)\n");
        return 2;
    }
    const double gib = 1.0 / 1073741824.0;
    const char* amodeName[] = {"none", "band", "full"};
    printf("GPU %s (sm_%d%d, %d SMs), CUDA runtime %d, cuFFT %d, cuFFTDx %d\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, rtVersion, cufftVersion, CUFFTDX_VERSION);
    printf("%d antennas x %d bands, %s data, FFT length %d, %d channels kept; %.0f MHz bands -> subint %.3f ms\n",
           g.nant, g.nband, complexData ? "complex" : "real", g.fftchannels, g.nchan, bw, subintMs);
    if (packed)
        printf("Input: packed VDIF, %d-bit, %d frames of %d samples per antenna, decode %s (fused); FFT input: ",
               nbit, nframes, framesamples, lean ? "lean" : "direct");
    else
        printf("FFT input: %s, %s layout, ", half ? "FP16" : "FP32", series ? "series" : "per-window");
    const char* shapeName[] = {"Tukey", "sine-taper", "Vorbis-taper"};
    if (taper > 0) printf("%s window (%.1f%% tapers), ", shapeName[(int)shape], 100 * taper);
    else printf("no window, ");
    printf("%d windows per antenna (hop %d: consecutive windows share %.1f%%)\n", g.nwin, hop,
           100.0 * (g.fftchannels - hop) / g.fftchannels);
    printf("unfused rotator: %s; fused rotator: fixed; autocorrelations: %s%s; %s\n",
           rotator == Rotator::DOUBLE ? "double (mpifxcorr)" : "fixed", amodeName[(int)amode],
           singlePol ? " (single pol)" : "", batched ? "one launch for all antennas" : "one launch per antenna");

    // ---- inputs ----
    Rng rng{seed * 7919 + 17};
    std::vector<float> fse((size_t)g.nant * g.nwin);
    std::vector<char> valid((size_t)g.nant * g.nwin);
    std::vector<double> clk((size_t)g.nant * g.nband), clkd(clk.size()), fracoff(clk.size(), 0.0);
    std::vector<unsigned char> flags(clk.size(), 0);
    for (size_t i = 0; i < fse.size(); i++) {
        fse[i] = (float)rng.range(-0.5, 0.5) * sampleTimeUs;     // within half a sample, as after
        valid[i] = rng.uniform() >= invalidFrac;                  // the integer-sample shift
    }
    for (size_t i = 0; i < clk.size(); i++) {
        clk[i] = rng.range(-0.05, 0.05);       // per-band clock offsets, microseconds
        clkd[i] = rng.range(-0.001, 0.001);
    }
    std::vector<unsigned int> indices((size_t)nfreq * MAX_INDICIES, 0);
    std::vector<int> counts(nfreq);
    for (int f = 0; f < nfreq; f++) {
        int c = 0;
        for (int b = 0; b < g.nband; b++)
            if ((singlePol ? b : b / 2) == f) indices[f * MAX_INDICIES + c++] = b;
        counts[f] = c;
    }

    PreVariant pre{};
    pre.half = half;
    pre.series = series;
    pre.hop = hop;
    pre.seriesLen = seriesLen;
    pre.window = taper > 0 ? devCopy(taperedWindow(g.fftchannels, taper, shape)) : nullptr;
    pre.packed = packed;
    pre.lean = lean;
    std::vector<unsigned char> hPacked;
    std::vector<int> sampleIndex;
    std::vector<double> bigA, bigB;
    std::vector<unsigned char> decodeFlags;
    std::vector<char> validFrames;
    const long datasamples = 1000001;          // odd: the ALTSIGN sign starts at -1
    if (packed) {
        hPacked.resize(g.nant * bytesPerAnt);
        for (size_t i = 0; i < hPacked.size(); i += 8) {
            const unsigned long long r = splitmix(seed * 0x1000003 + i);
            memcpy(&hPacked[i], &r, std::min<size_t>(8, hPacked.size() - i));
        }
        sampleIndex.resize((size_t)g.nant * g.nwin);
        for (int a = 0; a < g.nant; a++) {
            const int delay = (int)(rng.uniform() * maxDelay);
            for (int w = 0; w < g.nwin; w++) sampleIndex[(size_t)a * g.nwin + w] = w * hop + delay;
        }
        // fringe phase slope (turns per sample; |A| <= 3e-4 is e.g. 3 us/s of delay rate at 10 GHz
        // and 64 MHz) and offset, per window and band, as gpu_precompute_fringe_rotator makes them
        bigA.resize(g.spectra());
        bigB.resize(g.spectra());
        for (size_t i = 0; i < bigA.size(); i++) {
            bigA[i] = rng.range(-3e-4, 3e-4);
            bigB[i] = rng.range(-1.0, 1.0);
        }
        decodeFlags.resize((size_t)g.nant * g.nband);
        for (size_t i = 0; i < decodeFlags.size(); i++) decodeFlags[i] = (i % g.nband) % 2 ? GPUDECODE_BAND_ALTSIGN : 0;
        PackedInput& d = pre.pk;
        d.packed = devCopy(hPacked);
        d.bytesPerAnt = bytesPerAnt;
        d.ms = mark5_stream{framesamples, VDIF_HEADER + VDIF_PAYLOAD, VDIF_HEADER, nbit, g.nband, 1};
        d.nframes = nframes;
        validFrames.assign((size_t)g.nant * nframes, 1);
        if (badFrames > 0) {
            for (auto& v : validFrames) v = rng.uniform() >= badFrames;
            printf("%zu of %zu frames flagged invalid\n", (size_t)std::count(validFrames.begin(), validFrames.end(), 0),
                   validFrames.size());
        }
        d.validFrames = devCopyBool(validFrames);
        d.sampleIndex = devCopy(sampleIndex);
        d.decodeFlags = devCopy(decodeFlags);
        d.datasamples = datasamples;
        d.bigA = devCopy(bigA);
        d.bigBred = devCopy(bigB);
        d.fringePhase = devAlloc<ulonglong2>(g.spectra());
        d.winDesc = devAlloc<int4>((size_t)g.nant * g.nwin);
        d.fsMagic = frameMagic(framesamples);
    } else if (half) {
        __half2* in = devAlloc<__half2>(ninput);
        fill_noise<<<4096, 256>>>(in, ninput, seed);
        pre.in = in;
    } else {
        float2* in = devAlloc<float2>(ninput);
        fill_noise<<<4096, 256>>>(in, ninput, seed);
        pre.in = in;
    }
    CHECK_CUDA(cudaGetLastError());
    bool direct = false;
    withPre(pre, [&](auto stage) { direct = stage.direct(); });

    // Output buffers. When the unfused pipeline needs a staging buffer for its FFT input, the
    // fused kernel writes into that buffer (it is free once the unfused FFT has run), which keeps
    // the 128-channel overlap case within 8 GB.
    cudaStream_t st;
    CHECK_CUDA(cudaStreamCreate(&st));
    Unfused unf;
    unf.init(g, batched, !direct);
    cuFloatComplex* outUnfused = devAlloc<cuFloatComplex>(nelem);
    cuFloatComplex* outFused = direct ? devAlloc<cuFloatComplex>(nelem) : unf.staging;
    const size_t inputBytes = packed ? hPacked.size() : ninput * (half ? sizeof(__half2) : sizeof(float2));
    printf("GPU memory: input %.2f GiB, outputs %.2f GiB each%s\n", inputBytes * gib,
           nelem * sizeof(cuFloatComplex) * gib, direct ? "" : " (fused output shares the unfused staging buffer)");

    const size_t nauto = (size_t)g.nant * 2 * g.nband * g.nchan;
    const bool doAuto = amode != AutoMode::NONE;
    cuFloatComplex* autoUnfused = doAuto ? devAlloc<cuFloatComplex>(nauto) : nullptr;
    cuFloatComplex* autoFused = doAuto ? devAlloc<cuFloatComplex>(nauto) : nullptr;

    FracParams p{};
    p.fracSampleError = devCopy(fse);
    p.validSamples = devCopyBool(valid);
    p.clockOffsets = devCopy(clk);
    p.clockOffsetsDelta = devCopy(clkd);
    p.fracFreqOffset = devCopy(fracoff);
    p.bandFlags = devCopy(flags);
    p.indices = devCopy(indices);
    p.counts = devCopy(counts);
    p.recordedBandwidth = bw;
    p.numRecordedFreqs = nfreq;
    p.numBlocks = g.nwin;
    p.crossPol = (amode == AutoMode::FULL);
    p.fixedPhase = devAlloc<ulonglong2>(g.spectra());
    // the fused kernel can form the cross-pol products itself when each frequency's pair is
    // (2f, 2f+1); otherwise (and with --crosspol-pass) they are a separate pass
    bool adjacentPairs = !singlePol && g.nband % 2 == 0;
    for (int f = 0; f < nfreq && adjacentPairs; f++)
        adjacentPairs = counts[f] == 2 && indices[f * MAX_INDICIES] == (unsigned)(2 * f) &&
                        indices[f * MAX_INDICIES + 1] == (unsigned)(2 * f + 1);
    p.crossPolInBlock = adjacentPairs && !crossPolPass;

    unf.decodeRotator = rotator;
    unf.validSamples = p.validSamples;
    FracParams pU = p, pF = p;
    pU.autocorr = autoUnfused;
    pF.autocorr = autoFused;
    auto runUnfused = [&]() {
        if (doAuto) CHECK_CUDA(cudaMemsetAsync(autoUnfused, 0, nauto * sizeof(cuFloatComplex), st));
        unf.run(pre, pU, amode, rotator, outUnfused, st);
    };
    auto runFused = [&](const FusedVariant& v) {
        if (doAuto) CHECK_CUDA(cudaMemsetAsync(autoFused, 0, nauto * sizeof(cuFloatComplex), st));
        launchFixedPhase(g, pF, st);
        if (packed) launchPackedSetup(g, pre.pk, st);
        bool crossDone = false;
        if (batched) crossDone = v.launch(g, pre, pF, doAuto, outFused, 0, g.nant, st);
        else for (int a = 0; a < g.nant; a++) crossDone = v.launch(g, pre, pF, doAuto, outFused, a, 1, st);
        if (amode == AutoMode::FULL && !crossDone) launchCrossPol(g, pF, outFused, prop.maxThreadsPerBlock, st);
        return crossDone;
    };

    // ---- correctness: one run of each, then compare ----
    CHECK_CUDA(cudaMemset(outUnfused, 0, nelem * sizeof(cuFloatComplex)));
    runUnfused();
    CHECK_CUDA(cudaStreamSynchronize(st));
    CHECK_CUDA(cudaMemset(outFused, 0, nelem * sizeof(cuFloatComplex)));
    runFused(*chosen);
    CHECK_CUDA(cudaStreamSynchronize(st));
    CHECK_CUDA(cudaGetLastError());

    bool pass = true;
    const Diff dSpec = compareDevice(g, outFused, outUnfused, p.validSamples);
    // fused vs unfused spectra: single-precision FFTs of the same (rounded) input, rotated with
    // the same or (--rotator double) a different phase arithmetic: a few 1e-7 rms (relative)
    const double specTol = 1e-5;
    const bool specOk = dSpec.rms < specTol && dSpec.max < 30 * specTol;
    pass &= specOk;
    printf("\ncheck: fused vs unfused spectra: rms %.2e, max %.2e (relative to rms; limits %.0e, %.0e)  %s\n",
           dSpec.rms, dSpec.max, specTol, 30 * specTol, specOk ? "OK" : "FAIL");

    // both against a double-precision DFT and rotation of the exact input (before rounding to the
    // storage precision), on a sample of valid spectra; with FP16 input this includes the rounding
    if (ncheck > 0) {
        std::vector<int> which;
        for (int k = 0; k < ncheck * 20 && (int)which.size() < 3 * ncheck; k++) {
            const int ant = (int)(rng.uniform() * g.nant), win = (int)(rng.uniform() * g.nwin),
                      band = (int)(rng.uniform() * g.nband);
            if (valid[(size_t)ant * g.nwin + win]) { which.push_back(ant); which.push_back(win); which.push_back(band); }
        }
        const int nc = which.size() / 3;
        int* dWhich = devCopy(which);
        double2* dIn = devAlloc<double2>((size_t)nc * g.fftchannels);
        std::vector<double2> hIn((size_t)nc * g.fftchannels);
        if (packed) {
            // decode, sideband sign, window and fringe rotation, all in double on the host
            const std::vector<float> w = taper > 0 ? taperedWindow(g.fftchannels, taper, shape)
                                                   : std::vector<float>(g.fftchannels, 1.0f);
            for (int k = 0; k < nc; k++) {
                const int ant = which[3 * k], win = which[3 * k + 1], band = which[3 * k + 2];
                const size_t sp = ((size_t)ant * g.nwin + win) * g.nband + band;
                for (int s = 0; s < g.fftchannels; s++) {
                    const long gs = (long)sampleIndex[(size_t)ant * g.nwin + win] + s;
                    const int frame = (int)(gs / framesamples);
                    const unsigned char* payload = &hPacked[ant * bytesPerAnt + (size_t)frame * (VDIF_HEADER + VDIF_PAYLOAD) + VDIF_HEADER];
                    double x = validFrames[(size_t)ant * nframes + frame]
                             ? hostDecode(payload, nbit, g.nband, (int)(gs - (long)frame * framesamples), band) : 0.0;
                    if ((decodeFlags[(size_t)ant * g.nband + band] & GPUDECODE_BAND_ALTSIGN) && ((datasamples + gs) & 1))
                        x = -x;
                    x *= w[s];
                    const double ph = -TWO_PI * (bigA[sp] * s + bigB[sp]);
                    hIn[(size_t)k * g.fftchannels + s] = make_double2(x * cos(ph), x * sin(ph));
                }
            }
        } else {
            withPre(pre, [&](auto stage) {
                if constexpr (!decltype(stage)::packedInput) gather_exact<<<nc, 256>>>(g, stage, seed, dWhich, dIn);
            });
            CHECK_CUDA(cudaMemcpy(hIn.data(), dIn, hIn.size() * sizeof(double2), cudaMemcpyDeviceToHost));
        }
        std::vector<std::complex<double>> ref, gu, gf;
        std::vector<cuFloatComplex> row(g.nchan);
        for (int k = 0; k < nc; k++) {
            const int ant = which[3 * k], win = which[3 * k + 1], band = which[3 * k + 2];
            const int ab = ant * g.nband + band;
            const double bigA = (double)fse[(size_t)ant * g.nwin + win] - clk[ab] + clkd[ab] / 2;
            for (int c = 0; c < g.nchan; c++) {
                std::complex<double> s = 0;
                for (int n = 0; n < g.fftchannels; n++) {
                    const double2 x = hIn[(size_t)k * g.fftchannels + n];
                    const double ang = -TWO_PI * (double)((long long)n * c % g.fftchannels) / g.fftchannels;
                    s += std::complex<double>(x.x, x.y) * std::polar(1.0, ang);
                }
                const double subFreq = bw * c / g.nchan + fracoff[ab];
                ref.push_back(s * std::polar(1.0, TWO_PI * (bigA * subFreq - std::trunc(bigA * subFreq))));
            }
            for (auto [buf, dst] : {std::make_pair(outUnfused, &gu), std::make_pair(outFused, &gf)}) {
                CHECK_CUDA(cudaMemcpy(row.data(), buf + g.index(ant, win, band, 0), g.nchan * sizeof(cuFloatComplex),
                                      cudaMemcpyDeviceToHost));
                for (auto& v : row) dst->push_back({v.x, v.y});
            }
        }
        const Diff eu = compareHost(gu, ref), ef = compareHost(gf, ref);
        // the fused result must be as close to the exact answer as the unfused one is; from the
        // packed data (decoded exactly, nothing rounded to FP16) both must also be within 1e-5
        // rms, which they are by a factor of 30 (a few 1e-7), so that a decode error shared by
        // both pipelines cannot pass
        const bool refOk = ef.rms <= 2 * eu.rms + 1e-7 && (!packed || (ef.rms < 1e-5 && eu.rms < 1e-5));
        pass &= refOk;
        printf("check: vs double-precision DFT of the exact input (%d spectra): unfused rms %.2e max %.2e, "
               "fused rms %.2e max %.2e  %s\n", nc, eu.rms, eu.max, ef.rms, ef.max, refOk ? "OK" : "FAIL");
        cudaFree(dWhich);
        cudaFree(dIn);
    }

    // ---- packed input: the two-kernel pipeline (decode + fringe -> buffer -> fused FFT) ----
    // the buffer is FP16 or FP32; its first stage is either the fixed-point decode or (no window)
    // gpudifx's own kernel; the fused FFT kernel is the chosen variant, writing to outFused
    struct TwoKernel { const char* name; bool half, gpudifx; };
    std::vector<TwoKernel> twoKernels;
    void* twoBuf = nullptr;
    if (packed) {
        twoKernels = {{"2 kernels: decode + fringe -> FP16, fused FFT", true, false},
                      {"2 kernels: decode + fringe -> FP32, fused FFT", false, false}};
        if (taper == 0) twoKernels.push_back({"2 kernels: gpudifx decode (FP64) -> FP32, fused FFT", false, true});
        twoBuf = devAlloc<float2>(nelem);
    }
    PreVariant preDecode = pre;           // decode + fringe without the window
    preDecode.window = nullptr;
    auto runTwoStage1 = [&](const TwoKernel& tk) {
        if (tk.gpudifx) {
            Unfused tmp = unf;            // gpudifx's kernel, into twoBuf
            tmp.staging = (cuFloatComplex*)twoBuf;
            tmp.decodeRotator = Rotator::DOUBLE;
            tmp.runPre(pre, st);
            return;
        }
        launchPackedSetup(g, pre.pk, st);
        withPre(preDecode, [&](auto stage) {
            const unsigned tpb = decodeToThreads(g);
            const unsigned nb = (unsigned)g.nant * g.nwin;
            if (tk.half) decode_to<8><<<nb, tpb, 0, st>>>(g, stage, (__half2*)twoBuf);
            else         decode_to<8><<<nb, tpb, 0, st>>>(g, stage, (float2*)twoBuf);
        });
        CHECK_CUDA(cudaGetLastError());
    };
    auto runTwoStage2 = [&](const TwoKernel& tk) {
        PreVariant load{};
        load.half = tk.half;
        load.in = twoBuf;
        load.window = pre.window;
        load.hop = g.fftchannels;
        launchFixedPhase(g, pF, st);
        if (batched) chosen->launch(g, load, pF, doAuto, outFused, 0, g.nant, st);
        else for (int a = 0; a < g.nant; a++) chosen->launch(g, load, pF, doAuto, outFused, a, 1, st);
    };
    for (const auto& tk : twoKernels) {
        if (doAuto) CHECK_CUDA(cudaMemsetAsync(autoFused, 0, nauto * sizeof(cuFloatComplex), st));
        CHECK_CUDA(cudaMemsetAsync(outFused, 0, nelem * sizeof(cuFloatComplex), st));
        runTwoStage1(tk);
        runTwoStage2(tk);
        CHECK_CUDA(cudaStreamSynchronize(st));
        const Diff d2 = compareDevice(g, outFused, outUnfused, p.validSamples);
        // FP16 storage rounds every sample (unit roundoff 4.9e-4): ~2e-4 rms relative
        const double tol = tk.half ? 1e-3 : specTol;
        const bool ok = d2.rms < tol && d2.max < 30 * tol;
        pass &= ok;
        printf("check: %s vs unfused: rms %.2e, max %.2e (limits %.0e, %.0e)  %s\n", tk.name, d2.rms, d2.max,
               tol, 30 * tol, ok ? "OK" : "FAIL");
    }

    // ---- packed input, --chunk: the same pipelines W windows at a time, one stream per antenna ----
    // Antenna a's chunks run in order on stream sa[a], with its own part of the 2-kernel buffer
    // (W windows, reused for every chunk), so the antennas' chunks run together and nant x W
    // windows of decoded data are in flight: what an XMAC fed from L2 would need.
    struct Chunked { std::string name; int kind; };          // 0 end-to-end, 1 2 kernels FP16, 2 FP32
    std::vector<Chunked> chunked;
    std::vector<cudaStream_t> sa;
    std::vector<cudaEvent_t> joined;
    cudaEvent_t forked = nullptr;
    const size_t chunkElems = (size_t)std::min(chunk, g.nwin) * g.nband * g.fftchannels;  // per antenna
    if (chunk > 0) {
        const std::string w = std::to_string(chunk);
        char mb[2][32];
        for (int h = 0; h < 2; h++)                          // in flight, all antennas: FP16, FP32
            snprintf(mb[h], sizeof mb[h], " (%.1f MB)", g.nant * chunkElems * (h ? 8.0 : 4.0) / 1e6);
        chunked = {{"end-to-end, chunks of " + w + " windows", 0},
                   {"2 kernels -> FP16, chunks of " + w + " windows" + mb[0], 1},
                   {"2 kernels -> FP32, chunks of " + w + " windows" + mb[1], 2}};
        sa.resize(g.nant);
        joined.resize(g.nant);
        for (int a = 0; a < g.nant; a++) {
            CHECK_CUDA(cudaStreamCreateWithFlags(&sa[a], cudaStreamNonBlocking));
            CHECK_CUDA(cudaEventCreateWithFlags(&joined[a], cudaEventDisableTiming));
        }
        CHECK_CUDA(cudaEventCreateWithFlags(&forked, cudaEventDisableTiming));
    }
    auto runChunked = [&](const Chunked& c) {
        launchFixedPhase(g, pF, st);
        launchPackedSetup(g, pre.pk, st);
        CHECK_CUDA(cudaEventRecord(forked, st));             // the antennas' streams start after these
        for (int a = 0; a < g.nant; a++) CHECK_CUDA(cudaStreamWaitEvent(sa[a], forked, 0));
        for (int w0 = 0; w0 < g.nwin; w0 += chunk)
            for (int a = 0; a < g.nant; a++) {
                const int nw = std::min(chunk, g.nwin - w0);
                cuFloatComplex* out = outFused + g.index(a, w0, 0, 0);
                if (c.kind == 0) {
                    const ChunkView v = chunkView(g, pF, pre, a, w0, nw);
                    chosen->launch(v.g, v.pre, v.p, doAuto, out, 0, 1, sa[a]);
                    continue;
                }
                const ChunkView v = chunkView(g, pF, preDecode, a, w0, nw);
                void* buf = c.kind == 1 ? (void*)((__half2*)twoBuf + a * chunkElems)
                                        : (void*)((float2*)twoBuf + a * chunkElems);
                withPre(v.pre, [&](auto stage) {
                    const unsigned tpb = decodeToThreads(g);
                    if (c.kind == 1) decode_to<8><<<nw, tpb, 0, sa[a]>>>(v.g, stage, (__half2*)buf);
                    else             decode_to<8><<<nw, tpb, 0, sa[a]>>>(v.g, stage, (float2*)buf);
                });
                PreVariant load{};
                load.half = c.kind == 1;
                load.in = buf;
                load.window = pre.window;
                load.hop = g.fftchannels;
                chosen->launch(v.g, load, v.p, doAuto, out, 0, 1, sa[a]);
            }
        for (int a = 0; a < g.nant; a++) {                   // st waits for every antenna
            CHECK_CUDA(cudaEventRecord(joined[a], sa[a]));
            CHECK_CUDA(cudaStreamWaitEvent(st, joined[a], 0));
        }
        CHECK_CUDA(cudaGetLastError());
    };
    for (const auto& c : chunked) {
        if (doAuto) CHECK_CUDA(cudaMemsetAsync(autoFused, 0, nauto * sizeof(cuFloatComplex), st));
        CHECK_CUDA(cudaMemsetAsync(outFused, 0, nelem * sizeof(cuFloatComplex), st));
        runChunked(c);
        CHECK_CUDA(cudaStreamSynchronize(st));
        const Diff d2 = compareDevice(g, outFused, outUnfused, p.validSamples);
        const double tol = c.kind == 1 ? 1e-3 : specTol;
        const bool ok = d2.rms < tol && d2.max < 30 * tol;
        pass &= ok;
        printf("check: %s vs unfused: rms %.2e, max %.2e (limits %.0e, %.0e)  %s\n", c.name.c_str(), d2.rms,
               d2.max, tol, 30 * tol, ok ? "OK" : "FAIL");
    }

    if (doAuto) {
        std::vector<cuFloatComplex> au(nauto), af(nauto);
        CHECK_CUDA(cudaMemcpy(au.data(), autoUnfused, nauto * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(af.data(), autoFused, nauto * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
        // per-band autocorrelations (real) and cross-pol products, each relative to its own rms;
        // float atomics add in a different order in the two pipelines
        double worst[2] = {0, 0};
        for (int part = 0; part < (amode == AutoMode::FULL ? 2 : 1); part++) {
            double d2 = 0, r2 = 0;
            for (int a = 0; a < g.nant; a++)
                for (size_t i = 0; i < (size_t)g.nband * g.nchan; i++) {
                    const size_t idx = (size_t)a * 2 * g.nband * g.nchan + part * (size_t)g.nband * g.nchan + i;
                    const double dx = af[idx].x - au[idx].x, dy = af[idx].y - au[idx].y;
                    d2 += dx * dx + dy * dy;
                    r2 += (double)au[idx].x * au[idx].x + (double)au[idx].y * au[idx].y;
                }
            worst[part] = r2 > 0 ? sqrt(d2 / r2) : (d2 > 0 ? 1.0 : 0.0);
        }
        const double autoTol = 1e-4;
        const bool autoOk = worst[0] < autoTol && worst[1] < autoTol;
        pass &= autoOk;
        printf("check: autocorrelations fused vs unfused: band rms %.2e", worst[0]);
        if (amode == AutoMode::FULL) printf(", cross-pol rms %.2e", worst[1]);
        printf(" (relative; limit %.0e)  %s\n", autoTol, autoOk ? "OK" : "FAIL");
    }
    printf("check: %s\n", pass ? "PASS" : "FAIL");

    // ---- timing ----
    printf("\ntiming per subint (all antennas), median [min] of %d, ms:\n", reps);
    auto line = [&](const char* name, Timing t, double ref) {
        printf("  %-46s %8.3f [%8.3f]", name, t.median, t.min);
        if (ref > 0) printf("  %5.2fx", ref / t.median);
        else printf("        ");
        printf("  %6.1f ns/spectrum  %5.1fx real time\n", t.median * 1e6 / g.spectra(), subintMs / t.median);
    };
    const Timing tU = timeIt(runUnfused, warmup, reps, st);
    line("unfused total", tU, 0);
    if (!direct)
        line(!packed ? "  pre-FFT stage kernel (to FP32 buffer)"
             : rotator == Rotator::DOUBLE ? "  decode + fringe (gpudifx kernel, FP64)"
                                          : "  decode + fringe (fixed-point phases)",
             timeIt([&] { unf.runPre(pre, st); }, warmup, reps, st), 0);
    line("  cuFFT", timeIt([&] { unf.runFFT(pre, outUnfused, st); }, warmup, reps, st), 0);
    line(rotator == Rotator::DOUBLE ? "  rotation (mpifxcorr kernel, FP64)" : "  rotation (fixed-point phases)",
         timeIt([&] {
        if (doAuto) CHECK_CUDA(cudaMemsetAsync(autoUnfused, 0, nauto * sizeof(cuFloatComplex), st));
        unf.runFrac(pU, amode, rotator, outUnfused, st); }, warmup, reps, st), 0);
    for (const auto& v : variants) {
        if (!sweep && v.fpb != chosen->fpb) continue;
        if (!sweepEpt && v.ept != chosen->ept) continue;
        if (!sweepRegcap && v.regcap != chosen->regcap) continue;
        char name[112];
        bool crossInKernel = false;
        const Timing t = timeIt([&] { crossInKernel = runFused(v); }, warmup, reps, st);
        char cap[16] = "";
        if (v.regcap) snprintf(cap, sizeof cap, " cap%u", v.regcap);
        snprintf(name, sizeof name, "fused, %u FFT/block, %u/thr%s%s (%u thr, %u B, %d reg)", v.fpb, v.ept, cap,
                 v.suggested ? "*" : "", v.threads, v.smem, fusedLastRegisters);
        if (amode == AutoMode::FULL && crossInKernel) strncat(name, " +xpol", sizeof name - strlen(name) - 1);
        line(name, t, tU.median);
        if (amode == AutoMode::FULL && !crossInKernel && &v == chosen)
            line("  of which cross-pol pass", timeIt([&] { launchCrossPol(g, pF, outFused, prop.maxThreadsPerBlock, st); },
                                                      warmup, reps, st), 0);
    }
    for (const auto& tk : twoKernels) {
        line(tk.name, timeIt([&] { runTwoStage1(tk); runTwoStage2(tk); }, warmup, reps, st), tU.median);
        line(tk.gpudifx ? "  decode + fringe (gpudifx kernel, FP64), to FP32" :
             tk.half ? "  decode + fringe (fixed-point), to FP16" : "  decode + fringe (fixed-point), to FP32",
             timeIt([&] { runTwoStage1(tk); }, warmup, reps, st), 0);
        line(tk.half ? "  fused FFT, FP16 load" : "  fused FFT, FP32 load", timeIt([&] { runTwoStage2(tk); }, warmup, reps, st), 0);
    }
    for (const auto& c : chunked)
        line(c.name.c_str(), timeIt([&] { runChunked(c); }, warmup, reps, st), tU.median);
    printf("  (fused: FFTs per block, FFT elements per thread, threads and shared memory per block; * cuFFTDx's\n"
           "   suggestion; speed-up is unfused / fused. The 2-kernel lines use the variant --fpb and --ept select)\n");

    for (auto s : sa) cudaStreamDestroy(s);
    for (auto e : joined) cudaEventDestroy(e);
    if (forked) cudaEventDestroy(forked);
    unf.destroy();
    return pass ? 0 : 1;
}
