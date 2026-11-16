// fusebench.cu - is fusing the FFT and the fractional-sample correction with cuFFTDx worth it?
//
// Times, on synthetic data in GPU memory (no decode, no host transfers), one subint's
//   unfused:  [pre-FFT stage kernel] + cuFFT + rotation kernel, per antenna (mpifxcorr's shape)
//   fused:    one cuFFTDx kernel per antenna doing the same
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

// Tapered window: tapers of round(taper * n) samples at each end, flat between; raised-cosine
// tapers (Tukey) or sine tapers (whose squares sum to 1 across an overlap of the taper length).
static std::vector<float> taperedWindow(int n, double taper, bool sine) {
    std::vector<float> w(n, 1.0f);
    const int L = (int)std::lround(taper * n);
    for (int i = 0; i < L; i++) {
        const float v = (float)(sine ? sin(0.5 * M_PI * (i + 0.5) / L) : 0.5 * (1.0 - cos(M_PI * (i + 0.5) / L)));
        w[i] = v;
        w[n - 1 - i] = v;
    }
    return w;
}

// ---- comparison ------------------------------------------------------------------------------
// over the nchan used channels of every spectrum: max |a - b|, sum |a - b|^2, sum |b|^2
__global__ void compare_kernel(Geometry g, const cuFloatComplex* a, const cuFloatComplex* b, double* sums,
                               unsigned int* maxbits) {
    const size_t n = g.spectra() * g.nchan;
    double d2 = 0, r2 = 0;
    float dmax = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const size_t idx = (i / g.nchan) * g.fftchannels + i % g.nchan;
        const float dx = a[idx].x - b[idx].x, dy = a[idx].y - b[idx].y;
        const float d = dx * dx + dy * dy;
        d2 += d;
        r2 += (double)b[idx].x * b[idx].x + (double)b[idx].y * b[idx].y;
        dmax = fmaxf(dmax, d);
    }
    atomicAdd(&sums[0], d2);
    atomicAdd(&sums[1], r2);
    atomicMax(maxbits, __float_as_uint(sqrtf(dmax)));   // non-negative floats order as their bits
}

struct Diff { double rms, max; };   // relative to the rms of the reference

static Diff compareDevice(const Geometry& g, const cuFloatComplex* a, const cuFloatComplex* ref) {
    double* sums = devAlloc<double>(2);
    unsigned int* mx = devAlloc<unsigned int>(1);
    CHECK_CUDA(cudaMemset(sums, 0, 2 * sizeof(double)));
    CHECK_CUDA(cudaMemset(mx, 0, sizeof(unsigned int)));
    compare_kernel<<<1024, 256>>>(g, a, ref, sums, mx);
    double h[2];
    unsigned int m;
    CHECK_CUDA(cudaMemcpy(h, sums, sizeof h, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&m, mx, sizeof m, cudaMemcpyDeviceToHost));
    cudaFree(sums);
    cudaFree(mx);
    float fm;
    memcpy(&fm, &m, sizeof fm);
    const double rref = sqrt(h[1] / (g.spectra() * g.nchan));
    return {sqrt(h[0] / (g.spectra() * g.nchan)) / rref, fm / rref};
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
"FFT input (the output of decode + fringe rotation):\n"
"  --input P       fp32 (default) | fp16 storage; the FFT itself is FP32 either way\n"
"  --layout L      window (default: one buffer per FFT window, mpifxcorr's layout) | series\n"
"                  (one continuous series per antenna and band)\n"
"  --taper F       window on the FFT input with tapers of F of the FFT length at each end,\n"
"                  default 0 (no window)\n"
"  --window W      taper shape: tukey (default, raised cosine) | sine\n"
"  --overlap F     windows overlap by F of the FFT length, default 0 (needs --layout series)\n"
"  --wola          short for --layout series --taper 0.125 --overlap 0.25\n"
"Rotation and autocorrelations:\n"
"  --rotator R     unfused pipeline: fixed (default, as the fused kernel) | double (mpifxcorr's\n"
"                  kernel verbatim, FP64 per element)\n"
"  --autocorr M    none (default; autocorrelations in the XMAC) | band (per-band, atomics in the\n"
"                  rotation, as mpifxcorr with DIFX_GPU_XMAC_AUTOCORR unset) | full (plus cross-pol)\n"
"  --singlepol     no band pairs (no cross-pol products)\n"
"  --crosspol-pass fused: form the cross-pol products in a separate pass, not in the kernel\n"
"Fused kernel and run:\n"
"  --fpb N|sweep   FFTs per block: cuFFTDx's suggestion (default), N, or time all\n"
"  --batched       one launch for all antennas (one cuFFT plan; one fused kernel)\n"
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
    bool half = false, series = false, sine = false;
    double bw = 64.0, invalidFrac = 0.0, taper = 0.0, overlap = 0.0;
    AutoMode amode = AutoMode::NONE;
    Rotator rotator = Rotator::FIXED;
    int fpb = 0, reps = 20, warmup = 3, ncheck = 16, device = 0;
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
        else if (a == "--input") half = choose({"fp32", "fp16"}) == 1;
        else if (a == "--layout") series = choose({"window", "series"}) == 1;
        else if (a == "--taper") taper = std::stod(next());
        else if (a == "--window") sine = choose({"tukey", "sine"}) == 1;
        else if (a == "--overlap") overlap = std::stod(next());
        else if (a == "--wola") { series = true; taper = 0.125; overlap = 0.25; }
        else if (a == "--rotator") rotator = choose({"double", "fixed"}) == 0 ? Rotator::DOUBLE : Rotator::FIXED;
        else if (a == "--autocorr") amode = (AutoMode)choose({"none", "band", "full"});
        else if (a == "--singlepol") singlePol = true;
        else if (a == "--crosspol-pass") crossPolPass = true;
        else if (a == "--fpb") { std::string m = next(); if (m == "sweep") sweep = true; else fpb = std::stoi(m); }
        else if (a == "--batched") batched = true;
        else if (a == "--invalid") invalidFrac = std::stod(next());
        else if (a == "--reps") reps = std::stoi(next());
        else if (a == "--warmup") warmup = std::stoi(next());
        else if (a == "--check") ncheck = std::stoi(next());
        else if (a == "--seed") seed = std::stoull(next());
        else if (a == "--device") device = std::stoi(next());
        else { usage(); return a == "--help" || a == "-h" ? 0 : 2; }
    }
    g.fftchannels = complexData ? g.nchan : 2 * g.nchan;
    if (g.nant < 1 || g.nband < 1 || g.nwin < 1 || g.nchan < 1 || reps < 1 || taper < 0 || taper > 0.5 ||
        overlap < 0 || overlap >= 1) { usage(); return 2; }
    if (overlap > 0 && !series) { fprintf(stderr, "--overlap needs --layout series\n"); return 2; }
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
    std::sort(variants.begin(), variants.end(), [](const FusedVariant& a, const FusedVariant& b) { return a.fpb < b.fpb; });
    const FusedVariant* chosen = nullptr;
    for (const auto& v : variants)
        if ((fpb == 0 && v.suggested) || (fpb != 0 && (int)v.fpb == fpb)) chosen = &v;
    if (!chosen) {
        fprintf(stderr, "no fused variant with %d FFTs per block fits this GPU for FFT length %d (available:",
                fpb, g.fftchannels);
        for (const auto& v : variants) fprintf(stderr, " %u", v.fpb);
        fprintf(stderr, ")\n");
        return 2;
    }

    const double sampleTimeUs = complexData ? 1.0 / bw : 1.0 / (2.0 * bw);
    const double subintMs = subintSamples * sampleTimeUs / 1000.0;
    const size_t nelem = g.perAnt() * g.nant;                      // FFT input/output, per window
    const size_t seriesLen = (size_t)(g.nwin - 1) * hop + g.fftchannels;
    const size_t ninput = series ? (size_t)g.nant * g.nband * seriesLen : nelem;
    const double gib = 1.0 / 1073741824.0;
    const char* amodeName[] = {"none", "band", "full"};
    printf("GPU %s (sm_%d%d, %d SMs), CUDA runtime %d, cuFFT %d, cuFFTDx %d\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, rtVersion, cufftVersion, CUFFTDX_VERSION);
    printf("%d antennas x %d bands, %s data, FFT length %d, %d channels kept; %.0f MHz bands -> subint %.3f ms\n",
           g.nant, g.nband, complexData ? "complex" : "real", g.fftchannels, g.nchan, bw, subintMs);
    printf("FFT input: %s, %s layout, ", half ? "FP16" : "FP32", series ? "series" : "per-window");
    if (taper > 0) printf("%s window (%.1f%% tapers), ", sine ? "sine-taper" : "Tukey", 100 * taper);
    else printf("no window, ");
    printf("%d windows per antenna (hop %d = %.1f%% overlap)\n", g.nwin, hop,
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
    pre.window = taper > 0 ? devCopy(taperedWindow(g.fftchannels, taper, sine)) : nullptr;
    if (half) {
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
    const size_t inputBytes = ninput * (half ? sizeof(__half2) : sizeof(float2));
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
    const Diff dSpec = compareDevice(g, outFused, outUnfused);
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
        withPre(pre, [&](auto stage) { gather_exact<<<nc, 256>>>(g, stage, seed, dWhich, dIn); });
        std::vector<double2> hIn((size_t)nc * g.fftchannels);
        CHECK_CUDA(cudaMemcpy(hIn.data(), dIn, hIn.size() * sizeof(double2), cudaMemcpyDeviceToHost));
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
        // the fused result must be as close to the exact answer as the unfused one is
        const bool refOk = ef.rms <= 2 * eu.rms + 1e-7;
        pass &= refOk;
        printf("check: vs double-precision DFT of the exact input (%d spectra): unfused rms %.2e max %.2e, "
               "fused rms %.2e max %.2e  %s\n", nc, eu.rms, eu.max, ef.rms, ef.max, refOk ? "OK" : "FAIL");
        cudaFree(dWhich);
        cudaFree(dIn);
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
        line("  pre-FFT stage kernel (to FP32 buffer)", timeIt([&] { unf.runPre(pre, st); }, warmup, reps, st), 0);
    line("  cuFFT", timeIt([&] { unf.runFFT(pre, outUnfused, st); }, warmup, reps, st), 0);
    line(rotator == Rotator::DOUBLE ? "  rotation (mpifxcorr kernel, FP64)" : "  rotation (fixed-point phases)",
         timeIt([&] {
        if (doAuto) CHECK_CUDA(cudaMemsetAsync(autoUnfused, 0, nauto * sizeof(cuFloatComplex), st));
        unf.runFrac(pU, amode, rotator, outUnfused, st); }, warmup, reps, st), 0);
    for (const auto& v : variants) {
        if (!sweep && &v != chosen) continue;
        char name[96];
        snprintf(name, sizeof name, "fused, %u FFT/block%s (%u thr, %u B smem)", v.fpb, v.suggested ? "*" : "",
                 v.threads, v.smem);
        bool crossInKernel = false;
        const Timing t = timeIt([&] { crossInKernel = runFused(v); }, warmup, reps, st);
        if (amode == AutoMode::FULL && crossInKernel) strncat(name, " +xpol", sizeof name - strlen(name) - 1);
        line(name, t, tU.median);
        if (amode == AutoMode::FULL && !crossInKernel && &v == chosen)
            line("  of which cross-pol pass", timeIt([&] { launchCrossPol(g, pF, outFused, prop.maxThreadsPerBlock, st); },
                                                      warmup, reps, st), 0);
    }
    printf("  (* cuFFTDx's suggested FFTs per block; speed-up is unfused / fused)\n");

    unf.destroy();
    return pass ? 0 : 1;
}
