// xmacbench - the cross-multiply-accumulate (XMAC) stage on its own: gpudifx's kernel, a
// register-tiled FP32 kernel and a tensor-core kernel (xmac.cuh), from the same station spectra,
// checked against a double-precision reference and timed.  See README.md.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#include "xmac.cuh"

__host__ __device__ static inline unsigned long long splitmix(unsigned long long x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
// a complex Gaussian of unit variance per component, from a hash of (seed, i)
__device__ static inline float2 gauss(unsigned long long seed, unsigned long long i) {
    const unsigned long long r = splitmix(seed * 0x100000001b3ull + i);
    const float u1 = ((r >> 40) + 0.5f) * (1.0f / 16777216.0f);
    const float u2 = ((r & 0xffffff) + 0.5f) * (1.0f / 16777216.0f);
    const float rad = sqrtf(-2.0f * logf(u1));
    float sn, cs;
    sincospif(2.0f * u2, &sn, &cs);
    return make_float2(rad * cs, rad * sn);
}

// Station spectra: in each band, a signal common to all stations (correlation rho between any two
// stations, the same polarisation) plus each station's own noise; the polarisations are
// independent.  The channels past nchan, and invalid windows, are zero.
// The FP32 spectra as the staged tensor XMAC's FP16 input (halfIndex), standing in for the station
// kernel's FP16 output; not timed.
__global__ void to_half_spectra(XmacShape s, const float2* spec, __half2* h) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < s.halfCount(); i += (size_t)gridDim.x * blockDim.x) {
        const int c = (int)(i % s.nchan);
        const size_t t = i / s.nchan;             // (ant, win, band)
        h[i] = __float22half2_rn(spec[t * s.fftchannels + c]);
    }
}

__global__ void make_spectra(XmacShape s, unsigned long long seed, float rho, const bool* valid, float2* spec) {
    const size_t n = (size_t)s.nant * s.nwin * s.nband() * s.fftchannels;
    const float a = sqrtf(rho), b = sqrtf(1.f - rho);
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const int c = i % s.fftchannels;
        const size_t t = i / s.fftchannels;          // (ant, win, band)
        const int band = t % s.nband();
        const size_t aw = t / s.nband();             // ant * nwin + win
        const int win = aw % s.nwin;
        if (c >= s.nchan || !valid[aw]) { spec[i] = make_float2(0.f, 0.f); continue; }
        const float2 sig = gauss(seed, (((size_t)win * s.nband() + band) * s.fftchannels + c) * 2 + 1);
        const float2 own = gauss(seed + 7, i * 2);
        spec[i] = make_float2(a * sig.x + b * own.x, a * sig.y + b * own.y);
    }
}

struct Timing { double median, min; };
template<class F>
static Timing timeIt(F f, int warmup, int reps, cudaStream_t st) {
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

static void usage() {
    printf(
"Usage: xmacbench [options]\n"
"  --nant N      stations, default 10 (two polarisations each)\n"
"  --nfreq N     dual-polarisation frequencies (2 bands each), default 4\n"
"  --nchan N     channels per band, default 128\n"
"  --nwin N      FFT windows in the subint, default 10000\n"
"  --complex     complex-sampled data (FFT length = nchan; default real: 2 nchan, so the spectra\n"
"                buffer has 2 nchan per band, as the fused kernel writes it)\n"
"  --bw MHz      bandwidth per band, for the real-time figure, default 64\n"
"  --rho R       correlation of the common signal between stations, default 0.1\n"
"  --invalid F   fraction of (station, window) flagged invalid, default 0\n"
"  --only V      run only gpudifx | tiled | tensor | staged (default all)\n"
"  --reps N      timed repetitions, default 20 (median and minimum reported)\n"
"  --warmup N    untimed repetitions first, default 3\n"
"  --check N     (frequency, channel) pairs checked against a double-precision reference, default 4\n"
"  --seed N      random seed, default 1\n"
"  --device N    CUDA device, default 0\n");
}

int main(int argc, char** argv) {
    XmacShape s{10, 4, 128, 10000, 0};
    bool complexData = false;
    double bw = 64.0, rho = 0.1, invalidFrac = 0.0;
    int reps = 20, warmup = 3, ncheck = 4, device = 0;
    unsigned long long seed = 1;
    std::string only;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(2); }
            return argv[++i];
        };
        if (a == "--nant") s.nant = std::stoi(next());
        else if (a == "--nfreq") s.nfreq = std::stoi(next());
        else if (a == "--nchan") s.nchan = std::stoi(next());
        else if (a == "--nwin") s.nwin = std::stoi(next());
        else if (a == "--complex") complexData = true;
        else if (a == "--bw") bw = std::stod(next());
        else if (a == "--rho") rho = std::stod(next());
        else if (a == "--invalid") invalidFrac = std::stod(next());
        else if (a == "--only") only = next();
        else if (a == "--reps") reps = std::stoi(next());
        else if (a == "--warmup") warmup = std::stoi(next());
        else if (a == "--check") ncheck = std::stoi(next());
        else if (a == "--seed") seed = std::stoull(next());
        else if (a == "--device") device = std::stoi(next());
        else { usage(); return a == "--help" || a == "-h" ? 0 : 2; }
    }
    s.fftchannels = complexData ? s.nchan : 2 * s.nchan;
    if (s.nant < 2 || s.nfreq < 1 || s.nchan < 1 || s.nwin < 16 || reps < 1 || rho < 0 || rho > 1 ||
        (!only.empty() && only != "gpudifx" && only != "tiled" && only != "tensor" && only != "staged")) {
        usage(); return 2;
    }
    const bool runG = only.empty() || only == "gpudifx", runT = only.empty() || only == "tiled",
               runC = only.empty() || only == "tensor";
    bool runS = only.empty() || only == "staged";

    CHECK_CUDA(cudaSetDevice(device));
    if (runS && !xmacTensorStagedFits(s)) {
        printf("staged tensor XMAC: skipped, its shared memory for %d receivers is more than a block may have here\n",
               s.nrecv());
        if (only == "staged") return 2;
        runS = false;
    }
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    const int sms = prop.multiProcessorCount;
    const TensorLayout L = tensorLayout(s);
    const size_t nspec = (size_t)s.nant * s.nwin * s.nband() * s.fftchannels;
    const double gib = 1.0 / 1073741824.0;
    const double specBytes = nspec * sizeof(float2), hBytes = runC ? L.count(s) * sizeof(__half) : 0,
                 halfBytes = runS ? s.halfCount() * sizeof(__half2) : 0;
    const double subintMs = (double)s.nwin * s.fftchannels / (complexData ? bw : 2 * bw) / 1000.0;
    printf("GPU %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor, sms);
    printf("%d stations x %d frequencies x 2 polarisations, %d channels, %d windows (subint %.3f ms); "
           "%d baselines x 4 products, %.3g complex MACs per subint\n", s.nant, s.nfreq, s.nchan, s.nwin, subintMs,
           s.nbaseline(), s.cmacs());
    printf("GPU memory: spectra %.2f GiB (FP32, %d channels a band as the fused kernel writes them), tensor input "
           "%.2f GiB (FP16, %d receivers)\n", specBytes * gib, s.fftchannels, hBytes * gib, s.nrecv());
    size_t freeMem, totalMem;
    CHECK_CUDA(cudaMemGetInfo(&freeMem, &totalMem));
    if (specBytes + hBytes + halfBytes + 2 * s.visCount() * sizeof(float2) > 0.95 * freeMem) {
        fprintf(stderr, "not enough GPU memory (%.2f GiB free): use fewer --nwin\n", freeMem * gib);
        return 2;
    }

    // ---- inputs ----
    std::vector<unsigned char> valid((size_t)s.nant * s.nwin, 1);
    unsigned long long r = seed * 7919 + 17;
    for (auto& v : valid) { r = splitmix(r); v = (r >> 11) * (1.0 / 9007199254740992.0) >= invalidFrac; }
    bool* dValid;
    CHECK_CUDA(cudaMalloc(&dValid, valid.size()));
    CHECK_CUDA(cudaMemcpy(dValid, valid.data(), valid.size(), cudaMemcpyHostToDevice));
    cuFloatComplex* spec;
    CHECK_CUDA(cudaMalloc(&spec, nspec * sizeof(float2)));
    make_spectra<<<4096, 256>>>(s, seed, (float)rho, dValid, (float2*)spec);
    CHECK_CUDA(cudaGetLastError());
    __half* h = nullptr;
    if (runC) {
        CHECK_CUDA(cudaMalloc(&h, L.count(s) * sizeof(__half)));
        CHECK_CUDA(cudaMemset(h, 0, L.count(s) * sizeof(__half)));     // the slack at the end
    }
    __half2* hs = nullptr;
    if (runS) {
        CHECK_CUDA(cudaMalloc(&hs, s.halfCount() * sizeof(__half2)));
        to_half_spectra<<<4096, 256>>>(s, (const float2*)spec, hs);
        CHECK_CUDA(cudaGetLastError());
    }
    cuFloatComplex* vis;
    CHECK_CUDA(cudaMalloc(&vis, s.visCount() * sizeof(float2)));
    cudaStream_t st;
    CHECK_CUDA(cudaStreamCreate(&st));
    XmacGpudifx gx;
    if (runG) gx.init(s, spec, dValid, sms);

    struct Variant { const char* name; double tol; std::function<void()> run; };
    std::vector<Variant> variants;
    if (runG) variants.push_back({"gpudifx (FP32, a thread per baseline, product, channel)", 1e-5, [&] { gx.run(vis, st); }});
    if (runT) variants.push_back({"tiled (FP32, 4 x 4 receiver pairs a thread)", 1e-5,
                                  [&] { xmacTiled(s, spec, vis, sms, st); }});
    if (runC) variants.push_back({"tensor (FP16 in, FP32 sums): transpose + XMAC", 1e-3, [&] {
        xmacTranspose(s, L, spec, h, st);
        xmacTensor(s, L, h, vis, sms, st); }});
    if (runS) variants.push_back({"staged tensor (FP16 spectra, reshuffled in the XMAC)", 1e-3,
                                  [&] { xmacTensorStaged(s, hs, vis, sms, st); }});

    // ---- reference: the (frequency, channel) pairs checked, in double ----
    struct Pick { int f, c; };
    std::vector<Pick> picks;
    for (int k = 0; k < ncheck; k++) {
        r = splitmix(r); const int f = (int)(r % s.nfreq);
        r = splitmix(r); const int c = (int)(r % s.nchan);
        picks.push_back({f, c});
    }
    std::vector<std::complex<double>> ref;
    {
        std::vector<float2> col((size_t)s.nant * s.nwin * 2), rows((size_t)s.nwin * (s.fftchannels + 1));
        for (const auto& pk : picks) {
            for (int a = 0; a < s.nant; a++) {     // the pick's values: [ant][win][pol]
                // per window, from channel c of band 2f to channel c of band 2f + 1
                CHECK_CUDA(cudaMemcpy2D(rows.data(), (s.fftchannels + 1) * sizeof(float2),
                                        spec + s.specIndex(a, 0, 2 * pk.f, pk.c), (size_t)s.nband() * s.fftchannels * sizeof(float2),
                                        (s.fftchannels + 1) * sizeof(float2), s.nwin, cudaMemcpyDeviceToHost));
                for (int w = 0; w < s.nwin; w++) {
                    col[((size_t)a * s.nwin + w) * 2] = rows[(size_t)w * (s.fftchannels + 1)];
                    col[((size_t)a * s.nwin + w) * 2 + 1] = rows[(size_t)w * (s.fftchannels + 1) + s.fftchannels];
                }
            }
            for (int a = 0; a < s.nant; a++)
                for (int b = a + 1; b < s.nant; b++)
                    for (int p = 0; p < 4; p++) {
                        const int pol1 = p < 2 ? p : p - 2, pol2 = p < 2 ? p : 3 - p;
                        std::complex<double> v = 0;
                        for (int w = 0; w < s.nwin; w++) {
                            if (!valid[(size_t)a * s.nwin + w] || !valid[(size_t)b * s.nwin + w]) continue;
                            const float2 x = col[((size_t)a * s.nwin + w) * 2 + pol1];
                            const float2 y = col[((size_t)b * s.nwin + w) * 2 + pol2];
                            v += std::complex<double>(x.x, x.y) * std::conj(std::complex<double>(y.x, y.y));
                        }
                        ref.push_back(v);
                    }
        }
    }

    // ---- check ----
    bool pass = true;
    std::vector<float2> hv(s.visCount());
    for (const auto& v : variants) {
        CHECK_CUDA(cudaMemset(vis, 0xff, s.visCount() * sizeof(float2)));     // NaN: anything unwritten shows
        v.run();
        CHECK_CUDA(cudaStreamSynchronize(st));
        CHECK_CUDA(cudaMemcpy(hv.data(), vis, hv.size() * sizeof(float2), cudaMemcpyDeviceToHost));
        double d2 = 0, r2 = 0, dmax = 0;
        size_t k = 0;
        for (const auto& pk : picks)
            for (int bl = 0; bl < s.nbaseline(); bl++)
                for (int p = 0; p < 4; p++, k++) {
                    const float2 g = hv[s.visIndex(pk.f, bl, p, pk.c)];
                    const std::complex<double> d = std::complex<double>(g.x, g.y) - ref[k];
                    d2 += std::norm(d);
                    r2 += std::norm(ref[k]);
                    dmax = std::max(dmax, std::abs(d));
                }
        const double rref = sqrt(r2 / k), rms = sqrt(d2 / k) / rref, mx = dmax / rref;
        const bool ok = std::isfinite(rms) && rms < v.tol && mx < 30 * v.tol;
        pass &= ok;
        printf("check: %s vs double: rms %.2e, max %.2e (relative to rms; limits %.0e, %.0e)  %s\n", v.name, rms, mx,
               v.tol, 30 * v.tol, ok ? "OK" : "FAIL");
    }
    printf("check: %s\n", pass ? "PASS" : "FAIL");

    // ---- timing ----
    printf("\ntiming per subint, median [min] of %d, ms:\n", reps);
    auto line = [&](const char* name, Timing t) {
        printf("  %-56s %8.3f [%8.3f]  %7.1f Gcmac/s  %6.1fx real time\n", name, t.median, t.min,
               s.cmacs() / t.median * 1e-6, subintMs / t.median);
    };
    for (const auto& v : variants) line(v.name, timeIt(v.run, warmup, reps, st));
    if (runC) {
        line("  of which the transpose (FP32 -> FP16 tensor layout)",
             timeIt([&] { xmacTranspose(s, L, spec, h, st); }, warmup, reps, st));
        line("  of which the tensor-core XMAC", timeIt([&] { xmacTensor(s, L, h, vis, sms, st); }, warmup, reps, st));
    }
    printf("  (Gcmac/s: complex multiply-accumulates of the %d baselines x 4 products per second)\n", s.nbaseline());
    if (runG) gx.destroy();
    return pass ? 0 : 1;
}
