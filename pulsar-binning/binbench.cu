// binbench - pulsar binning in the XMAC (README.md, BACKLOG.md, docs/mpifxcorr.md): each (FFT
// window, channel) product goes to the bin of the pulse phase at that window and channel, as
// mpifxcorr does (Polyco::getBins, Core::processdata), and the bins are written separately or
// scrunched (summed with their weights).  This first version has:
//   bins    the bin of every (window, frequency, channel) for the subint, from the phase in integer
//           turns (a 32-bit fraction: the window's phase + the channel's dispersion offset, the wrap
//           at 1 free) and mpifxcorr's arbitrary bin ends (a binary search)
//   naive   the obvious port of mpifxcorr's loop: a thread per (frequency, baseline, product,
//           channel) over the windows, its sum held while the bin stays the same and added to that
//           bin's output when it changes (or, scrunched, one sum of weight x product)
// timed against the unbinned staged XMAC (common/xmac.cu), and checked: the bins against mpifxcorr's
// rule in double on the host, the binned visibilities against the host's sums in double, their sum
// over the bins against the unbinned XMAC, and the simulated pulse's power in each bin against the
// profile's.
//
// The phase model, as mpifxcorr's (polyco.cpp): per subint the pulse frequency f at its middle, and
// per channel a dispersion offset f d_c turns, d_c = -DM K / nu_c^2 + DM K / nu_ref^2 (K = 1/0.000241
// s MHz^2, nu_c the channel's frequency, band edge + c df); per window the phase of the pulsar's
// spin at the window's start (--phase mid: its middle).  The phase here is phase0 + t / P + fdot t^2 / 2.
// The bins: ends e[0] < ... < e[N-1] in [0, 1]; bin k (k >= 1) is [e[k-1], e[k]), bin 0 the rest
// (phase < e[0] or >= e[N-1]), as mpifxcorr.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#include "xmac.cuh"
#include "benchutil.cuh"

constexpr int MAXBINS = 256;
constexpr double DM_K = 1.0 / 0.000241;      // s MHz^2 (mpifxcorr's DM_CONSTANT_SECS)
constexpr double TWO32 = 4294967296.0;

struct Setup {
    int nant, nfreq, nchan, nwin, nbl, nbins;
    double bwHz, flo0, df, twin, period, fdot, dm, frefMHz, phase0, fwhm;
    bool mid, scrunch;
    std::vector<double> ends, weights;
    double subint() const { return nwin * twin; }
    double chanMHz(int f, int c) const { return (flo0 + f * bwHz + c * df) * 1e-6; }       // USB
    double phaseAt(double t) const { return phase0 + t / period + 0.5 * fdot * t * t; }    // turns
    double freqAt(double t) const { return 1.0 / period + fdot * t; }
    double dmDelay(int f, int c) const {
        const double nu = chanMHz(f, c);
        return -dm * DM_K / (nu * nu) + dm * DM_K / (frefMHz * frefMHz);
    }
    double tw(int w) const { return (w + (mid ? 0.5 : 0.0)) * twin; }        // the binning's time of window w
    // mpifxcorr's bin of a phase in [0, 1)
    int binOf(double ph) const {
        if (ph < ends[0] || ph >= ends[nbins - 1]) return 0;
        for (int k = nbins - 1; k > 0; k--)
            if (ph < ends[k] && ph >= ends[k - 1]) return k;
        return 0;
    }
};
static double frac(double x) { return x - std::floor(x); }
static uint32_t turns32(double x) { return (uint32_t)(unsigned long long)(frac(x) * TWO32); }

// ---- the bins: one per (window, frequency, channel) ---------------------------------------------
// ends64: the bin ends in units of 2^-32 turns (e[N-1] may be 2^32, so 64 bits)
__global__ void bins_kernel(int nwin, int nfreq, int nchan, int nbins, const unsigned long long* __restrict__ ends64,
                            const uint32_t* __restrict__ winPhase, const uint32_t* __restrict__ dmOff, uint8_t* bins) {
    __shared__ unsigned long long e[MAXBINS];
    for (int i = threadIdx.x; i < nbins; i += blockDim.x) e[i] = ends64[i];
    __syncthreads();
    const size_t n = (size_t)nwin * nfreq * nchan, fc = (size_t)nfreq * nchan;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const uint32_t ph = winPhase[i / fc] + dmOff[i % fc];          // modulo one turn
        int b = 0;
        if (ph >= e[0] && ph < e[nbins - 1]) {                         // the first k >= 1 with ph < e[k]
            int lo = 1, hi = nbins - 1;
            while (lo < hi) {
                const int m = (lo + hi) >> 1;
                if (ph < e[m]) hi = m; else lo = m + 1;
            }
            b = lo;
        }
        bins[i] = (uint8_t)b;
    }
}

// ---- the naive port -----------------------------------------------------------------------------
// A thread per output (frequency, baseline, product, channel), visIndex order; out is [bin][visCount]
// (zeroed first) or, scrunched, [visCount].
template<bool SCRUNCH>
__global__ void bin_naive(XmacShape s, const int2* __restrict__ bls, const __half2* __restrict__ spec,
                          const uint8_t* __restrict__ bins, const float* __restrict__ wts, float2* out) {
    const size_t n = s.visCount();
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const int c = (int)(i % s.nchan);
        const int p = (int)(i / s.nchan % 4);
        const int bl = (int)(i / ((size_t)s.nchan * 4) % s.nbaseline());
        const int f = (int)(i / ((size_t)s.nchan * 4 * s.nbaseline()));
        const int pol1 = p == 1 || p == 3, pol2 = p == 1 || p == 2;
        const int2 ab = bls[bl];
        const size_t wstep = (size_t)s.nband() * s.nchan;
        const __half2* x = spec + s.halfIndex(ab.x, 0, 2 * f + pol1, c);
        const __half2* y = spec + s.halfIndex(ab.y, 0, 2 * f + pol2, c);
        const uint8_t* bw = bins + (size_t)f * s.nchan + c;
        const size_t bstep = (size_t)s.nfreq * s.nchan;
        float2 acc = make_float2(0.f, 0.f);
        int cur = bw[0];
        for (int w = 0; w < s.nwin; w++) {
            const float2 a = __half22float2(x[w * wstep]), b = __half22float2(y[w * wstep]);
            const float re = a.x * b.x + a.y * b.y, im = a.y * b.x - a.x * b.y;     // a conj(b)
            const int bin = bw[w * bstep];
            if (SCRUNCH) {
                const float wt = wts[bin];
                acc.x += wt * re;
                acc.y += wt * im;
            } else {
                if (bin != cur) {
                    float2* o = out + (size_t)cur * n + i;
                    o->x += acc.x;
                    o->y += acc.y;
                    acc = make_float2(0.f, 0.f);
                    cur = bin;
                }
                acc.x += re;
                acc.y += im;
            }
        }
        if (SCRUNCH) out[i] = acc;
        else {
            float2* o = out + (size_t)cur * n + i;
            o->x += acc.x;
            o->y += acc.y;
        }
    }
}

// ---- the station spectra ------------------------------------------------------------------------
// A pulsar: in each band a unit-modulus signal common to every station (a random phase per window,
// band and channel), times the pulse's amplitude at that window and channel (a Gaussian profile at
// phase 0.5, by the true phase: the window's middle, with the channel's dispersion), plus each
// station's own noise.  So a baseline's X_a conj(X_b) has the expectation amp^2: the profile's power.
__device__ static inline float2 gauss(unsigned long long seed, unsigned long long i) {
    const unsigned long long r = splitmix(seed * 0x100000001b3ull + i);
    const float u1 = ((r >> 40) + 0.5f) * (1.0f / 16777216.0f);
    const float u2 = ((r & 0xffffff) + 0.5f) * (1.0f / 16777216.0f);
    const float rad = sqrtf(-2.0f * logf(u1));
    float sn, cs;
    sincospif(2.0f * u2, &sn, &cs);
    return make_float2(rad * cs, rad * sn);
}
__device__ inline float profileAmp(double ph, float sigma) {   // ph in [0, 1)
    const float d = (float)(ph - 0.5) / sigma;
    return expf(-0.25f * d * d);                                // amplitude: the power is exp(-d^2 / 2)
}
__global__ void make_spectra(XmacShape s, const double* __restrict__ truePhase, const double* __restrict__ trueDm,
                             float sigma, unsigned long long seed, float noise, __half2* h) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < s.halfCount(); i += (size_t)gridDim.x * blockDim.x) {
        const int c = (int)(i % s.nchan);
        const size_t t = i / s.nchan;
        const int band = (int)(t % s.nband());
        const size_t aw = t / s.nband();
        const int w = (int)(aw % s.nwin), a = (int)(aw / s.nwin), f = band >> 1;
        const unsigned long long r = splitmix(seed * 0x9e3779b97f4a7c15ull + ((size_t)w * s.nband() + band) * s.nchan + c);
        float sn, cs;
        sincospif(2.f * (float)((r >> 11) * (1.0 / 9007199254740992.0)), &sn, &cs);
        double ph = truePhase[w] + trueDm[(size_t)f * s.nchan + c];
        ph -= floor(ph);
        const float amp = profileAmp(ph, sigma);
        float2 v = make_float2(amp * cs, amp * sn);
        if (noise > 0) {
            const float2 g = gauss(seed + 7 + 1000ull * a, i);
            v.x += noise * g.x;
            v.y += noise * g.y;
        }
        h[i] = __floats2half2_rn(v.x, v.y);
    }
}

// ---- host ---------------------------------------------------------------------------------------
static void usage() {
    printf(
"Usage: binbench [options]\n"
"  --nant N        stations, default 10 (two polarisations each)\n"
"  --nfreq N       dual-polarisation frequencies (2 bands each), default 4\n"
"  --nchan N       channels per band, default 128\n"
"  --bw MHz        bandwidth per band, default 32 (real data: a window is nchan / bw)\n"
"  --flo MHz       the lowest band's lower edge, default 1400 (the bands contiguous, USB)\n"
"  --subint-ms T   the subint, default 25.6 (windows: T / the window, rounded); or --nwin N\n"
"  --period-ms P   the pulse period, default 1.5\n"
"  --fdot X        its frequency's derivative, Hz/s, default 0\n"
"  --dm X          dispersion measure, default 30\n"
"  --fref MHz      the phase model's reference frequency (the polyco's), default the bands' middle\n"
"  --phase0 X      the phase at the subint's start, turns; default: the pulse (0.5) at the subint's\n"
"                  middle (at the reference frequency)\n"
"  --phase start|mid  the binning's time of a window: its start (mpifxcorr, default) or middle\n"
"  --bins B        gate (2 bins: on 0.45-0.55 weight 1, off weight 0; scrunched), 16 (unequal,\n"
"                  the narrowest 1/1024, round the pulse; on 1, off -1/7 when scrunched), 128, or\n"
"                  even:N; default 128\n"
"  --scrunch       write the weighted sum of the bins instead of each bin (gate: always)\n"
"  --fwhm X        the simulated pulse's width, turns (FWHM of its power), default 0.03\n"
"  --noise X       each station's own noise, relative to the pulse's peak amplitude, default 1\n"
"  --only LIST     comma-separated versions to run: naive (default all)\n"
"  --reps N        timed repetitions, default 20; --warmup N, default 3\n"
"  --check N       outputs checked against the host, default 128 (each with all its bins)\n"
"  --csv FILE      append one row per timing\n"
"  --seed N        random seed, default 1;  --device N, default 0\n");
}

static void binLayout(Setup& u, const std::string& b) {
    u.ends.clear();
    u.weights.clear();
    if (b == "gate") {
        u.ends = {0.45, 0.55};
        u.weights = {0.0, 1.0};
        u.scrunch = true;
    } else if (b == "16") {
        // 14 fine bins round 0.5: widths 1/32 1/64 1/128 1/256 1/512 1/1024 1/1024 | mirrored, from
        // 0.4375 to 0.5625; bin 15 to 1 and bin 0 from 0 the off-pulse halves
        const double w[7] = {1.0 / 32, 1.0 / 64, 1.0 / 128, 1.0 / 256, 1.0 / 512, 1.0 / 1024, 1.0 / 1024};
        double e = 0.4375;
        u.ends.push_back(e);
        for (int i = 0; i < 7; i++) u.ends.push_back(e += w[i]);
        for (int i = 6; i >= 0; i--) u.ends.push_back(e += w[i]);
        u.ends.push_back(1.0);
        // on - off, the off-pulse scaled to the on-pulse's width (0.125 against 0.875)
        u.weights.assign(16, 1.0);
        u.weights[0] = u.weights[15] = -0.125 / 0.875;
    } else {
        int n = 128;
        if (b.rfind("even:", 0) == 0) n = std::stoi(b.substr(5));
        else if (b != "128") { usage(); exit(2); }
        for (int k = 1; k <= n; k++) u.ends.push_back((double)k / n);
        u.weights.assign(n, 1.0);
    }
    u.nbins = (int)u.ends.size();
}

int main(int argc, char** argv) {
    Setup u{};
    u.nant = 10; u.nfreq = 4; u.nchan = 128;
    double bwMHz = 32, floMHz = 1400, subintMs = 25.6, noise = 1, frefMHz = 0;
    u.period = 1.5e-3; u.fdot = 0; u.dm = 30; u.phase0 = NAN; u.fwhm = 0.03;
    int nwinArg = 0, reps = 20, warmup = 3, ncheck = 128, device = 0;
    unsigned long long seed = 1;
    std::string only = "naive", csv, bins = "128", phase = "start";
    bool scrunchArg = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(2); }
            return argv[++i];
        };
        if (a == "--nant") u.nant = std::stoi(next());
        else if (a == "--nfreq") u.nfreq = std::stoi(next());
        else if (a == "--nchan") u.nchan = std::stoi(next());
        else if (a == "--bw") bwMHz = std::stod(next());
        else if (a == "--flo") floMHz = std::stod(next());
        else if (a == "--subint-ms") subintMs = std::stod(next());
        else if (a == "--nwin") nwinArg = std::stoi(next());
        else if (a == "--period-ms") u.period = std::stod(next()) * 1e-3;
        else if (a == "--fdot") u.fdot = std::stod(next());
        else if (a == "--dm") u.dm = std::stod(next());
        else if (a == "--fref") frefMHz = std::stod(next());
        else if (a == "--phase0") u.phase0 = std::stod(next());
        else if (a == "--phase") phase = next();
        else if (a == "--bins") bins = next();
        else if (a == "--scrunch") scrunchArg = true;
        else if (a == "--fwhm") u.fwhm = std::stod(next());
        else if (a == "--noise") noise = std::stod(next());
        else if (a == "--only") only = next();
        else if (a == "--reps") reps = std::stoi(next());
        else if (a == "--warmup") warmup = std::stoi(next());
        else if (a == "--check") ncheck = std::stoi(next());
        else if (a == "--csv") csv = next();
        else if (a == "--seed") seed = std::stoull(next());
        else if (a == "--device") device = std::stoi(next());
        else { usage(); return a == "--help" || a == "-h" ? 0 : 2; }
    }
    u.bwHz = bwMHz * 1e6;
    u.flo0 = floMHz * 1e6;
    u.df = u.bwHz / u.nchan;
    u.twin = u.nchan / u.bwHz;
    u.nwin = nwinArg > 0 ? nwinArg : (int)std::lround(subintMs * 1e-3 / u.twin);
    u.frefMHz = frefMHz > 0 ? frefMHz : floMHz + u.nfreq * bwMHz / 2;
    u.mid = phase == "mid";
    if (std::isnan(u.phase0)) u.phase0 = frac(0.5 - u.nwin * u.twin / (2 * u.period));   // the pulse mid-subint
    u.scrunch = scrunchArg;
    binLayout(u, bins);
    u.nbl = u.nant * (u.nant - 1) / 2;
    if (u.nant < 2 || u.nfreq < 1 || u.nchan % 4 || u.nwin < 1 || u.nbins < 2 || u.nbins > MAXBINS || reps < 1 ||
        (phase != "start" && phase != "mid") || u.period <= 0) {
        usage(); return 2;
    }
    for (int k = 1; k < u.nbins; k++)
        if (!(u.ends[k] > u.ends[k - 1])) { fprintf(stderr, "bin ends must ascend\n"); return 2; }
    const bool runNaive = only.find("naive") != std::string::npos;

    XmacShape s{};
    s.nant = u.nant; s.nfreq = u.nfreq; s.nchan = u.nchan; s.nwin = u.nwin; s.fftchannels = 2 * u.nchan;
    CHECK_CUDA(cudaSetDevice(device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    const int sms = prop.multiProcessorCount;
    if (!xmacTensorStagedFits(s)) { fprintf(stderr, "the staged XMAC does not fit %d receivers here\n", s.nrecv()); return 2; }
    const double subMs = u.subint() * 1e3;
    double minW = 1;
    for (int k = 0; k < u.nbins; k++) minW = std::min(minW, k ? u.ends[k] - u.ends[k - 1] : u.ends[0] + 1 - u.ends[u.nbins - 1]);
    printf("GPU %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor, sms);
    printf("%d stations x %d frequencies x 2 pols, %g MHz bands from %g MHz, %d channels (%.4g kHz), %d windows of %.4g us "
           "(subint %.4g ms)\n", u.nant, u.nfreq, bwMHz, floMHz, u.nchan, u.df * 1e-3, u.nwin, u.twin * 1e6, subMs);
    printf("pulsar: P %.4g ms (%.4g windows), DM %g (reference %.6g MHz), phase at the window's %s; %d bins (narrowest "
           "1/%.0f of the period, %.3g windows), %s\n", u.period * 1e3, u.period / u.twin, u.dm, u.frefMHz,
           u.mid ? "middle" : "start (mpifxcorr)", u.nbins, 1 / minW, minW * u.period / u.twin,
           u.scrunch ? "scrunched (weighted sum)" : "each written");
    const double gib = 1.0 / 1073741824.0;
    const size_t nOut = u.scrunch ? s.visCount() : (size_t)u.nbins * s.visCount();
    const size_t nBin = (size_t)u.nwin * u.nfreq * u.nchan;
    printf("GPU memory: station spectra (FP16) %.2f GiB, bins %.3f GiB, output %.3f GiB\n", s.halfCount() * 4.0 * gib,
           nBin * gib, nOut * 8.0 * gib);
    size_t freeMem, totalMem;
    CHECK_CUDA(cudaMemGetInfo(&freeMem, &totalMem));
    if (s.halfCount() * 4.0 + nBin + (nOut + s.visCount()) * 8.0 > 0.9 * freeMem) {
        fprintf(stderr, "not enough GPU memory (%.2f GiB free)\n", freeMem * gib);
        return 2;
    }

    // ---- the phase model: per window and per (frequency, channel) ----
    const double fmid = u.freqAt(u.subint() / 2);                     // mpifxcorr: f at the subint's middle
    std::vector<uint32_t> winPhase(u.nwin), dmOff((size_t)u.nfreq * u.nchan);
    std::vector<double> truePhase(u.nwin), dmTurns((size_t)u.nfreq * u.nchan);
    for (int w = 0; w < u.nwin; w++) {
        winPhase[w] = turns32(u.phaseAt(u.tw(w)));
        truePhase[w] = frac(u.phaseAt((w + 0.5) * u.twin));
    }
    for (int f = 0; f < u.nfreq; f++)
        for (int c = 0; c < u.nchan; c++) {
            dmTurns[(size_t)f * u.nchan + c] = frac(fmid * u.dmDelay(f, c));
            dmOff[(size_t)f * u.nchan + c] = turns32(fmid * u.dmDelay(f, c));
        }
    std::vector<unsigned long long> ends64(u.nbins);
    for (int k = 0; k < u.nbins; k++) ends64[k] = (unsigned long long)std::llround(u.ends[k] * TWO32);
    std::vector<float> wts(u.weights.begin(), u.weights.end());
    std::vector<int2> bls;
    for (int a = 0; a < u.nant; a++)
        for (int b = a + 1; b < u.nant; b++) bls.push_back(make_int2(a, b));
    uint32_t* dWinPhase = devCopy(winPhase);
    uint32_t* dDmOff = devCopy(dmOff);
    double* dTruePhase = devCopy(truePhase);
    double* dDmTurns = devCopy(dmTurns);
    unsigned long long* dEnds = devCopy(ends64);
    float* dWts = devCopy(wts);
    int2* dBls = devCopy(bls);
    uint8_t* dBins = devAlloc<uint8_t>(nBin);

    // ---- inputs ----
    __half2* spec = devAlloc<__half2>(s.halfCount());
    const float sigma = (float)(u.fwhm / (2 * std::sqrt(2 * std::log(2.0))));
    make_spectra<<<4096, 256>>>(s, dTruePhase, dDmTurns, sigma, seed, (float)noise, spec);
    CHECK_CUDA(cudaGetLastError());
    XmacMap map = xmacUploadMap(xmacRegularMap(s));
    float2* vis = devAlloc<float2>(s.visCount());
    float2* out = devAlloc<float2>(nOut);
    cudaStream_t st;
    CHECK_CUDA(cudaStreamCreate(&st));

    auto runBins = [&] {
        bins_kernel<<<sms * 8, 256, 0, st>>>(u.nwin, u.nfreq, u.nchan, u.nbins, dEnds, dWinPhase, dDmOff, dBins);
        CHECK_CUDA(cudaGetLastError());
    };
    auto runXmac = [&] { xmacTensorStagedMapped(s, map, spec, (cuFloatComplex*)vis, sms, st); };
    struct Variant { std::string key, name; std::function<void()> run; };
    std::vector<Variant> variants;
    if (runNaive) variants.push_back({"naive", "naive (a thread per output, over the windows)", [&] {
        if (u.scrunch) bin_naive<true><<<sms * 16, 256, 0, st>>>(s, dBls, spec, dBins, dWts, out);
        else {
            CHECK_CUDA(cudaMemsetAsync(out, 0, nOut * sizeof(float2), st));
            bin_naive<false><<<sms * 16, 256, 0, st>>>(s, dBls, spec, dBins, dWts, out);
        }
        CHECK_CUDA(cudaGetLastError()); }});

    bool pass = true;
    runBins();
    runXmac();
    CHECK_CUDA(cudaStreamSynchronize(st));

    // ---- check: the bins against mpifxcorr's rule in double ----
    std::vector<uint8_t> hBins(nBin);
    CHECK_CUDA(cudaMemcpy(hBins.data(), dBins, nBin, cudaMemcpyDeviceToHost));
    std::vector<double> occupancy(u.nbins, 0.0);
    {
        size_t bad = 0;
        for (int w = 0; w < u.nwin; w++) {
            const double pw = u.phaseAt(u.tw(w));
            for (int f = 0; f < u.nfreq; f++)
                for (int c = 0; c < u.nchan; c++) {
                    const size_t i = ((size_t)w * u.nfreq + f) * u.nchan + c;
                    const int b = u.binOf(frac(pw + fmid * u.dmDelay(f, c)));
                    bad += b != hBins[i];
                    occupancy[hBins[i]] += 1.0 / nBin;
                }
        }
        int empty = 0;
        for (double o : occupancy) empty += o == 0;
        const bool ok = bad <= nBin / 1000000;
        printf("check: bins vs mpifxcorr's rule (double): %zu of %zu differ%s; %d of %d bins empty this subint  %s\n", bad,
               nBin, bad ? " (phases within ~2^-32 of an end)" : "", empty, u.nbins, ok ? "OK" : "FAIL");
        pass &= ok;
    }

    // ---- check: each version against the host's sums (double) of the same FP16 spectra and bins ----
    std::vector<__half2> hSpec(s.halfCount());
    CHECK_CUDA(cudaMemcpy(hSpec.data(), spec, s.halfCount() * sizeof(__half2), cudaMemcpyDeviceToHost));
    std::vector<float2> hVis(s.visCount());
    CHECK_CUDA(cudaMemcpy(hVis.data(), vis, s.visCount() * sizeof(float2), cudaMemcpyDeviceToHost));
    struct Pick { int f, bl, p, c; };
    std::vector<Pick> picks;
    Rng rng{seed * 77 + 3};
    auto rnd = [&](int n) { return std::min(n - 1, (int)(rng.uniform() * n)); };
    for (int i = 0; i < ncheck; i++) picks.push_back({rnd(u.nfreq), rnd(u.nbl), rnd(4), rnd(u.nchan)});
    const int nb = u.scrunch ? 1 : u.nbins;
    std::vector<std::complex<double>> ref(picks.size() * nb), refAll(picks.size());
    for (size_t q = 0; q < picks.size(); q++) {
        const Pick& k = picks[q];
        const int pol1 = k.p == 1 || k.p == 3, pol2 = k.p == 1 || k.p == 2;
        for (int w = 0; w < u.nwin; w++) {
            const float2 a = __half22float2(hSpec[s.halfIndex(bls[k.bl].x, w, 2 * k.f + pol1, k.c)]);
            const float2 b = __half22float2(hSpec[s.halfIndex(bls[k.bl].y, w, 2 * k.f + pol2, k.c)]);
            const std::complex<double> v = std::complex<double>(a.x, a.y) * std::conj(std::complex<double>(b.x, b.y));
            const int bin = hBins[((size_t)w * u.nfreq + k.f) * u.nchan + k.c];
            if (u.scrunch) ref[q] += u.weights[bin] * v;
            else ref[q * nb + bin] += v;
            refAll[q] += v;
        }
    }
    {   // the unbinned staged XMAC against the same sums
        double e2 = 0, r2 = 0;
        for (size_t q = 0; q < picks.size(); q++) {
            const Pick& k = picks[q];
            const float2 g = hVis[s.visIndex(k.f, k.bl, k.p, k.c)];
            e2 += std::norm(std::complex<double>(g.x, g.y) - refAll[q]);
            r2 += std::norm(refAll[q]);
        }
        const double rel = std::sqrt(e2 / std::max(r2, 1e-300));
        printf("check: unbinned staged XMAC vs the host: rms %.2e (relative)  %s\n", rel, rel < 1e-4 ? "OK" : "FAIL");
        pass &= rel < 1e-4;
    }
    std::vector<float2> hOut(nOut);
    for (auto& v : variants) {
        v.run();
        CHECK_CUDA(cudaStreamSynchronize(st));
        CHECK_CUDA(cudaMemcpy(hOut.data(), out, nOut * sizeof(float2), cudaMemcpyDeviceToHost));
        double e2 = 0, r2 = 0, emax = 0, s2 = 0, a2 = 0;
        for (size_t q = 0; q < picks.size(); q++) {
            const Pick& k = picks[q];
            std::complex<double> sum = 0;
            for (int b = 0; b < nb; b++) {
                const float2 g = hOut[(size_t)b * s.visCount() + s.visIndex(k.f, k.bl, k.p, k.c)];
                const std::complex<double> gv(g.x, g.y);
                const double e = std::norm(gv - ref[q * nb + b]);
                e2 += e;
                emax = std::max(emax, e);
                r2 += std::norm(ref[q * nb + b]);
                sum += gv;
            }
            s2 += std::norm(sum - refAll[q]);
            a2 += std::norm(refAll[q]);
        }
        // (a scrunched output can be exactly zero: a gate whose on-pulse is not in the subint)
        const double rms = std::sqrt(r2 / (picks.size() * nb)), rel = r2 > 0 ? std::sqrt(e2 / (picks.size() * nb)) / rms : e2;
        const double relSum = std::sqrt(s2 / a2);           // separate bins: their sum against the unbinned sums
        const bool ok = rel < 1e-4 && (r2 > 0 ? std::sqrt(emax) / rms < 1e-2 : emax == 0) && (u.scrunch || relSum < 1e-4);
        if (r2 == 0) printf("check: %s: the reference is zero (no weighted bin in this subint)\n", v.key.c_str());
        printf("check: %s vs the host: rms %.2e, max %.2e (relative to rms)", v.key.c_str(), rel, std::sqrt(emax) / rms);
        if (!u.scrunch) printf("; summed over the bins vs the unbinned sums %.2e", relSum);
        printf("  %s\n", ok ? "OK" : "FAIL");
        pass &= ok;
    }

    // ---- check: the pulse's power in each bin against the profile's (all baselines, parallel hands) ----
    if (!u.scrunch && runNaive) {
        // measured: mean Re V over (frequency, baseline, p in {AA, BB}, channel), per sample in the bin;
        // expected: the mean of amp^2 over the bin's samples
        std::vector<double> meas(u.nbins, 0), expect(u.nbins, 0), cnt(u.nbins, 0);
        variants[0].run();
        CHECK_CUDA(cudaStreamSynchronize(st));
        CHECK_CUDA(cudaMemcpy(hOut.data(), out, nOut * sizeof(float2), cudaMemcpyDeviceToHost));
        for (int b = 0; b < u.nbins; b++)
            for (int f = 0; f < u.nfreq; f++)
                for (int bl = 0; bl < u.nbl; bl++)
                    for (int p = 0; p < 2; p++)
                        for (int c = 0; c < u.nchan; c++) meas[b] += hOut[(size_t)b * s.visCount() + s.visIndex(f, bl, p, c)].x;
        for (int w = 0; w < u.nwin; w++)
            for (int f = 0; f < u.nfreq; f++)
                for (int c = 0; c < u.nchan; c++) {
                    const int b = hBins[((size_t)w * u.nfreq + f) * u.nchan + c];
                    const double ph = frac(truePhase[w] + dmTurns[(size_t)f * u.nchan + c]);
                    const double d = (ph - 0.5) / sigma;
                    expect[b] += std::exp(-0.5 * d * d);
                    cnt[b] += 1;
                }
        double worst = 0, peak = 0;
        int nused = 0;
        for (int b = 0; b < u.nbins; b++) {
            if (cnt[b] == 0) continue;
            const double n = cnt[b] * u.nbl * 2;                 // products summed
            const double m = meas[b] / n, e = expect[b] / cnt[b];
            // a product's real part: amp^2 + noise terms of variance ~2 amp^2 s^2 + 2 s^4 (gauss() has
            // E|n|^2 = 2, so s = noise); the n products taken as independent (they nearly are)
            const double s2n = noise * noise, sd = std::sqrt((2 * e * s2n + 2 * s2n * s2n) / n);
            worst = std::max(worst, std::fabs(m - e) / sd);
            peak = std::max(peak, e);
            nused++;
        }
        printf("signal: the pulse's power per bin vs the profile's, %d bins with samples: peak %.3f, worst %.1f sigma%s\n",
               nused, peak, worst, worst > 6 ? "  (high: baselines share the signal, so this sigma is a lower bound)" : "");
    }
    if (pass) printf("check: PASS\n");
    else printf("check: FAIL\n");

    // ---- timing ----
    printf("\ntiming per subint, median [min] of %d, ms (ms per ms of data):\n", reps);
    FILE* fc = nullptr;
    if (!csv.empty()) {
        FILE* t = fopen(csv.c_str(), "r");
        const bool header = !t;
        if (t) fclose(t);
        fc = fopen(csv.c_str(), "a");
        if (header) fprintf(fc, "gpu,nant,nfreq,nchan,nwin,period_ms,dm,bins,nbins,scrunch,what,ms,ms_per_ms\n");
    }
    auto line = [&](const std::string& key, const std::string& name, Timing t) {
        printf("  %-52s %8.3f [%8.3f]  %8.4f\n", name.c_str(), t.median, t.min, t.median / subMs);
        if (fc) fprintf(fc, "\"%s\",%d,%d,%d,%d,%.6g,%g,%s,%d,%d,%s,%.5f,%.6f\n", prop.name, u.nant, u.nfreq, u.nchan, u.nwin,
                        u.period * 1e3, u.dm, bins.c_str(), u.nbins, (int)u.scrunch, key.c_str(), t.median, t.median / subMs);
    };
    const Timing tx = timeIt(runXmac, warmup, reps, st);
    line("xmac", "xmac (staged tensor, unbinned)", tx);
    line("bins", "bins (per window, frequency, channel)", timeIt(runBins, warmup, reps, st));
    for (auto& v : variants) {
        const Timing t = timeIt(v.run, warmup, reps, st);
        line(v.key, "binned: " + v.name, t);
        printf("    %.2f x the unbinned XMAC\n", t.median / tx.median);
    }
    if (fc) fclose(fc);
    xmacFreeMap(map);
    return pass ? 0 : 1;
}
