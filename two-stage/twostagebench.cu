// twostagebench - two-stage channelisation (docs/two-stage.md, twostage.cuh): a polyphase filter
// bank of Nc coarse channels, then an FFT of each coarse channel's series, from packed VDIF to FP16
// spectra in the staged XMAC's order; checked against a double-precision reference of the same
// chain and timed, against gpudifx today (decode, cuFFT, rotation) at the same channel count.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <string>
#include <vector>
#include "twostage.cuh"
#include "benchutil.cuh"

static bool* devOnes(size_t n) {
    bool* p = devAlloc<bool>(n);
    CHECK_CUDA(cudaMemset(p, 1, n));
    return p;
}

// The PFB prototype, as two_stage.py's pfb_design: a sinc falling to half at +-O/2 coarse channels
// times a Kaiser window of P L1 taps, beta from Kaiser's formula for the transition O - 1 coarse
// channels wide over P L1 taps (A = 8 + 2.285 x 2 pi (O - 1) P dB).
static double besselI0(double x) {
    double s = 1, t = 1;
    for (int k = 1; k < 60; k++) { t *= (x / (2 * k)) * (x / (2 * k)); s += t; }
    return s;
}
static std::vector<float> pfbTaps(int L1, int P, double O, double& A) {
    A = 8 + 2.285 * 2 * M_PI * (O - 1) * P;
    const double beta = A > 50 ? 0.1102 * (A - 8.7) : A > 21 ? 0.5842 * pow(A - 21, 0.4) + 0.07886 * (A - 21) : 0.0;
    const int N = P * L1;
    std::vector<float> h(N);
    for (int i = 0; i < N; i++) {
        const double n = i - (N - 1) / 2.0, x = 2 * (O / 2) * n / L1;
        const double sinc = x == 0 ? 1.0 : sin(M_PI * x) / (M_PI * x);
        const double r = 2.0 * i / (N - 1) - 1.0;
        h[i] = (float)(sinc * besselI0(beta * sqrt(std::max(0.0, 1 - r * r))) / besselI0(beta));
    }
    return h;
}

static void usage() {
    printf(
"Usage: twostagebench [options]\n"
"  --nc N          coarse channels (PFB channels; stage 1 is an FFT of 2 N points): 16, 32 or 64, default 32\n"
"  --os 4:3|2      stage-1 oversampling (hop 3/4 or 1/2 of 2 N), default 4:3\n"
"  --taps P        PFB taps (segments of 2 N samples), default 8\n"
"  --nchan N       the channel count wanted; the nearest the oversampling allows (stage-2 FFTs of\n"
"                  128-8192 points) is used, default 65536\n"
"  --nant N        antennas, default 6\n"
"  --nband N       bands (2 per dual-pol frequency), default 8\n"
"  --nbit N        1, 2 or 4, default 2\n"
"  --bw MHz        bandwidth per band, default 64\n"
"  --nwin N        stage-2 windows (the XMAC's) per subint, default 16\n"
"  --rate R        delay rate, s/s (the integer shifts and fractional delays per coarse window follow it), default 3e-6\n"
"  --accel A       delay acceleration, s/s^2, default 0\n"
"  --reps N, --warmup N, --seed N, --device N\n"
"  --check N       (antenna, band, coarse channel, window) series checked against a double-precision\n"
"                  reference of the whole chain, default 4 (always coarse channels 0 and N among them)\n"
"  --noref         skip gpudifx today at the same channel count\n"
"  --csv FILE      append the timings as CSV rows (header when new)\n");
}

int main(int argc, char** argv) {
    int Nc = 32, P = 8, nant = 6, nband = 8, nbit = 2, nwin2 = 16, reps = 20, warmup = 3, ncheck = 4, device = 0;
    int target = 65536;
    std::string os = "4:3", csvPath;
    double bw = 64.0, rate = 3e-6, accel = 0;
    bool ref = true;
    unsigned long long seed = 1;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(2); }
            return argv[++i];
        };
        if (a == "--nc") Nc = std::stoi(next());
        else if (a == "--os") os = next();
        else if (a == "--taps") P = std::stoi(next());
        else if (a == "--nchan") target = std::stoi(next());
        else if (a == "--nant") nant = std::stoi(next());
        else if (a == "--nband") nband = std::stoi(next());
        else if (a == "--nbit") nbit = std::stoi(next());
        else if (a == "--bw") bw = std::stod(next());
        else if (a == "--nwin") nwin2 = std::stoi(next());
        else if (a == "--rate") rate = std::stod(next());
        else if (a == "--accel") accel = std::stod(next());
        else if (a == "--reps") reps = std::stoi(next());
        else if (a == "--warmup") warmup = std::stoi(next());
        else if (a == "--check") ncheck = std::stoi(next());
        else if (a == "--seed") seed = std::stoull(next());
        else if (a == "--device") device = std::stoi(next());
        else if (a == "--noref") ref = false;
        else if (a == "--csv") csvPath = next();
        else { usage(); return a == "--help" || a == "-h" ? 0 : 2; }
    }
    // ---- the plan ----
    TwoStagePlan t{};
    t.Nc = Nc;
    t.L1 = 2 * Nc;
    t.P = P;
    double O;
    if (os == "4:3") { t.H = t.L1 * 3 / 4; O = 4.0 / 3; }
    else if (os == "2") { t.H = t.L1 / 2; O = 2.0; }
    else { usage(); return 2; }
    if (!twoStage1Supported(t.L1, nbit) || P < 1 || nband % 2 || (VDIF_PAYLOAD * 8) % (nband * nbit)) { usage(); return 2; }
    int best = -1;
    for (int N2 = 128; N2 <= 8192; N2 *= 2) {
        const int K = N2 * t.H / t.L1, nchan = Nc * K;
        if (N2 * t.H % t.L1 || K % 2) continue;
        if (best < 0 || std::abs(nchan - target) < std::abs(best * t.H / t.L1 * Nc - target)) best = N2;
    }
    t.N2 = best;
    t.K = t.N2 * t.H / t.L1;
    const int nchan = t.nchan();
    const int ncoarse = nwin2 * t.N2;
    const double sampleUs = 1.0 / (2 * bw), subintMs = (double)nwin2 * 2 * nchan * sampleUs / 1000;
    double A;
    const std::vector<float> taps = pfbTaps(t.L1, P, O, A);

    CHECK_CUDA(cudaSetDevice(device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    printf("GPU %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("two stages: %d coarse channels (PFB: %d-point FFT, %d taps, Kaiser %.0f dB stopband, hop %d: O = %s), "
           "stage-2 FFT %d points keeping %d bins: %d channels\n", Nc, t.L1, P, A, t.H, os.c_str(), t.N2, t.K, nchan);
    printf("%d antennas x %d bands, %d-bit, %.0f MHz: %d stage-2 windows of %d samples (subint %.3f ms), %d coarse "
           "windows per antenna and band\n", nant, nband, nbit, bw, nwin2, 2 * nchan, subintMs, ncoarse);

    // ---- packed input and the delay model ----
    // per antenna a delay tau(x) = tau0 + rate x + d2 x^2 (samples; x the sample, d2 = accel / 2 fs):
    // coarse window n (its taps centred at x = n H + (P L1 - 1) / 2) gets the integer shift round(tau)
    // and the fractional rest (in microseconds); a fringe rate per (antenna, band) (turns a sample),
    // its phase continuous.  Stage 1 gets the polynomials (TwoStageModel); the host reference and
    // gpudifx today the per-window values (Windows).
    const double d2 = 0.5 * accel / (2 * bw * 1e6);
    const int framesamples = VDIF_PAYLOAD * 8 / (nband * nbit);
    const int span = P * t.L1;
    const double xmax = (double)ncoarse * t.H + span;
    const long lastSample = (long)(ncoarse - 1) * t.H + span + 64 + (long)(std::fabs(rate) * xmax + std::fabs(d2) * xmax * xmax) + 2;
    const int nframes = (int)((lastSample + framesamples - 1) / framesamples);
    const size_t bytesPerAnt = (size_t)nframes * (VDIF_HEADER + VDIF_PAYLOAD);
    if (!leanDecodeFits(framesamples, span, bytesPerAnt)) { fprintf(stderr, "the lean decode cannot take this shape\n"); return 2; }
    Rng rng{seed * 7919 + 17};
    std::vector<unsigned char> hPacked((size_t)nant * bytesPerAnt);
    for (size_t i = 0; i < hPacked.size(); i += 8) {
        const unsigned long long r = splitmix(seed * 0x1000003 + i);
        memcpy(&hPacked[i], &r, std::min<size_t>(8, hPacked.size() - i));
    }
    std::vector<double> tau0(nant), frate((size_t)nant * nband), fphase0((size_t)nant * nband);
    for (int a = 0; a < nant; a++) tau0[a] = rng.range(8.0, 48.0);
    for (auto& v : frate) v = rng.range(-3e-4, 3e-4);
    for (auto& v : fphase0) v = rng.range(-1.0, 1.0);
    std::vector<double> clk((size_t)nant * nband), clkd(clk.size()), fracoff(clk.size(), 0.0);
    for (size_t i = 0; i < clk.size(); i++) { clk[i] = rng.range(-0.05, 0.05); clkd[i] = rng.range(-0.001, 0.001); }
    std::vector<unsigned char> flags(clk.size(), 0), decodeFlags((size_t)nant * nband);
    for (size_t i = 0; i < decodeFlags.size(); i++) decodeFlags[i] = (i % nband) % 2 ? GPUDECODE_BAND_ALTSIGN : 0;
    const long datasamples = 1000001;
    const int nfreq = nband / 2;
    std::vector<unsigned int> indices((size_t)nfreq * MAX_INDICIES, 0);
    std::vector<int> counts(nfreq, 2);
    for (int f = 0; f < nfreq; f++) { indices[f * MAX_INDICIES] = 2 * f; indices[f * MAX_INDICIES + 1] = 2 * f + 1; }

    // the windows of a geometry (window w starts at w x hop, len samples long, its delay taken at
    // its centre): sample index, fractional delay (us), fringe slope and phase per band
    struct Windows { std::vector<int> sampleIndex; std::vector<float> fse; std::vector<double> bigA, bigB; };
    auto windows = [&](int nwin, int hop, int len) {
        Windows w;
        w.sampleIndex.resize((size_t)nant * nwin);
        w.fse.resize(w.sampleIndex.size());
        w.bigA.resize(w.sampleIndex.size() * nband);
        w.bigB.resize(w.bigA.size());
        for (int a = 0; a < nant; a++)
            for (int n = 0; n < nwin; n++) {
                const double centre = (double)n * hop + (len - 1) / 2.0;
                const double tau = tau0[a] + centre * (rate + d2 * centre);
                const int shift = (int)std::lround(tau);
                const size_t aw = (size_t)a * nwin + n;
                w.sampleIndex[aw] = n * hop + shift;
                w.fse[aw] = (float)((tau - shift) * sampleUs);
                for (int b = 0; b < nband; b++) {
                    const size_t i = aw * nband + b;
                    w.bigA[i] = frate[(size_t)a * nband + b];
                    const double ph = fphase0[(size_t)a * nband + b] + w.bigA[i] * w.sampleIndex[aw];
                    w.bigB[i] = ph - floor(ph);
                }
            }
        return w;
    };
    auto packedInput = [&](const Geometry& g, const Windows& w) {
        PackedInput d{};
        d.packed = devCopy(hPacked);
        d.bytesPerAnt = bytesPerAnt;
        d.ms = mark5_stream{framesamples, VDIF_HEADER + VDIF_PAYLOAD, VDIF_HEADER, nbit, nband, 1};
        d.nframes = nframes;
        d.validFrames = devOnes((size_t)nant * nframes);
        d.sampleIndex = devCopy(w.sampleIndex);
        d.decodeFlags = devCopy(decodeFlags);
        d.datasamples = datasamples;
        d.bigA = devCopy(w.bigA);
        d.bigBred = devCopy(w.bigB);
        d.fringePhase = devAlloc<ulonglong2>(g.spectra());
        d.winDesc = devAlloc<int4>((size_t)g.nant * g.nwin);
        d.fsMagic = frameMagic(framesamples);
        return d;
    };
    auto fracParams = [&](const Geometry& g, const Windows& w) {
        FracParams p{};
        p.fracSampleError = devCopy(w.fse);
        p.validSamples = devOnes((size_t)g.nant * g.nwin);
        p.clockOffsets = devCopy(clk);
        p.clockOffsetsDelta = devCopy(clkd);
        p.fracFreqOffset = devCopy(fracoff);
        p.bandFlags = devCopy(flags);
        p.indices = devCopy(indices);
        p.counts = devCopy(counts);
        p.recordedBandwidth = bw;
        p.numRecordedFreqs = nfreq;
        p.numBlocks = g.nwin;
        p.fixedPhase = devAlloc<ulonglong2>(g.spectra());
        return p;
    };

    // ---- two stages ----
    const Geometry g1{nant, nband, ncoarse, span, Nc};
    const Windows w1 = windows(ncoarse, t.H, span);
    PreVariant pre1{};
    pre1.packed = true;
    pre1.lean = true;
    pre1.window = devCopy(taps);
    pre1.pk = packedInput(g1, w1);
    // stage 1 reads nothing per coarse window: null those arrays (a read would fault)
    pre1.pk.sampleIndex = nullptr;
    pre1.pk.bigA = pre1.pk.bigBred = nullptr;
    pre1.pk.fringePhase = nullptr;
    pre1.pk.winDesc = nullptr;
    FracParams p1 = fracParams(g1, w1);
    p1.fracSampleError = nullptr;
    p1.validSamples = nullptr;
    p1.fixedPhase = nullptr;
    std::vector<double> hDelay((size_t)nant * 3), hFringe((size_t)nant * nband * 3);
    for (int a = 0; a < nant; a++) { hDelay[3 * a] = tau0[a]; hDelay[3 * a + 1] = rate; hDelay[3 * a + 2] = d2; }
    for (size_t i = 0; i < (size_t)nant * nband; i++) { hFringe[3 * i] = fphase0[i]; hFringe[3 * i + 1] = frate[i]; hFringe[3 * i + 2] = 0; }
    const TwoStageModel model{devCopy(hDelay), devCopy(hFringe)};
    double tapPower = 0;
    for (float v : taps) tapPower += (double)v * v;
    const float scale1 = ldexpf(1.f, -(int)std::lround(0.5 * log2(tapPower)));       // coarse values ~1
    const float scale2 = ldexpf(1.f, -(int)std::floor(0.5 * log2((double)t.N2)));    // fine values ~1
    __half2* coarse = devAlloc<__half2>((size_t)nant * nband * (Nc + 1) * ncoarse);
    __half2* spectra = devAlloc<__half2>((size_t)nant * nwin2 * nband * nchan);
    CHECK_CUDA(cudaMemset(spectra, 0, (size_t)nant * nwin2 * nband * nchan * sizeof(__half2)));
    cudaStream_t st;
    CHECK_CUDA(cudaStreamCreate(&st));
    auto stage1 = [&] { launchStage1(t, g1, pre1, p1, model, coarse, scale1, st); };
    auto stage2 = [&] { launchStage2(t, g1, nwin2, coarse, spectra, scale2, st); };
    stage1(); stage2();
    CHECK_CUDA(cudaStreamSynchronize(st));
    const double gib = 1.0 / 1073741824.0;
    printf("GPU memory: coarse series %.2f GiB, spectra %.2f GiB (FP16)\n",
           (size_t)nant * nband * (Nc + 1) * ncoarse * 4 * gib, (size_t)nant * nwin2 * nband * nchan * 4 * gib);

    // ---- check: the whole chain in double precision on the host, for a few (ant, band, c, m) ----
    bool pass = true;
    if (ncheck > 0) {
        std::vector<__half2> got((size_t)nant * nwin2 * nband * nchan);
        CHECK_CUDA(cudaMemcpy(got.data(), spectra, got.size() * sizeof(__half2), cudaMemcpyDeviceToHost));
        double d2 = 0, r2 = 0, dmax = 0;
        size_t cnt = 0;
        for (int k = 0; k < ncheck; k++) {
            const int ant = (int)(rng.uniform() * nant), band = (int)(rng.uniform() * nband), m = (int)(rng.uniform() * nwin2);
            const int c = k == 0 ? 0 : k == 1 ? Nc : 1 + (int)(rng.uniform() * (Nc - 1));
            std::vector<std::complex<double>> z(t.N2);
            for (int j = 0; j < t.N2; j++) {                        // coarse sample n of channel c
                const int n = m * t.N2 + j;
                const size_t aw = (size_t)ant * ncoarse + n, sp = aw * nband + band;
                std::complex<double> acc = 0;
                for (int s = 0; s < span; s++) {
                    const long gs = (long)w1.sampleIndex[aw] + s;
                    const int frame = (int)(gs / framesamples);
                    const unsigned char* payload = &hPacked[ant * bytesPerAnt + (size_t)frame * (VDIF_HEADER + VDIF_PAYLOAD) + VDIF_HEADER];
                    double x = hostDecode(payload, nbit, nband, (int)(gs - (long)frame * framesamples), band);
                    if ((decodeFlags[(size_t)ant * nband + band] & GPUDECODE_BAND_ALTSIGN) && ((datasamples + gs) & 1)) x = -x;
                    x *= taps[s];
                    const double ph = -TWO_PI * (w1.bigA[sp] * s + w1.bigB[sp]);
                    acc += x * std::polar(1.0, ph - TWO_PI * (double)c * (s % t.L1) / t.L1);   // fold + DFT bin c
                }
                const size_t ab = (size_t)ant * nband + band;
                const double bigAval = (double)w1.fse[aw] - clk[ab] + clkd[ab] / 2;
                acc *= std::polar(1.0, TWO_PI * (bigAval * bw / Nc * c));                       // fractional, at f_c
                acc *= std::polar(1.0, -TWO_PI * (double)(((long long)c * n * t.H) % t.L1) / t.L1);   // derotation
                z[j] = acc;
            }
            for (int b = -t.K / 2; b < t.K / 2; b++) {                // stage 2: the kept bins
                const int jj = c * t.K + b;
                if (jj < 0 || jj >= nchan) continue;
                std::complex<double> y = 0;
                for (int j = 0; j < t.N2; j++) y += z[j] * std::polar(1.0, -TWO_PI * (double)((long long)b * j % t.N2) / t.N2);
                const __half2 h = got[(((size_t)ant * nwin2 + m) * nband + band) * nchan + jj];
                const float2 g = __half22float2(h);
                const std::complex<double> gv(g.x / (scale1 * scale2), g.y / (scale1 * scale2));
                d2 += std::norm(gv - y);
                r2 += std::norm(y);
                dmax = std::max(dmax, std::abs(gv - y));
                cnt++;
            }
        }
        // FP16 storage of the coarse series and of the spectra: ~4e-4 rms relative
        const double rms = sqrt(d2 / r2), mx = dmax / sqrt(r2 / std::max<size_t>(cnt, 1)), tol = 2e-3;
        const bool ok = std::isfinite(rms) && rms < tol && mx < 30 * tol;
        pass &= ok;
        printf("check: two stages vs double-precision reference (%d series, coarse channels 0 and %d among them): "
               "rms %.2e, max %.2e (relative to rms; limits %.0e, %.0e)  %s\n", ncheck, Nc, rms, mx, tol, 30 * tol,
               ok ? "OK" : "FAIL");
    }
    printf("check: %s\n", pass ? "PASS" : "FAIL");

    // ---- timing ----
    printf("\ntiming per subint, median [min] of %d, ms (and ms per ms of data):\n", reps);
    FILE* csv = nullptr;
    if (!csvPath.empty()) {
        csv = fopen(csvPath.c_str(), "a");
        if (csv && ftell(csv) == 0)
            fprintf(csv, "gpu,nant,nband,nbit,nc,os,taps,n2,k,nchan,nwin,subint_ms,name,median_ms,min_ms,ms_per_ms\n");
    }
    auto line = [&](const char* name, Timing tm) {
        printf("  %-52s %9.3f [%9.3f]  %8.4f\n", name, tm.median, tm.min, tm.median / subintMs);
        if (csv) fprintf(csv, "\"%s\",%d,%d,%d,%d,%s,%d,%d,%d,%d,%d,%.6g,\"%s\",%.6g,%.6g,%.6g\n", prop.name, nant, nband,
                         nbit, Nc, os.c_str(), P, t.N2, t.K, nchan, nwin2, subintMs, name, tm.median, tm.min,
                         tm.median / subintMs);
    };
    line("two stages: stage 1 (PFB, coarse FFT, corrections)", timeIt(stage1, warmup, reps, st));
    line("two stages: stage 2 (fine FFT)", timeIt(stage2, warmup, reps, st));
    line("two stages: total", timeIt([&] { stage1(); stage2(); }, warmup, reps, st));

    if (ref) {
        // gpudifx today at the same channel count: its decode + fringe kernel (FP64), cuFFT, rotation
        const Geometry gF{nant, nband, nwin2, 2 * nchan, nchan};
        const Windows wF = windows(nwin2, 2 * nchan, 2 * nchan);
        PreVariant preF{};
        preF.packed = true;
        preF.lean = false;
        preF.pk = packedInput(gF, wF);
        FracParams pF = fracParams(gF, wF);
        Unfused unf;
        unf.init(gF, false, true);
        unf.decodeRotator = Rotator::DOUBLE;
        unf.validSamples = pF.validSamples;
        cuFloatComplex* outF = devAlloc<cuFloatComplex>(gF.perAnt() * nant);
        line("gpudifx today (decode FP64, cuFFT, rotation)",
             timeIt([&] { launchFixedPhase(gF, pF, st); unf.run(preF, pF, AutoMode::NONE, Rotator::DOUBLE, outF, st); },
                    warmup, reps, st));
        unf.destroy();
    }
    if (csv) fclose(csv);
    return pass ? 0 : 1;
}
