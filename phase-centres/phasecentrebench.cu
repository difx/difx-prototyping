// phasecentrebench - multiple phase centres after the XMAC (README.md, BACKLOG.md): the subint's
// windows split into pieces (as DiFX's maxNSBetweenUVShifts asks), the staged XMAC writing each
// piece's fine visibilities, then for every phase centre a shift (the differential delay to it, as
// a phase across the band) and an average over channels, accumulated over the pieces into each
// centre's output.  Three versions of the shift, checked against a double-precision reference and
// timed:
//   naive   a thread per output (baseline, frequency, product, output channel, centre), a sincos
//           per fine channel: the obvious port of mpifxcorr's loop
//   simt    the shift as a matrix product per baseline, FP32 on the CUDA cores
//   tensor  the same on the tensor cores (FP16 in, FP32 sums, WMMA)
// The matrix product: per (baseline, piece), with A fine channels averaged into each output
// channel g (fine channel c = g A + k, width df) and Δτ_s the differential delay to centre s,
//     out[f, p, g; s] = e^{2πi Δτ_s (fLO_f + g A df)}  Σ_k  V[f, p, g A + k] e^{2πi Δτ_s k df}
// so the sum is [frequencies x products x output channels, A] x [A, centres] (the frequencies and
// products share Δτ_s), and the phasors are A + 1 per centre instead of one per fine channel.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#include <mma.h>
#include "xmac.cuh"
#include "benchutil.cuh"

using namespace nvcuda;
typedef std::complex<double> cd;

// ---- the delay model ---------------------------------------------------------------------------
// Per antenna a and centre s (s = 0 the pointing centre, 1..S the phase centres), the delay
// tau[a][s](t) = c0 + c1 t + c2 t^2, seconds, t from the start of the subint.  As mpifxcorr
// (Core::uvshiftAndAverageBaselineFreq), for the baseline (a, b) at time t, with D_a(t) =
// (tau[a][s] - tau[a][0]) (1 + tau'[a][0]) (the pointing centre's geometric rate):
//   Δτ_s = D_b - D_a        the delay applied (the phase 2π Δτ_s f at sky frequency f)
//   Δτ'_s = (tau'[b][s] - tau'[b][0]) - (tau'[a][s] - tau'[a][0])     (the decorrelation estimate)
struct Poly { double c0, c1, c2; };
__host__ __device__ inline double pval(const Poly& p, double t) { return p.c0 + t * (p.c1 + t * p.c2); }
__host__ __device__ inline double prate(const Poly& p, double t) { return p.c1 + 2 * t * p.c2; }
__host__ __device__ inline double Dof(const Poly* poly, int S, int a, int s, double t) {   // s: 1..S
    const Poly& p0 = poly[(size_t)a * (S + 1)];
    return (pval(poly[(size_t)a * (S + 1) + s], t) - pval(p0, t)) * (1 + prate(p0, t));
}

struct Setup {
    int nant, nfreq, nchan, nwin, K, A, G, S, nbl;
    double bwHz, flo0, df, twin;      // band width, the lowest band's (USB) edge, fine channel width (Hz); window (s)
    std::vector<int> w0;              // the pieces' first windows, K + 1 (the last = nwin)
    int R() const { return nfreq * 4 * G; }      // rows of the product per baseline: (frequency, product, g)
    double tmid(int k) const { return 0.5 * (w0[k] + w0[k + 1]) * twin; }
    double width(int k) const { return (w0[k + 1] - w0[k]) * twin; }
    double flo(int f) const { return flo0 + f * bwHz; }
    size_t outCount() const { return (size_t)nfreq * nbl * S * 4 * G; }
    size_t outIndex(int f, int bl, int s, int p, int g) const {
        return ((((size_t)f * nbl + bl) * S + s) * 4 + p) * G + g;
    }
};

// what every kernel needs, by value
struct Geo {
    int nfreq, nchan, nbl, A, G, S, K, R, nant;
    double flo0, bwHz, df, twin;
    size_t visCount;
};
static Geo geo(const Setup& u, const XmacShape& s) {
    return {u.nfreq, u.nchan, u.nbl, u.A, u.G, u.S, u.K, u.R(), u.nant, u.flo0, u.bwHz, u.df, u.twin, s.visCount()};
}
__device__ inline size_t visIdx(const Geo& g, int f, int bl, int p, int c) {
    return (((size_t)f * g.nbl + bl) * 4 + p) * g.nchan + c;
}
__device__ inline size_t outIdx(const Geo& g, int f, int bl, int s, int p, int gc) {
    return ((((size_t)f * g.nbl + bl) * g.S + s) * 4 + p) * g.G + gc;
}
__device__ inline float2 cmul(float2 a, float2 b) { return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
__device__ inline float2 phasor(double turns) {      // e^{2πi turns}
    const float t = (float)(turns - floor(turns));
    float sn, cs;
    sincospif(2.f * t, &sn, &cs);
    return make_float2(cs, sn);
}

// ---- the delays: Δτ per (piece, baseline, centre), and mpifxcorr's decorrelation estimate -------
// decorr[f][bl][s] += width x sinc(π Δτ' width fLO_f) x (1 - |Δτ| / (nchan / bw)), per piece, as
// mpifxcorr accumulates its baselineshiftdecorr (the band's edge frequency, as it does).
__host__ __device__ inline double decorrOf(double dt, double dr, double width, double flo, double nchan, double bwHz) {
    double ts = 1, dd = 1;
    const double x = M_PI * dr * width * flo;
    if (fabs(x) > 1e-12) ts = fmax(0.0, sin(x) / x);
    dd = fmax(0.0, 1 - fabs(dt) / (nchan / bwHz));
    return width * ts * dd;
}
__global__ void delays_kernel(Geo g, const Poly* poly, const int2* bls, const int* w0, double* dtau, float* decorr) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.nbl * g.S) return;
    const int bl = i / g.S, s = i % g.S;
    const int a = bls[bl].x, b = bls[bl].y;
    for (int k = 0; k < g.K; k++) {
        const double tm = 0.5 * (w0[k] + w0[k + 1]) * g.twin, width = (w0[k + 1] - w0[k]) * g.twin;
        const double dt = Dof(poly, g.S, b, s + 1, tm) - Dof(poly, g.S, a, s + 1, tm);
        const Poly* pa = poly + (size_t)a * (g.S + 1), * pb = poly + (size_t)b * (g.S + 1);
        const double dr = (prate(pb[s + 1], tm) - prate(pb[0], tm)) - (prate(pa[s + 1], tm) - prate(pa[0], tm));
        dtau[((size_t)k * g.nbl + bl) * g.S + s] = dt;
        for (int f = 0; f < g.nfreq; f++)
            decorr[((size_t)f * g.nbl + bl) * g.S + s] +=
                (float)decorrOf(dt, dr, width, g.flo0 + f * g.bwHz, g.nchan, g.bwHz);
    }
}

// ---- naive: a thread per output, a sincos per fine channel --------------------------------------
__global__ void shift_naive(Geo g, const float2* __restrict__ vis, const double* __restrict__ dtau, float2* out) {
    const size_t n = (size_t)g.nfreq * g.nbl * g.S * 4 * g.G;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const int gc = (int)(i % g.G);
        size_t t = i / g.G;
        const int p = (int)(t % 4); t /= 4;
        const int s = (int)(t % g.S); t /= g.S;
        const int bl = (int)(t % g.nbl), f = (int)(t / g.nbl);
        float2 run = make_float2(0.f, 0.f);
        for (int k = 0; k < g.K; k++) {
            const double dt = dtau[((size_t)k * g.nbl + bl) * g.S + s];
            const double e = dt * (g.flo0 + f * g.bwHz + (double)gc * g.A * g.df);
            const float e0 = (float)(e - floor(e)), x = (float)(dt * g.df);
            const float2* v = vis + k * g.visCount + visIdx(g, f, bl, p, gc * g.A);
            for (int c = 0; c < g.A; c++) {
                float tt = e0 + x * c;
                tt -= floorf(tt);
                float sn, cs;
                sincospif(2.f * tt, &sn, &cs);
                const float2 r = cmul(v[c], make_float2(cs, sn));
                run.x += r.x;
                run.y += r.y;
            }
        }
        const float inv = 1.f / g.A;
        out[i].x += run.x * inv;
        out[i].y += run.y * inv;
    }
}

// ---- the matrix product: a block per (64 rows, TN centres, baseline), looping over the pieces ---
// Per piece: the A channels in chunks of KC (<= 32): the chunk of V (64 rows x KC) and of the inner
// phasors U (KC x TN centres) into shared memory, the product summed over the chunks; then each
// element times its outer phasor e^{2πi Δτ_s (fLO_f + g A df)}, added to the running sums (registers,
// over the pieces).  At the end, out += the sums / A.
// tensor: V in FP16 (scaled by 1 / the piece's windows), the complex product as a real one,
//   [Vre Vim interleaved] x [[cos sin], [-sin cos] per (k, s)] -> [re im interleaved], 16x16x16 WMMA,
//   8 warps each 16 rows x TN real columns; TN = 64 (32 for fewer centres).
// simt: FP32, TN = 32, a thread 2 rows x 4 centres.
// The outer phase in FP32 without losing the bits a double would keep: Δτ fLO_f mod 1 per (frequency,
// centre), and y = Δτ A df mod 1 as yh + yl (two floats), g y mod 1 = frac(g yh) + (the fma's
// rounding error of g yh) + g yl.
constexpr int TM = 64, KCMAX = 32, NT = 256, MAXF = 16;
template<int TN> struct TileSizes {
    static constexpr int LDV = 2 * KCMAX + 8, LDU = 2 * TN + 8, LDC = 2 * TN + 4, LDVF = KCMAX + 1;
    static constexpr int BUF_TC = (TM * LDV + 2 * KCMAX * LDU) * 2, BUF_SIMT = (TM * LDVF + KCMAX * TN) * 8;
    static constexpr int BUF_C = TM * LDC * 4;
    static constexpr int BUF = BUF_TC > BUF_SIMT ? (BUF_TC > BUF_C ? BUF_TC : BUF_C) : (BUF_SIMT > BUF_C ? BUF_SIMT : BUF_C);
};

template<bool TC, int TN>
__global__ void __launch_bounds__(NT) shift_tile(Geo g, const float2* __restrict__ vis, const double* __restrict__ dtau,
                                                 const float* __restrict__ wk, float2* out) {
    using Z = TileSizes<TN>;
    constexpr int LDV = Z::LDV, LDU = Z::LDU, LDC = Z::LDC, LDVF = Z::LDVF, NJ = TN / 4;
    __shared__ float sx[TN], syh[TN], syl[TN], sef[MAXF][TN];
    __shared__ long long rowOff[TM];
    __shared__ __align__(32) unsigned char buf[Z::BUF];    // the chunk's V and U; then the piece's product
    float* Cf = reinterpret_cast<float*>(buf);
    const int tid = threadIdx.x, r0 = blockIdx.x * TM, s0 = blockIdx.y * TN, bl = blockIdx.z;
    const int KC = min(g.A, KCMAX);
    if (tid < TM) {
        const int q = r0 + tid;
        rowOff[tid] = q < g.R ? (long long)visIdx(g, q / (4 * g.G), bl, (q / g.G) % 4, (q % g.G) * g.A) : -1;
    }
    // the thread's elements of the result: row tid % 64, centres tid / 64 + 4 j
    const int row = tid % TM, rr = r0 + row;
    const int fr = rr / (4 * g.G), gr = rr % g.G;
    float2 run[NJ];
    for (int j = 0; j < NJ; j++) run[j] = make_float2(0.f, 0.f);

    for (int k = 0; k < g.K; k++) {
        for (int i = tid; i < TN * (g.nfreq + 1); i += NT) {
            const int sl = i % TN, f = i / TN - 1, s = s0 + sl;
            const double dt = s < g.S ? dtau[((size_t)k * g.nbl + bl) * g.S + s] : 0.0;
            if (f < 0) {
                sx[sl] = (float)(dt * g.df);
                double y = dt * g.A * g.df;
                y -= floor(y);
                const float yh = (float)y;
                syh[sl] = yh;
                syl[sl] = (float)(y - yh);
            } else {
                const double e = dt * (g.flo0 + f * g.bwHz);
                sef[f][sl] = (float)(e - floor(e));
            }
        }
        const float inv = TC ? 1.f / wk[k] : 1.f;
        const float2* vk = vis + k * g.visCount;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[TN / 16];
        float2 a8[8];
        if (TC) for (int q = 0; q < TN / 16; q++) wmma::fill_fragment(acc[q], 0.f);
        else for (int j = 0; j < 8; j++) a8[j] = make_float2(0.f, 0.f);
        __syncthreads();
        for (int c0 = 0; c0 < g.A; c0 += KC) {
            for (int i = tid; i < TM * KC / 2; i += NT) {    // V: 64 rows x KC, two channels a load
                const int r = i / (KC / 2), kk = 2 * (i % (KC / 2));
                const long long o = rowOff[r];
                const float4 v = o >= 0 ? *reinterpret_cast<const float4*>(vk + o + c0 + kk) : make_float4(0.f, 0.f, 0.f, 0.f);
                if (TC) {
                    __half2* V = reinterpret_cast<__half2*>(buf) + (r * LDV + 2 * kk) / 2;
                    V[0] = __floats2half2_rn(v.x * inv, v.y * inv);
                    V[1] = __floats2half2_rn(v.z * inv, v.w * inv);
                } else {
                    float2* V = reinterpret_cast<float2*>(buf) + r * LDVF + kk;
                    V[0] = make_float2(v.x, v.y);
                    V[1] = make_float2(v.z, v.w);
                }
            }
            for (int i = tid; i < KC * TN; i += NT) {        // U: KC x TN centres
                const int kk = i / TN, sl = i % TN;
                float t = sx[sl] * (float)(c0 + kk);
                t -= floorf(t);
                float sn, cs;
                sincospif(2.f * t, &sn, &cs);
                if (TC) {
                    __half2* U = reinterpret_cast<__half2*>(reinterpret_cast<__half*>(buf) + TM * LDV);
                    U[((2 * kk) * LDU + 2 * sl) / 2] = __floats2half2_rn(cs, sn);
                    U[((2 * kk + 1) * LDU + 2 * sl) / 2] = __floats2half2_rn(-sn, cs);
                } else reinterpret_cast<float2*>(buf)[TM * LDVF + kk * TN + sl] = make_float2(cs, sn);
            }
            __syncthreads();
            if (TC) {
                const __half* V = reinterpret_cast<const __half*>(buf), * U = V + TM * LDV;
                const int w = tid / 32, mi = w % 4, nj = w / 4;
                for (int kq = 0; kq < 2 * KC / 16; kq++) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                    wmma::load_matrix_sync(fa, V + mi * 16 * LDV + kq * 16, LDV);
                    for (int q = 0; q < TN / 16; q++) {
                        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
                        wmma::load_matrix_sync(fb, U + kq * 16 * LDU + nj * TN + q * 16, LDU);
                        wmma::mma_sync(acc[q], fa, fb, acc[q]);
                    }
                }
            } else {
                const float2* V = reinterpret_cast<const float2*>(buf), * U = V + TM * LDVF;
                const int rg = tid / 8, sc = tid % 8;
                for (int kk = 0; kk < KC; kk++) {
                    const float2 v0 = V[(2 * rg) * LDVF + kk], v1 = V[(2 * rg + 1) * LDVF + kk];
                    for (int j = 0; j < 4; j++) {
                        const float2 u = U[kk * TN + sc + 8 * j];
                        a8[2 * j].x += v0.x * u.x - v0.y * u.y;
                        a8[2 * j].y += v0.x * u.y + v0.y * u.x;
                        a8[2 * j + 1].x += v1.x * u.x - v1.y * u.y;
                        a8[2 * j + 1].y += v1.x * u.y + v1.y * u.x;
                    }
                }
            }
            __syncthreads();
        }
        // the piece's product into Cf [64 rows][TN centres x (re, im)] (over the chunk buffers)
        if (TC) {
            const int w = tid / 32, mi = w % 4, nj = w / 4;
            for (int q = 0; q < TN / 16; q++)
                wmma::store_matrix_sync(Cf + mi * 16 * LDC + nj * TN + q * 16, acc[q], LDC, wmma::mem_row_major);
        } else {
            const int rg = tid / 8, sc = tid % 8;
            for (int j = 0; j < 4; j++)
                for (int h = 0; h < 2; h++) {
                    Cf[(2 * rg + h) * LDC + 2 * (sc + 8 * j)] = a8[2 * j + h].x;
                    Cf[(2 * rg + h) * LDC + 2 * (sc + 8 * j) + 1] = a8[2 * j + h].y;
                }
        }
        __syncthreads();
        // the outer phasors, into the running sums
        if (rr < g.R) {
            const float scale = TC ? wk[k] : 1.f, gf = (float)gr;
            for (int j = 0; j < NJ; j++) {
                const int sl = tid / TM + 4 * j;
                const float p = gf * syh[sl], e1 = fmaf(gf, syh[sl], -p);
                float t = (p - floorf(p)) + (e1 + gf * syl[sl]) + sef[fr][sl];
                t -= floorf(t);
                float sn, cs;
                sincospif(2.f * t, &sn, &cs);
                const float2 v = make_float2(Cf[row * LDC + 2 * sl] * scale, Cf[row * LDC + 2 * sl + 1] * scale);
                const float2 r = cmul(v, make_float2(cs, sn));
                run[j].x += r.x;
                run[j].y += r.y;
            }
        }
        __syncthreads();
    }
    if (rr >= g.R) return;
    const int p = (rr / g.G) % 4;
    const float invA = 1.f / g.A;
    for (int j = 0; j < NJ; j++) {
        const int s = s0 + tid / TM + 4 * j;
        if (s >= g.S) continue;
        float2& o = out[outIdx(g, fr, bl, s, p, gr)];
        o.x += run[j].x * invA;
        o.y += run[j].y * invA;
    }
}

// ---- the station spectra -----------------------------------------------------------------------
// One source, at phase centre src (1..S): in each band a unit-modulus signal common to every station
// (a random phase per window, band and channel; the polarisations independent), at station a
// delayed by D_a(t_w) relative to the pointing centre, so that its visibilities X_a conj(X_b) carry
// e^{-2πi Δτ f} and the shift to src undoes it; plus, optionally, each station's own noise.
// Dw[a][w] = D_a at the window's centre.
__device__ static inline float2 gauss(unsigned long long seed, unsigned long long i) {
    const unsigned long long r = splitmix(seed * 0x100000001b3ull + i);
    const float u1 = ((r >> 40) + 0.5f) * (1.0f / 16777216.0f);
    const float u2 = ((r & 0xffffff) + 0.5f) * (1.0f / 16777216.0f);
    const float rad = sqrtf(-2.0f * logf(u1));
    float sn, cs;
    sincospif(2.0f * u2, &sn, &cs);
    return make_float2(rad * cs, rad * sn);
}
__global__ void make_spectra(XmacShape s, Geo g, const double* Dw, unsigned long long seed, float noise, __half2* h) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < s.halfCount(); i += (size_t)gridDim.x * blockDim.x) {
        const int c = (int)(i % s.nchan);
        const size_t t = i / s.nchan;
        const int band = (int)(t % s.nband());
        const size_t aw = t / s.nband();
        const int w = (int)(aw % s.nwin), a = (int)(aw / s.nwin), f = band >> 1;
        const unsigned long long r = splitmix(seed * 0x9e3779b97f4a7c15ull + ((size_t)w * s.nband() + band) * s.nchan + c);
        float sn, cs;
        sincospif(2.f * (float)((r >> 11) * (1.0 / 9007199254740992.0)), &sn, &cs);
        float2 v = cmul(make_float2(cs, sn), phasor(Dw[aw] * (g.flo0 + f * g.bwHz + c * g.df)));
        if (noise > 0) {
            const float2 n = gauss(seed + 7 + 1000ull * a, i);
            v.x += noise * n.x;
            v.y += noise * n.y;
        }
        h[i] = __floats2half2_rn(v.x, v.y);
    }
}

// ---- host ---------------------------------------------------------------------------------------
static void usage() {
    printf(
"Usage: phasecentrebench [options]\n"
"  --nant N        stations, default 10 (two polarisations each)\n"
"  --nfreq N       dual-polarisation frequencies (2 bands each), default 4\n"
"  --nchan N       fine channels per band, default 8192\n"
"  --bw MHz        bandwidth per band, default 32 (real data: a window is nchan / bw)\n"
"  --flo MHz       the lowest band's lower edge, default 1400 (upper sideband; the bands contiguous)\n"
"  --nwin N        FFT windows in the subint, default 100\n"
"  --pieces K      pieces the subint's windows are split into (as evenly as possible), each shifted\n"
"                  at its midpoint, default 1\n"
"  --uvshift-ns T  instead of --pieces: DiFX's maxNSBetweenUVShifts, K = ceil(subint / T)\n"
"  --avg A         fine channels averaged into each output channel, default 64 (a power of 2, >= 8\n"
"                  for the matrix-product versions)\n"
"  --centres S     phase centres, default 100\n"
"  --field-us D    the centres' delay offsets from the pointing centre, per station, uniform in\n"
"                  +-D us (default 20), their rates +-7.3e-5 x that per second (Earth rotation)\n"
"  --source S0     the phase centre (1..S) the simulated source is at, default 1\n"
"  --noise X       each station's own noise, relative to the source, default 0 (the source-centre\n"
"                  check is exact only without it)\n"
"  --only LIST     comma-separated versions to run: naive,simt,tensor (default all)\n"
"  --reps N        timed repetitions, default 20; --warmup N, default 3\n"
"  --check N       outputs checked against the double-precision shift, default 256 (x 4 products)\n"
"  --csv FILE      append one row per timing\n"
"  --seed N        random seed, default 1;  --device N, default 0\n");
}

int main(int argc, char** argv) {
    Setup u{};
    u.nant = 10; u.nfreq = 4; u.nchan = 8192; u.nwin = 100; u.K = 1; u.A = 64; u.S = 100;
    double bwMHz = 32, floMHz = 1400, fieldUs = 20, noise = 0, uvshiftNs = 0;
    int src = 1, reps = 20, warmup = 3, ncheck = 256, device = 0;
    unsigned long long seed = 1;
    std::string only = "naive,simt,tensor", csv;
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
        else if (a == "--nwin") u.nwin = std::stoi(next());
        else if (a == "--pieces") u.K = std::stoi(next());
        else if (a == "--uvshift-ns") uvshiftNs = std::stod(next());
        else if (a == "--avg") u.A = std::stoi(next());
        else if (a == "--centres") u.S = std::stoi(next());
        else if (a == "--field-us") fieldUs = std::stod(next());
        else if (a == "--source") src = std::stoi(next());
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
    const double subintMs = u.nwin * u.twin * 1e3;
    if (uvshiftNs > 0) u.K = (int)std::ceil(subintMs * 1e6 / uvshiftNs - 1e-9);
    const bool runNaive = only.find("naive") != std::string::npos, runSimt = only.find("simt") != std::string::npos,
               runTensor = only.find("tensor") != std::string::npos;
    if (u.nant < 2 || u.nfreq < 1 || u.nchan % 4 || u.nwin < 1 || u.K < 1 || u.K > u.nwin || u.S < 1 || u.A < 1 ||
        (u.A & (u.A - 1)) || u.nchan % u.A || src < 1 || src > u.S || reps < 1) {
        usage(); return 2;
    }
    const bool tiles = u.A >= 8 && u.nfreq <= MAXF;
    if (!tiles && (runSimt || runTensor)) printf("simt, tensor: skipped (they need --avg >= 8 and --nfreq <= %d)\n", MAXF);
    u.G = u.nchan / u.A;
    u.nbl = u.nant * (u.nant - 1) / 2;
    for (int k = 0; k <= u.K; k++) u.w0.push_back((int)std::lround((double)k * u.nwin / u.K));
    const XmacShape s{u.nant, u.nfreq, u.nchan, u.nwin, u.nchan};
    const Geo G = geo(u, s);

    CHECK_CUDA(cudaSetDevice(device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    const int sms = prop.multiProcessorCount;
    if (!xmacTensorStagedFits(s)) { fprintf(stderr, "the staged XMAC does not fit %d receivers here\n", s.nrecv()); return 2; }
    int wmin = u.nwin, wmax = 0;
    for (int k = 0; k < u.K; k++) { wmin = std::min(wmin, u.w0[k + 1] - u.w0[k]); wmax = std::max(wmax, u.w0[k + 1] - u.w0[k]); }
    printf("GPU %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor, sms);
    printf("%d stations x %d frequencies x 2 pols, %g MHz bands from %g MHz, %d channels (%.4g kHz), %d windows of %.4g us "
           "(subint %.4g ms)\n", u.nant, u.nfreq, bwMHz, floMHz, u.nchan, u.df * 1e-3, u.nwin, u.twin * 1e6, subintMs);
    printf("%d piece%s of %d-%d windows (%.4g-%.4g ms); %d centres; %d channels averaged into %d output channels "
           "(%.4g kHz)\n", u.K, u.K > 1 ? "s" : "", wmin, wmax, wmin * u.twin * 1e3, wmax * u.twin * 1e3, u.S, u.A, u.G,
           u.A * u.df * 1e-3);
    const double gib = 1.0 / 1073741824.0;
    const double bytesSpec = s.halfCount() * 4.0, bytesFine = (double)u.K * s.visCount() * 8,
                 bytesOut = u.outCount() * 8.0;
    printf("GPU memory: station spectra (FP16) %.2f GiB, fine visibilities %.2f GiB (%d x %.3f), output %.2f GiB\n",
           bytesSpec * gib, bytesFine * gib, u.K, s.visCount() * 8.0 * gib, bytesOut * gib);
    size_t freeMem, totalMem;
    CHECK_CUDA(cudaMemGetInfo(&freeMem, &totalMem));
    if (bytesSpec + bytesFine + 2 * bytesOut > 0.9 * freeMem) {
        fprintf(stderr, "not enough GPU memory (%.2f GiB free)\n", freeMem * gib);
        return 2;
    }

    // ---- the delay model ----
    Rng rng{seed * 31 + 5};
    std::vector<Poly> poly((size_t)u.nant * (u.S + 1));
    const double field = fieldUs * 1e-6;
    for (int a = 0; a < u.nant; a++) {
        const Poly p0{rng.range(-20e-3, 20e-3), rng.range(-3e-6, 3e-6), rng.range(-1e-10, 1e-10)};
        poly[(size_t)a * (u.S + 1)] = p0;
        for (int c = 1; c <= u.S; c++) {
            const double d0 = rng.range(-field, field), d1 = rng.range(-1, 1) * 7.3e-5 * field;
            poly[(size_t)a * (u.S + 1) + c] = {p0.c0 + d0, p0.c1 + d1, p0.c2};
        }
    }
    std::vector<int2> bls;
    for (int a = 0; a < u.nant; a++)
        for (int b = a + 1; b < u.nant; b++) bls.push_back(make_int2(a, b));
    std::vector<double> Dw((size_t)u.nant * u.nwin);
    for (int a = 0; a < u.nant; a++)
        for (int w = 0; w < u.nwin; w++) Dw[(size_t)a * u.nwin + w] = Dof(poly.data(), u.S, a, src, (w + 0.5) * u.twin);
    std::vector<float> wk(u.K);
    for (int k = 0; k < u.K; k++) wk[k] = (float)(u.w0[k + 1] - u.w0[k]);
    Poly* dPoly = devCopy(poly);
    int2* dBls = devCopy(bls);
    int* dW0 = devCopy(u.w0);
    double* dDw = devCopy(Dw);
    float* dWk = devCopy(wk);
    double* dTau = devAlloc<double>((size_t)u.K * u.nbl * u.S);
    float* dDecorr = devAlloc<float>((size_t)u.nfreq * u.nbl * u.S);

    // ---- inputs: the station spectra; the map for the XMAC over the whole subint ----
    __half2* spec = devAlloc<__half2>(s.halfCount());
    make_spectra<<<4096, 256>>>(s, G, dDw, seed, (float)noise, spec);
    CHECK_CUDA(cudaGetLastError());
    XmacMap map = xmacUploadMap(xmacRegularMap(s));
    float2* fine = devAlloc<float2>((size_t)u.K * s.visCount());
    float2* out = devAlloc<float2>(u.outCount());
    cudaStream_t st;
    CHECK_CUDA(cudaStreamCreate(&st));

    auto runXmac = [&] {
        for (int k = 0; k < u.K; k++) {
            XmacShape ps = s;
            ps.nwin = u.w0[k + 1] - u.w0[k];
            xmacTensorStagedMapped(ps, map, spec + (size_t)u.w0[k] * s.nband() * s.nchan,
                                   (cuFloatComplex*)(fine + (size_t)k * s.visCount()), sms, st);
        }
    };
    auto runDelays = [&] {
        CHECK_CUDA(cudaMemsetAsync(dDecorr, 0, (size_t)u.nfreq * u.nbl * u.S * sizeof(float), st));
        delays_kernel<<<(u.nbl * u.S + 127) / 128, 128, 0, st>>>(G, dPoly, dBls, dW0, dTau, dDecorr);
        CHECK_CUDA(cudaGetLastError());
    };
    auto tgrid = [&](int tn) { return dim3((u.R() + TM - 1) / TM, (u.S + tn - 1) / tn, u.nbl); };
    const bool wide = u.S > 32;      // the tensor version: 64 centres a block when there are more than 32
    auto zeroOut = [&] { CHECK_CUDA(cudaMemsetAsync(out, 0, u.outCount() * sizeof(float2), st)); };
    struct Variant { std::string name; double tol; std::function<void()> run; };
    std::vector<Variant> variants;
    if (runNaive) variants.push_back({"naive (a thread per output, sincos per channel)", 1e-4, [&] {
        zeroOut(); shift_naive<<<sms * 16, 256, 0, st>>>(G, fine, dTau, out); CHECK_CUDA(cudaGetLastError()); }});
    if (runSimt && tiles) variants.push_back({"simt (matrix product, FP32)", 1e-4, [&] {
        zeroOut(); shift_tile<false, 32><<<tgrid(32), NT, 0, st>>>(G, fine, dTau, dWk, out); CHECK_CUDA(cudaGetLastError()); }});
    if (runTensor && tiles) variants.push_back({"tensor (matrix product, FP16 WMMA)", 3e-3, [&] {
        zeroOut(); if (wide) shift_tile<true, 64><<<tgrid(64), NT, 0, st>>>(G, fine, dTau, dWk, out);
        else shift_tile<true, 32><<<tgrid(32), NT, 0, st>>>(G, fine, dTau, dWk, out);
        CHECK_CUDA(cudaGetLastError()); }});

    runXmac();
    runDelays();
    CHECK_CUDA(cudaStreamSynchronize(st));
    bool pass = true;

    // ---- check: the delays and the decorrelation estimate against the host ----
    std::vector<double> hTau((size_t)u.K * u.nbl * u.S);
    std::vector<float> hDec((size_t)u.nfreq * u.nbl * u.S);
    CHECK_CUDA(cudaMemcpy(hTau.data(), dTau, hTau.size() * 8, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hDec.data(), dDecorr, hDec.size() * 4, cudaMemcpyDeviceToHost));
    {
        double eT = 0, eD = 0, maxTau = 0, meanDec = 0, minDec = 1;
        std::vector<double> dec(hDec.size(), 0.0);
        for (int k = 0; k < u.K; k++)
            for (int bl = 0; bl < u.nbl; bl++)
                for (int c = 0; c < u.S; c++) {
                    const int a = bls[bl].x, b = bls[bl].y;
                    const double tm = u.tmid(k);
                    const double dt = Dof(poly.data(), u.S, b, c + 1, tm) - Dof(poly.data(), u.S, a, c + 1, tm);
                    const Poly* pa = &poly[(size_t)a * (u.S + 1)], * pb = &poly[(size_t)b * (u.S + 1)];
                    const double dr = (prate(pb[c + 1], tm) - prate(pb[0], tm)) - (prate(pa[c + 1], tm) - prate(pa[0], tm));
                    eT = std::max(eT, fabs(hTau[((size_t)k * u.nbl + bl) * u.S + c] - dt));
                    maxTau = std::max(maxTau, fabs(dt));
                    for (int f = 0; f < u.nfreq; f++)
                        dec[((size_t)f * u.nbl + bl) * u.S + c] += decorrOf(dt, dr, u.width(k), u.flo(f), u.nchan, u.bwHz);
                }
        for (size_t i = 0; i < dec.size(); i++) {
            eD = std::max(eD, fabs(hDec[i] - dec[i]) / (u.nwin * u.twin));
            meanDec += dec[i] / (u.nwin * u.twin) / dec.size();
            minDec = std::min(minDec, dec[i] / (u.nwin * u.twin));
        }
        const bool ok = eT < 1e-15 && eD < 1e-5;
        pass &= ok;
        printf("delays: |Δτ| up to %.3g us (lag window +-%.3g us); mpifxcorr's decorrelation estimate: mean %.4f, worst %.4f\n",
               maxTau * 1e6, u.twin * 1e6, meanDec, minDec);
        printf("check: delays vs host: %.1e s, decorrelation %.1e  %s\n", eT, eD, ok ? "OK" : "FAIL");
    }

    // ---- check: each version against the double-precision shift of the same fine visibilities ----
    struct Pick { int f, bl, s, g; };
    std::vector<Pick> picks;
    unsigned long long r = seed * 7919 + 3;
    auto rnd = [&](int n) { r = splitmix(r); return (int)(r % n); };
    for (int i = 0; i < ncheck; i++) picks.push_back({rnd(u.nfreq), rnd(u.nbl), rnd(u.S), rnd(u.G)});
    std::vector<cd> ref;
    {
        std::vector<float2> row(u.A);
        for (const auto& pk : picks)
            for (int p = 0; p < 4; p++) {
                cd sum = 0;
                for (int k = 0; k < u.K; k++) {
                    CHECK_CUDA(cudaMemcpy(row.data(), fine + (size_t)k * s.visCount() + s.visIndex(pk.f, pk.bl, p, pk.g * u.A),
                                          u.A * sizeof(float2), cudaMemcpyDeviceToHost));
                    const double dt = hTau[((size_t)k * u.nbl + pk.bl) * u.S + pk.s];
                    for (int c = 0; c < u.A; c++) {
                        const double ph = dt * (u.flo(pk.f) + (pk.g * u.A + c) * u.df);
                        sum += cd(row[c].x, row[c].y) * std::polar(1.0, 2 * M_PI * (ph - std::floor(ph)));
                    }
                }
                ref.push_back(sum / (double)u.A);
            }
    }
    // the source's centre against what the simulated data must give: the coherent sum, exactly (no
    // noise), its loss only the delay's drift over each piece (the shift is at the piece's midpoint)
    struct SPick { int f, bl, g; };
    std::vector<SPick> spicks;
    std::vector<cd> sref;
    double coh = 0, cohPred = 0;
    if (noise == 0) {
        for (int i = 0; i < std::min(ncheck, 64); i++) spicks.push_back({rnd(u.nfreq), rnd(u.nbl), rnd(u.G)});
        for (const auto& pk : spicks) {
            const int a = bls[pk.bl].x, b = bls[pk.bl].y;
            cd sum = 0;
            for (int k = 0; k < u.K; k++) {
                const double dtm = Dof(poly.data(), u.S, b, src, u.tmid(k)) - Dof(poly.data(), u.S, a, src, u.tmid(k));
                for (int w = u.w0[k]; w < u.w0[k + 1]; w++) {
                    const double d = Dw[(size_t)a * u.nwin + w] - Dw[(size_t)b * u.nwin + w] + dtm;
                    for (int c = 0; c < u.A; c++) {
                        const double ph = d * (u.flo(pk.f) + (pk.g * u.A + c) * u.df);
                        sum += std::polar(1.0, 2 * M_PI * (ph - std::floor(ph)));
                    }
                }
            }
            sref.push_back(sum / (double)u.A);
            coh += std::abs(sum) / ((double)u.A * u.nwin) / spicks.size();
            cohPred += hDec[((size_t)pk.f * u.nbl + pk.bl) * u.S + src - 1] / (u.nwin * u.twin) / spicks.size();
        }
        printf("source at centre %d: coherence after the shift %.5f (mpifxcorr's estimate %.5f)\n", src, coh, cohPred);
    }
    std::vector<float2> hout(u.outCount());
    for (const auto& v : variants) {
        CHECK_CUDA(cudaMemsetAsync(out, 0xff, u.outCount() * sizeof(float2), st));
        v.run();
        CHECK_CUDA(cudaStreamSynchronize(st));
        CHECK_CUDA(cudaMemcpy(hout.data(), out, hout.size() * sizeof(float2), cudaMemcpyDeviceToHost));
        auto cmp = [&](auto&& at, const std::vector<cd>& rf, double tol, const char* what) {
            double d2 = 0, r2 = 0, dmax = 0;
            for (size_t i = 0; i < rf.size(); i++) {
                const float2 gv = at(i);
                const cd d = cd(gv.x, gv.y) - rf[i];
                d2 += std::norm(d); r2 += std::norm(rf[i]); dmax = std::max(dmax, std::abs(d));
            }
            const double rr = sqrt(r2 / rf.size()), rms = sqrt(d2 / rf.size()) / rr, mx = dmax / rr;
            const bool ok = std::isfinite(rms) && rms < tol && mx < 30 * tol;
            printf("check: %s, %s: rms %.2e, max %.2e (relative to rms; limits %.0e, %.0e)  %s\n", v.name.c_str(), what,
                   rms, mx, tol, 30 * tol, ok ? "OK" : "FAIL");
            return ok;
        };
        if (!picks.empty())
            pass &= cmp([&](size_t i) { const Pick& pk = picks[i / 4];
                                        return hout[u.outIndex(pk.f, pk.bl, pk.s, (int)(i % 4), pk.g)]; },
                        ref, v.tol, "vs the double shift");
        if (!sref.empty())
            pass &= cmp([&](size_t i) { const SPick& pk = spicks[i];
                                        return hout[u.outIndex(pk.f, pk.bl, src - 1, 0, pk.g)]; },
                        sref, std::max(v.tol, 3e-3), "source centre vs the exact sum");
    }
    printf("check: %s\n", pass ? "PASS" : "FAIL");

    // ---- timing ----
    const double xmacCmacs = s.cmacs(), shiftCmacs = (double)u.K * u.nfreq * u.nbl * 4 * u.nchan * u.S;
    printf("\ntiming per subint, median [min] of %d, ms (ms per ms of data):\n", reps);
    FILE* fc = nullptr;
    if (!csv.empty()) {
        FILE* t = fopen(csv.c_str(), "r");
        const bool header = !t;
        if (t) fclose(t);
        fc = fopen(csv.c_str(), "a");
        if (header) fprintf(fc, "gpu,nant,nfreq,nchan,nwin,pieces,avg,centres,what,ms,ms_per_ms\n");
    }
    auto line = [&](const std::string& key, const std::string& name, Timing t, double cmacs) {
        printf("  %-52s %8.3f [%8.3f]  %8.4f", name.c_str(), t.median, t.min, t.median / subintMs);
        if (cmacs > 0) printf("  %7.1f Gcmac/s", cmacs / t.median * 1e-6);
        printf("\n");
        if (fc) fprintf(fc, "\"%s\",%d,%d,%d,%d,%d,%d,%d,%s,%.5f,%.6f\n", prop.name, u.nant, u.nfreq, u.nchan, u.nwin, u.K,
                        u.A, u.S, key.c_str(), t.median, t.median / subintMs);
    };
    const Timing tx = timeIt(runXmac, warmup, reps, st);
    line("xmac", "xmac (staged tensor, per piece)", tx, xmacCmacs);
    line("delays", "delays (+ decorrelation estimate)", timeIt(runDelays, warmup, reps, st), 0);
    double best = 1e30;
    for (const auto& v : variants) {
        const Timing t = timeIt(v.run, warmup, std::min(reps, v.name[0] == 'n' && shiftCmacs > 1e11 ? 3 : reps), st);
        line(v.name.substr(0, v.name.find(' ')), "shift: " + v.name, t, shiftCmacs);
        best = std::min(best, t.median);
    }
    printf("  (Gcmac/s: the XMAC's complex multiply-accumulates, or the shift's: one per fine visibility per centre)\n");
    if (best < 1e30)
        printf("shift / XMAC: %.2f (the fastest shift); the work's ratio, centres / windows per piece: %.2f\n",
               best / tx.median, (double)u.S * u.K / u.nwin);
    if (fc) fclose(fc);
    xmacFreeMap(map);
    return pass ? 0 : 1;
}
