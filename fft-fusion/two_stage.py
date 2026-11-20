#!/usr/bin/env python3
"""two_stage.py - what two-stage channelisation does to sensitivity and spectral leakage
(docs/two-stage.md).

Usage:  python3 two_stage.py          # the tables in docs/two-stage.md
        python3 two_stage.py offset   # leakage beyond 1, 2, 3, 5 coarse channels: two stages against one
        python3 two_stage.py ghost    # a line's largest spurious response against its place in its coarse channel

The model: complex samples at unit rate (the band is [0, 1) cycles a sample; real-sampled data is
the same with half the coarse channels).
  stage 1  a window h of P x L1 samples (P = 1: an FFT window, tapered for WOLA; P > 1: a polyphase
           filter bank), every H samples (oversampling O = L1 / H), and an L1-point DFT: coarse
           channel c, sample n:  y_c[n] = e^{-2 pi i c n H / L1} sum_t h[t] x[n H + t] e^{-2 pi i c t / L1}.
           The first factor ("derotation") puts each coarse channel's centre at zero frequency of its
           series; with O = 1 it is 1.
  stage 2  an N2-point DFT of N2 consecutive coarse samples (weights w2: rectangular, the windows
           back to back), keeping the K = N2 / O bins k = -K/2 .. K/2 - 1 that tile the coarse
           channel: fine channel (c, k) is centred at c / L1 + k / (N2 H), spacing 1 / (N2 H).
  1 stage  the reference: one FFT of L = N2 H points (the same channel spacing), rectangular
           (mpifxcorr now) or a Vorbis WOLA window (fusebench --wola).
A fine channel's response to a tone at f (relative to its coarse channel's centre) is
H1(f) x D2(f H - k / N2), H1 the DTFT of h and D2 that of w2: the coarse filter times the fine one.
D2 is periodic in f with period 1 / H, so whatever H1 passes at f + m / H (m != 0) lands in the same
fine channel: that is the aliasing, and O > 1 with the outer bins dropped keeps it to H1's skirts.

The sensitivity is computed exactly, from the linear map A from a circular stream of T input
samples to every kept output (fine channel x stage-2 window): for white noise at both stations, a
weak common signal, and the visibilities summed over every channel (continuum), the SNR is
tr(G) / ||G||_F with G = A A^H (the outputs are not independent: aliasing and oversampling
correlate them), relative to sqrt(T), which an ideal estimator from T samples reaches.  At a
residual delay tau each channel keeps |(A D_tau A^H)_jj| of its correlation (the phase slope across
the band is the fringe fit's), D_tau the delay.
"""
import numpy as np

# ---- windows ---------------------------------------------------------------------------------------

def vorbis(x):
    """The Vorbis taper, sin(pi/2 sin^2(pi x / 2)): power-complementary and smooth at both ends."""
    return np.sin(0.5 * np.pi * np.sin(0.5 * np.pi * x) ** 2)


def wola_window(L, taper):
    """L samples, Vorbis tapers of taper x L samples at each end, flat between (window_response.py)."""
    w = np.ones(L)
    taps = int(round(taper * L))
    if taps:
        t = vorbis((np.arange(taps) + 0.5) / taps)
        w[:taps] = t
        w[L - taps:] = t[::-1]
    return w


def pfb_window(L, P, cutoff, beta):
    """A polyphase filter bank's prototype: a sinc whose response falls to half at +-cutoff coarse
    channels (cutoff 0.5: the critically sampled design), times a Kaiser window, P x L samples."""
    n = np.arange(P * L) - (P * L - 1) / 2
    return np.sinc(2 * cutoff * n / L) * np.kaiser(P * L, beta)


def pfb_design(L, P, O):
    """A PFB prototype for oversampling O: the kept band is +-0.5 coarse channels and the nearest
    alias image starts at O - 0.5, so the cutoff goes midway (O / 2) and Kaiser's formula gives the
    stopband the transition (O - 1 coarse channels over P x L taps) allows: A = 8 + 2.285 x 2 pi x
    (O - 1) x P dB.  O = 1: the critically sampled design, cutoff 0.5 and beta 8."""
    if O <= 1:
        return pfb_window(L, P, 0.5, 8.0), 0.0
    A = 8 + 2.285 * 2 * np.pi * (O - 1) * P
    beta = 0.1102 * (A - 8.7) if A > 50 else 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21) if A > 21 else 0.0
    return pfb_window(L, P, O / 2, beta), A


# ---- designs -----------------------------------------------------------------------------------------

class Design:
    """One channelisation.  Two-stage: stage-1 window h (L1-point DFT, hop H), stage-2 N2-point DFT
    with weights w2, K bins kept.  One stage: L1 = 1 coarse channel of the whole band, h the FFT
    window of L points, hop H, no stage 2 (N2 = 1)."""

    def __init__(self, name, h, L1, H, N2=1, K=1, w2=None, hop2=None, short=None):
        self.name, self.h, self.L1, self.H, self.N2, self.K = name, np.asarray(h, float), L1, H, N2, K
        self.w2 = np.ones(N2) if w2 is None else np.asarray(w2, float)
        self.hop2 = hop2 or N2                 # stage-2 windows every hop2 coarse samples
        self.short = short or name
        self.two = N2 > 1

    @property
    def O(self):
        return self.L1 / self.H if self.two else len(self.h) / self.H

    def ks(self):
        return np.arange(-self.K // 2, self.K // 2) if self.two else np.zeros(1, int)

    def spacing(self):
        """The fine channels' spacing, cycles a sample."""
        return 1.0 / (self.N2 * self.H) if self.two else 1.0 / len(self.h)

    def power(self, oversample=32):
        """Power response of the kept fine channels of coarse channel 0 to a tone at f, on a grid of
        oversample points a fine channel over the whole band: f [-0.5, 0.5) cycles a sample, R[k][f]."""
        Nf = oversample * (self.N2 * self.H if self.two else len(self.h))
        P1 = np.abs(np.fft.fft(self.h, Nf)) ** 2                 # |DTFT of h|^2 at f = -j / Nf
        f = -np.fft.fftfreq(Nf)
        order = np.argsort(f)
        f, P1 = f[order], P1[order]
        if not self.two:
            return f, P1[None, :]
        R = np.empty((self.K, Nf))
        for i, k in enumerate(self.ks()):
            nu = f * self.H - k / self.N2                         # cycles a coarse sample
            if np.all(self.w2 == 1):                             # the Dirichlet kernel
                s = np.sin(np.pi * nu)
                with np.errstate(divide="ignore", invalid="ignore"):
                    D = np.where(np.abs(s) < 1e-12, self.N2 ** 2, (np.sin(np.pi * self.N2 * nu) / s) ** 2)
            else:
                n = np.arange(self.N2)
                D = np.abs(np.exp(2j * np.pi * np.outer(nu, n)) @ self.w2) ** 2
            R[i] = P1 * D
        return f, R

    def matrix(self, M):
        """The linear map from a circular stream of T samples to every kept output, T = M hop2 H (one
        stage: M windows of len(h) every H samples, T = M H): [output][sample]."""
        if not self.two:
            L = len(self.h)
            T = M * self.H
            rows = []
            for m in range(M):
                idx = (m * self.H + np.arange(L)) % T
                for j in range(L):
                    r = np.zeros(T, complex)
                    np.add.at(r, idx, self.h * np.exp(-2j * np.pi * j * np.arange(L) / L))
                    rows.append(r)
            return np.array(rows), T
        L1, H, N2 = self.L1, self.H, self.N2
        T = M * self.hop2 * H
        Lh = len(self.h)
        t = np.arange(Lh)
        rows = []
        for m in range(M):
            for c in range(L1):
                # stage 1 for coarse samples g = m N2 + n: phase e^{-2 pi i c (g H + t) / L1} on sample g H + t
                for k in self.ks():
                    r = np.zeros(T, complex)
                    for n in range(N2):
                        g = m * self.hop2 + n
                        idx = (g * H + t) % T
                        coef = (self.w2[n] * np.exp(-2j * np.pi * k * n / N2) * np.exp(-2j * np.pi * c * g * H / L1)
                                * self.h * np.exp(-2j * np.pi * c * t / L1))
                        np.add.at(r, idx, coef)
                    rows.append(r)
        return np.array(rows), T


def sensitivity(design, M, taus=(0.0,)):
    """Continuum SNR relative to the ideal, at residual delays tau (in samples), from the exact map."""
    A, T = design.matrix(M)
    G = A @ A.conj().T
    noise = np.sqrt((np.abs(G) ** 2).sum())
    f = np.fft.fftfreq(T)
    out = []
    for tau in taus:
        # A D A^H with D the (circular, fractional) delay: D = F^H diag(e^{-2 pi i f tau}) F
        AD = np.fft.ifft(np.fft.fft(A, axis=1) * np.exp(-2j * np.pi * f * tau)[None, :], axis=1)
        diag = np.einsum("ij,ij->i", AD, A.conj())
        out.append(np.abs(diag).sum() / noise / np.sqrt(T))
    return out


# ---- response metrics ----------------------------------------------------------------------------------

def metrics(design, oversample=32):
    """Per kept fine channel k: gain at its centre (dB, relative to the coarse centre's), and the
    fraction of its response power outside its own channel (more than 0.5 channels from its centre),
    more than 5 and 20 channels away (dB), and the power it takes from the alias images (f +- m / H,
    m != 0: other parts of the band, folded onto it) relative to its own band (dB)."""
    d = design.spacing()
    f, R = design.power(oversample)
    total = R.sum(axis=1)
    ks = design.ks()
    centre = int(np.argmin(np.abs(ks)))
    gain, outside, b5, b20, alias = [], [], [], [], []
    for i, k in enumerate(ks):
        fk = k * d
        dist = np.abs(f - fk)
        mine = dist <= 0.5 * d
        own = R[i][mine].sum()
        gain.append(R[i][np.argmin(dist)])
        outside.append(1 - own / total[i])
        b5.append(R[i][dist > 5 * d].sum() / total[i])
        b20.append(R[i][dist > 20 * d].sum() / total[i])
        if design.two:
            P = 1.0 / design.H                                   # the images' period
            m = np.round((f - fk) / P)                           # which image a frequency is near
            folded = (np.abs(f - fk - m * P) <= 0.5 * d) & (m != 0)
            alias.append(R[i][folded].sum() / own)
        else:
            alias.append(0.0)
    db = lambda x: 10 * np.log10(np.maximum(np.asarray(x, float), 1e-30))
    g = np.asarray(gain)
    return {"k": ks, "gain_db": db(g / g[centre]), "out_db": db(outside), "beyond5_db": db(b5),
            "beyond20_db": db(b20), "alias_db": db(alias)}


# ---- the designs compared --------------------------------------------------------------------------------

L1 = 64           # coarse channels (complex data; 32 coarse channels of a real-sampled band)


def designs(N2_base=64):
    """The designs at N2_base fine bins per coarse channel before trimming (the response metrics),
    each with the oversampling its stage 1 has."""
    rect = Design
    out = []
    # one stage, the same fine spacing as the 8/7 two-stage designs (L = 64 x 56 x ... scaled)
    out.append(rect("1 stage, rectangular (mpifxcorr now)", np.ones(L1 * 56), L1 * 56, L1 * 56, short="1 stage rect"))
    out.append(rect("1 stage, Vorbis WOLA 12.5%", wola_window(L1 * 56, 0.125), L1 * 56, L1 * 56 * 7 // 8,
                    short="1 stage WOLA"))
    out.append(rect("2 stage: rectangular stage 1, O = 1", np.ones(L1), L1, L1, N2_base, N2_base,
                    short="rect, O=1"))
    out.append(rect("2 stage: Vorbis WOLA 12.5% stage 1, O = 8/7", wola_window(L1, 0.125), L1, 56, N2_base,
                    N2_base * 7 // 8, short="WOLA 12.5%, O=8/7"))
    out.append(rect("2 stage: Vorbis WOLA 25% stage 1, O = 4/3", wola_window(L1, 0.25), L1, 48, N2_base,
                    N2_base * 3 // 4, short="WOLA 25%, O=4/3"))
    out.append(rect("2 stage: Vorbis WOLA 50% stage 1, O = 2", wola_window(L1, 0.5), L1, 32, N2_base,
                    N2_base // 2, short="WOLA 50%, O=2"))
    for P, O, H, K in ((8, 1, L1, N2_base), (8, 4 / 3, 48, N2_base * 3 // 4), (16, 4 / 3, 48, N2_base * 3 // 4),
                       (32, 8 / 7, 56, N2_base * 7 // 8), (4, 2, 32, N2_base // 2), (8, 2, 32, N2_base // 2)):
        h, A = pfb_design(L1, P, O)
        on = {1: "1", 4 / 3: "4/3", 8 / 7: "8/7", 2: "2"}[O]
        out.append(rect(f"2 stage: PFB {P} taps, O = {on}" + (f" (Kaiser, {A:.0f} dB)" if A else ""), h, L1, H,
                        N2_base, K, short=f"PFB {P}, O={on}"))
    h, A = pfb_design(L1, 16, 4 / 3)
    out.append(rect(f"2 stage: PFB 16 taps, O = 4/3, stage 2 Vorbis WOLA 12.5%", h, L1, 48, N2_base,
                    N2_base * 3 // 4, w2=wola_window(N2_base, 0.125), hop2=N2_base * 7 // 8,
                    short="PFB 16, O=4/3, WOLA 2"))
    return out


def model(design, N2):
    """The design with N2 fine bins a coarse channel (stage 2; one stage: the same window shape at
    N2 x 56 points), and the number of stage-2 windows M for the exact sensitivity: the circular
    stream at least twice the longest filter."""
    if not design.two:
        L = N2 * 56
        rect = "rect" in design.short
        h = np.ones(L) if rect else wola_window(L, 0.125)
        return Design(design.name, h, L, L if rect else L * 7 // 8, short=design.short), 4
    K = int(round(N2 * design.H / design.L1))
    tapered = not np.all(design.w2 == 1)
    w2 = wola_window(N2, 0.125) if tapered else None
    hop2 = N2 - int(round(0.125 * N2)) if tapered else N2
    M = max(2, int(np.ceil(2 * len(design.h) / (hop2 * design.H))))
    return Design(design.name, design.h, design.L1, design.H, N2, K, w2=w2, hop2=hop2, short=design.short), M


def delay_rate_coherence(nchan, rate, Nc=None):
    """Coherence kept, against a delay rate (s/s), when the fractional-sample (baseband) delay
    correction uses one delay per window (the window's middle), the windows nchan / B long: channel
    j's phase drifts by 2 pi j x rate across a window (f T = j, whatever the bandwidth), which keeps
    sinc(j x rate) of its amplitude.  Two stages (Nc coarse channels): the coarse stage corrects the
    delay at each coarse sample (its residual is negligible), so only the offset within the coarse
    channel drifts, |j - j_c| <= nchan / (2 Nc).  Returns (band top, band average)."""
    j = np.arange(nchan)
    if Nc:
        K = nchan / Nc
        j = (j + K / 2) % K - K / 2                    # offset from the coarse channel's centre
    c = np.abs(np.sinc(j * rate))
    return c[-1] if not Nc else c.min(), c.mean()


def main():
    print(__doc__.split("\n\n")[0])
    print()
    print("Responses at 64 coarse channels and 64 fine bins a coarse channel before trimming (worst = the fine")
    print("channels at the coarse channels' edges); sensitivity from the exact map at N2 = 32 (K = 32 / O fine")
    print("channels a coarse channel), relative to the ideal, at residual delays of 0, 1/64 and 1/16 of the")
    print("XMAC's window (the stage-2 window); for the PFBs the zero-delay loss falls as 1 / K: loss x K.")
    print()
    print(f"{'design':58s} {'FFTs':>5s} {'SNR 0':>6s} {'W/64':>6s} {'W/16':>6s} {'loss x K':>8s} {'edge gain':>9s} "
          f"{'alias worst / median':>21s} {'beyond 5 (median)':>17s} {'beyond 20 (median)':>18s}")
    for d in designs():
        m = metrics(d)
        sd, M = model(d, 32)
        W = sd.N2 * sd.H if sd.two else len(sd.h)
        s0, s1, s2 = sensitivity(sd, M, (0.0, W / 64, W / 16))
        lk = (1 - s0) * sd.K if (sd.two and "PFB" in d.short) else float("nan")
        al = f"{np.max(m['alias_db']):7.1f} / {np.median(m['alias_db']):6.1f}" if d.two else f"{'-':>16s}"
        print(f"{d.name:58s} {d.O:5.3f} {s0:6.3f} {s1:6.3f} {s2:6.3f} {lk:8.3f} {m['gain_db'][0]:7.1f}dB {al:>21s} "
              f"{np.median(m['beyond5_db']):14.1f} dB {np.median(m['beyond20_db']):15.1f} dB", flush=True)
    print()
    print("Coherence against a delay rate of 3e-6 s/s (an Earth-rotation maximum), one delay per window:")
    print(f"{'channels':>9s} {'1 stage: top / mean':>21s} " + " ".join(f"{'2 stage, Nc ' + str(n) + ': worst / mean':>27s}" for n in (16, 32, 64)))
    for nchan in (4096, 16384, 65536, 262144, 1048576):
        one = delay_rate_coherence(nchan, 3e-6)
        two = [delay_rate_coherence(nchan, 3e-6, n) for n in (16, 32, 64)]
        print(f"{nchan:9d} {one[0]:10.4f} / {one[1]:.4f} " + " ".join(f"{t[0]:16.4f} / {t[1]:.4f}" for t in two))


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    main()


# ---- leakage against frequency offset: two stages against fewer channels in one stage --------------------

def offset_response(design, k, x):
    """Fine channel k's power response (two stages; one stage: its channel) to a tone x coarse
    channels from the channel's centre, relative to its peak.  x in coarse channels (1 / L1 cycles a
    sample, L1 = 64)."""
    t = np.arange(len(design.h)) - (len(design.h) - 1) / 2
    d = design.spacing()
    f = np.asarray(x, float) / L1 + (k * d if design.two else 0.0)   # relative to the coarse centre
    P1 = np.empty_like(f)
    for i in range(0, len(f), 4096):                                  # the stage-1 window's DTFT
        P1[i:i + 4096] = np.abs(np.exp(2j * np.pi * np.outer(f[i:i + 4096], t)) @ design.h) ** 2
    if not design.two:
        return P1 / (np.abs(design.h.sum()) ** 2)
    nu = f * design.H - k / design.N2
    s = np.sin(np.pi * nu)
    with np.errstate(divide="ignore", invalid="ignore"):
        D = np.where(np.abs(s) < 1e-12, design.N2 ** 2, (np.sin(np.pi * design.N2 * nu) / s) ** 2)
    R = P1 * D
    fk = np.array([k * d])
    peak = (np.abs(np.exp(2j * np.pi * np.outer(fk, t)) @ design.h) ** 2)[0] * design.N2 ** 2
    return R / peak


def offset_designs():
    """The comparison: one stage of 64 channels (the stage-1 window alone: the channel count one
    would have used before), one stage at the fine resolution (64 x 56 channels), and two stages
    (64 coarse channels, 56 fine each), WOLA and PFB first stages; Vorbis 12.5% WOLA throughout."""
    coarse = Design("one stage, 64 channels, Vorbis WOLA 12.5%", wola_window(L1, 0.125), L1, 56,
                    short="1 stage, coarse")
    fine = Design("one stage, 64 x 56 channels, Vorbis WOLA 12.5%", wola_window(L1 * 56, 0.125), L1 * 56,
                  L1 * 49, short="1 stage, fine")
    wola = Design("two stage: WOLA 12.5% (O = 8/7), rectangular stage 2", wola_window(L1, 0.125), L1, 56, 64, 56,
                  short="2 stage WOLA")
    pfb = Design("two stage: PFB 32 taps (O = 8/7), rectangular stage 2", pfb_design(L1, 32, 8 / 7)[0], L1, 56,
                 64, 56, short="2 stage PFB")
    return coarse, fine, wola, pfb


def offset_table():
    """Fraction of a channel's response power beyond 1, 2, 3 and 5 coarse channels from its centre (dB)."""
    x = np.arange(-40, 40, 1 / 56 / 16)                               # +-40 coarse channels, 16 points a fine channel
    coarse, fine, wola, pfb = offset_designs()
    rows = [(coarse.name, coarse, 0), (fine.name, fine, 0),
            (wola.name + ", fine channel at the coarse centre", wola, 0),
            (wola.name + ", fine channel at the coarse edge", wola, -28),
            (pfb.name + ", fine channel at the coarse centre", pfb, 0),
            (pfb.name + ", fine channel at the coarse edge", pfb, -28)]
    out = []
    for name, d, k in rows:
        R = offset_response(d, k, x)
        tot = R.sum()
        out.append((name, [10 * np.log10(R[np.abs(x) > b].sum() / tot) for b in (1, 2, 3, 5)]))
    return out


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "offset":
    print("Fraction of a channel's response power beyond 1 / 2 / 3 / 5 coarse channels from its centre:")
    for name, v in offset_table():
        print(f"  {name:90s} " + " / ".join(f"{x:6.1f}" for x in v) + " dB")


def ghost_table(positions=(0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5)):
    """A narrow line at position p (coarse channels from a coarse centre).  Two stages: the largest
    response in any fine channel more than half a coarse channel from the line (dB, of the line's own
    peak), where it is (offset from the line), and which coarse channel it is in.  One stage at the
    coarse resolution: the line's response in that same coarse channel (dB, of the line's own channel)."""
    coarse, fine, wola, pfb = offset_designs()
    t1 = np.arange(len(coarse.h)) - (len(coarse.h) - 1) / 2
    one = lambda p, c: np.abs(np.exp(2j * np.pi * (p - c) / L1 * t1) @ coarse.h) ** 2
    out = []
    for p in positions:
        row = [p]
        for d in (wola, pfb):
            t = np.arange(len(d.h)) - (len(d.h) - 1) / 2
            n = np.arange(d.N2) - (d.N2 - 1) / 2
            xs, cs, ps = [], [], []
            for c in range(-6, 7):
                fr = (p - c) / L1
                P1 = np.abs(np.exp(2j * np.pi * fr * t) @ d.h) ** 2
                for k in d.ks():
                    D = np.abs(np.exp(2j * np.pi * (fr * d.H - k / d.N2) * n) @ d.w2) ** 2
                    xs.append(c + k / d.K)
                    cs.append(c)
                    ps.append(P1 * D)
            xs, cs, ps = np.array(xs), np.array(cs), np.array(ps) / max(ps)
            i = np.argmax(np.where(np.abs(xs - p) > 0.5, ps, 0))
            own = int(np.round(p)) if abs(p) < 0.5 else 0
            row += [10 * np.log10(ps[i]), xs[i] - p, cs[i], 10 * np.log10(one(p, cs[i]) / one(p, 0))]
        out.append(row)
    return out


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "ghost":
    print("A narrow line p coarse channels from a coarse centre.  Two stages: the largest response more than half")
    print("a coarse channel from it (dB of the line's peak), its offset from the line, the coarse channel it is in;")
    print("one stage at 64 channels: the line's response in that same coarse channel:")
    for r in ghost_table():
        p = r[0]
        (w, wx, wc, wo), (f, fx, fc, fo) = r[1:5], r[5:9]
        print(f"  p = {p:4.2f}:  WOLA stage 1 {w:6.1f} dB at {wx:+5.2f} (coarse {wc:+d}; one stage there {wo:6.1f} dB)   "
              f"PFB stage 1 {f:6.1f} dB at {fx:+5.2f} (coarse {fc:+d}; one stage there {fo:6.1f} dB)")
