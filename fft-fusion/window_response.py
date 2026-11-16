#!/usr/bin/env python3
"""window_response.py - what a tapered, overlapped FFT window does to sensitivity and spectral response.

Usage:  python3 window_response.py [fft length, default 256]

For each window design (taper shape, taper length, overlap) prints:
  FFTs      FFTs per unit of data, relative to no overlap (the compute cost)
  SNR       continuum sensitivity relative to rectangular windows without overlap, at zero delay:
            sum(W) / sqrt(T sum(W^2)), where W(t) = sum over windows of w(t)^2 is the weight each
            sample ends up with in the band-integrated cross-power (white noise, all channels)
  W         the range of W(t): uneven weights waste sensitivity
  coh(tau)  the fraction of the correlation kept at a residual delay of tau samples (per window:
            sum w(t) w(t+tau) / sum w^2, the same with or without overlap); SNR x coh is the
            delay-dependent sensitivity
  leakage   the fraction of a channel's cross-power response |FT(w)|^2 more than 5 and 20 channels
            away (what a strong line or a lumpy bandpass leaks into distant channels)
"""
import sys
import numpy as np


def window(n, taper, shape):
    """Tapers of round(taper * n) samples each end, flat between: raised cosine (Tukey) or sine."""
    w = np.ones(n)
    taps = int(round(taper * n))
    if taps:
        x = (np.arange(taps) + 0.5) / taps
        t = 0.5 * (1 - np.cos(np.pi * x)) if shape == "tukey" else np.sin(0.5 * np.pi * x)
        w[:taps] = t
        w[n - taps:] = t[::-1]
    return w


def response(n, taper, overlap, shape, nwin=400):
    w = window(n, taper, shape)
    hop = n - int(round(overlap * n))
    weights = np.zeros((nwin - 1) * hop + n)
    for k in range(nwin):
        weights[k * hop:k * hop + n] += w ** 2
    core = weights[n:-n]                                  # away from the ends of the run
    snr = core.sum() / np.sqrt(core.size * (core ** 2).sum())
    coh = {tau: (w[:n - tau] * w[tau:]).sum() / (w ** 2).sum() for tau in (1, 4, 16, 64)}
    p = np.abs(np.fft.fft(w, 64 * n)) ** 2
    p /= p.sum()
    f = np.abs(np.fft.fftfreq(64 * n, 1.0 / n))          # in channels
    leak = [10 * np.log10(p[f > k].sum()) for k in (5, 20)]
    return n / hop, snr, core.min(), core.max(), coh, leak


DESIGNS = [
    ("rectangular, no overlap", 0.0, 0.0, "tukey"),
    ("Tukey 12.5% tapers, 25% overlap", 0.125, 0.25, "tukey"),
    ("Tukey 12.5% tapers, 12.5% overlap", 0.125, 0.125, "tukey"),
    ("sine 12.5% tapers, 12.5% overlap", 0.125, 0.125, "sine"),
    ("Tukey 25% tapers, 25% overlap", 0.25, 0.25, "tukey"),
    ("sine 25% tapers, 25% overlap", 0.25, 0.25, "sine"),
    ("Hann, 50% overlap", 0.5, 0.5, "tukey"),
]

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    print(f"FFT length {n}")
    print(f"{'design':36s} {'FFTs':>5s} {'SNR':>6s} {'W':>9s}   {'SNR x coh, tau = 1, 4, 16, 64 samples':37s}"
          f" {'leakage >5, >20 ch':>18s}")
    for name, taper, overlap, shape in DESIGNS:
        ffts, snr, wmin, wmax, coh, leak = response(n, taper, overlap, shape)
        sc = "  ".join(f"{snr * coh[t]:.4f}" for t in (1, 4, 16, 64))
        print(f"{name:36s} {ffts:5.3f} {snr:6.4f} {wmin:4.2f}-{wmax:4.2f}   {sc:37s} {leak[0]:7.1f} {leak[1]:7.1f} dB")
