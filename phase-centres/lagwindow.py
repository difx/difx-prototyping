#!/usr/bin/env python3
"""lagwindow.py - mpifxcorr's lag-window term of its phase-centre decorrelation estimate, checked
by simulation (docs/results.md, "The lag-window term").

mpifxcorr (Core::uvshiftAndAverageBaselineFreq) estimates the loss of a centre at differential delay
dtau from the pointing centre as 1 - |dtau| / (nchan / bw): the stations' FFT windows are aligned
for the pointing centre, so for a source at dtau the two windows share only N - |d| of their N
samples (d = dtau in samples).  This simulates it in the time domain: real white noise, station 2
the same signal delayed by d samples (a fractional delay, applied in the frequency domain on the
whole series), both cut into the same N-sample windows (rectangular, no overlap, as mpifxcorr's
FFTs), N/2 channels each, cross-multiplied, rotated by the shift to the source (e^{-2 pi i d k / N}
for channel k, undoing the e^{+2 pi i d k / N} that X conj(Y) carries), averaged over channels and
windows.  The coherence is |<V>| / sqrt(<|X|^2><|Y|^2>).

Usage:  python3 lagwindow.py [N] [windows]   (defaults 1024, 4000)
"""
import sys
import numpy as np

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
W = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
rng = np.random.default_rng(1)
L = N * W
x = rng.standard_normal(L)
X = np.fft.rfft(x)
fr = np.fft.rfftfreq(L)                        # cycles per sample

print("N = %d samples a window (%d channels), %d windows" % (N, N // 2, W))
print("%10s %10s %12s %12s %10s" % ("d/N", "d", "coherence", "1 - |d|/N", "differ"))
for frac in [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
    d = frac * N + 0.37 * (frac > 0)           # a fractional part too
    y = np.fft.irfft(X * np.exp(-2j * np.pi * fr * d), L)    # y(t) = x(t - d)
    xs = np.fft.rfft(x.reshape(W, N), axis=1)[:, : N // 2]
    ys = np.fft.rfft(y.reshape(W, N), axis=1)[:, : N // 2]
    k = np.arange(N // 2)
    v = (xs * np.conj(ys)) * np.exp(-2j * np.pi * d * k / N)    # X conj(Y) carries e^{+2 pi i d k / N}
    coh = abs(v.mean()) / np.sqrt((abs(xs) ** 2).mean() * (abs(ys) ** 2).mean())
    est = 1 - abs(d) / N
    print("%10.3f %10.2f %12.5f %12.5f %10.5f" % (frac, d, coh, est, coh - est))
