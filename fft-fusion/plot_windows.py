#!/usr/bin/env python3
"""plot_windows.py - pictures of the tapered-window choices (see docs/windows.md).

Usage:  python3 plot_windows.py [output directory, default plots/]

Every design here overlaps neighbouring windows by exactly the taper length (12.5% taper: hop
0.875 FFT lengths). Three taper shapes (window_response.SHAPES): Tukey (raised cosine), sine,
and Vorbis (sin(pi/2 * raised cosine)). Writes
  tapers.png       the three taper shapes, and the weight a sample gets where two tapers overlap
  windows.png      consecutive windows and the per-sample weight, each shape at 12.5% and 25%
  tradeoffs.png    sensitivity at zero delay, and leakage, against taper length
  response.png     a channel's cross-power response (sidelobe envelope), at 12.5% and 25%
  sensitivity.png  continuum sensitivity against residual delay, at 12.5% and 25%
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.patches
from window_response import SHAPES, window, response

# reference palette (dataviz skill), light mode; one colour per taper shape in every figure,
# rectangular windows (mpifxcorr now) as the neutral reference
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
WEIGHT = "#d9d8d2"
COLOUR = {"tukey": "#2a78d6", "sine": "#eb6834", "vorbis": "#1baf7a"}
NAME = {"tukey": "Tukey (raised cosine)", "sine": "sine", "vorbis": "Vorbis (sin of raised cosine)"}
SHORT = {"tukey": "Tukey", "sine": "Sine", "vorbis": "Vorbis"}
REF = dict(color=INK2, linewidth=1.5, linestyle=(0, (4, 2)))
N = 256


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def legend(ax_or_fig, **kw):
    leg = ax_or_fig.legend(frameon=False, fontsize=9, **kw)
    for t in leg.get_texts():
        t.set_color(INK)
    return leg


def finish(fig, path, title, note=None):
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    if note:
        fig.text(0.01, 0.008, note, fontsize=8.5, color=INK2)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def tapers_png(path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.9), facecolor=SURFACE)
    x = np.linspace(0, 1, 401)
    for shape in SHAPES:
        rise = SHAPES[shape](x)
        a1.plot(x, rise, color=COLOUR[shape], linewidth=2, label=NAME[shape])
        a2.plot(x, rise ** 2 + rise[::-1] ** 2, color=COLOUR[shape], linewidth=2, label=NAME[shape])
    for ax in (a1, a2):
        style(ax)
        ax.set_xlim(0, 1)
        ax.set_xlabel("position across the taper (overlap)", color=INK2)
    a1.set_ylim(0, 1.05)
    a1.set_title("taper shape w (rising edge)", loc="left", fontsize=10, color=INK)
    a2.set_ylim(0, 1.1)
    a2.set_title("sample weight where two tapers overlap: w$^2$ + w'$^2$", loc="left", fontsize=10, color=INK)
    legend(a1, loc="upper left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    finish(fig, path, "Three taper shapes",
           "Tukey tapers add to one in amplitude, so the power weight dips to 0.5 mid-overlap; sine and Vorbis "
           "tapers add to one in power.")


def windows_png(path):
    fig, axes = plt.subplots(3, 2, figsize=(12, 8.4), sharex=True, sharey=True, facecolor=SURFACE)
    for row, shape in enumerate(SHAPES):
        for col, taper in enumerate((0.125, 0.25)):
            ax = axes[row, col]
            style(ax)
            w = window(N, taper, shape)
            hop = N - int(round(taper * N))
            nwin = 5
            total = (nwin - 1) * hop + N
            weight = np.zeros(total)
            for k in range(nwin):
                t = (k * hop + np.arange(N)) / N
                ax.plot(t, w, color=COLOUR[shape], linewidth=2, alpha=1.0 if k % 2 == 0 else 0.55, zorder=2)
                weight[k * hop:k * hop + N] += w ** 2
            x = np.arange(total) / N
            inner = (x >= 1) & (x <= (total - N) / N)
            ax.fill_between(x[inner], 0, weight[inner], color=WEIGHT, linewidth=0, zorder=0)
            ffts, snr, wmin, wmax, _, _ = response(N, taper, taper, shape)
            ax.set_title(f"{SHORT[shape]}, {100 * taper:g}% tapers:  {ffts:.3f}x FFTs,  weight {wmin:.2f}-{wmax:.2f},"
                         f"  sensitivity {snr:.3f}", loc="left", fontsize=9.5, color=INK)
            ax.set_ylim(0, 1.25)
            ax.set_yticks([0, 0.5, 1])
    for ax in axes[-1]:
        ax.set_xlabel("time, in FFT lengths", color=INK2)
    handles = [matplotlib.lines.Line2D([], [], color=INK2, linewidth=2, label="window k, k+2, ..."),
               matplotlib.lines.Line2D([], [], color=INK2, alpha=0.55, linewidth=2, label="window k+1, k+3, ..."),
               matplotlib.patches.Patch(color=WEIGHT, label="sample weight: sum over windows of w$^2$")]
    legend(fig, handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=3)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    finish(fig, path, f"Tapered FFT windows, neighbours overlapping over their tapers ({N}-point FFT)",
           "Sensitivity: continuum, relative to rectangular windows without overlap (white noise, all channels, "
           "zero residual delay).")


def tradeoffs_png(path):
    n = 1024
    tapers = np.round(np.arange(0.01, 0.3001, 0.005), 4)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), facecolor=SURFACE)
    for shape in SHAPES:
        snr, l5, l20 = [], [], []
        for t in tapers:
            _, s, _, _, _, leak = response(n, t, t, shape, nwin=60)
            snr.append(s)
            l5.append(leak[0])
            l20.append(leak[1])
        for ax, y in zip(axes, (snr, l5, l20)):
            ax.plot(100 * tapers, y, color=COLOUR[shape], linewidth=2, label=NAME[shape])
    # rectangular reference
    _, s0, _, _, _, leak0 = response(n, 0.0, 0.0, "tukey", nwin=60)
    for ax, y in zip(axes, (s0, leak0[0], leak0[1])):
        ax.axhline(y, **REF, label="rectangular, no overlap")
    titles = ("continuum sensitivity, zero delay", "leakage beyond 5 channels, dB", "leakage beyond 20 channels, dB")
    for ax, title in zip(axes, titles):
        style(ax)
        ax.set_xlim(0, 30)
        for v in (12.5, 25):
            ax.axvline(v, color=GRID, linewidth=1.5, zorder=0)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("taper (= overlap) at each end, % of the FFT length", color=INK2)
    axes[0].set_ylim(0.97, 1.003)
    legend(axes[0], loc="lower left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    finish(fig, path, "What the taper length buys: leakage falls, FFT count rises as 1 / (1 - taper)",
           "FFTs per unit of data: 1.067x at 6.25%, 1.143x at 12.5%, 1.231x at 18.75%, 1.333x at 25%. "
           "Leakage: share of a channel's cross-power response beyond the given distance.")


def response_png(path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, facecolor=SURFACE)
    pad = 128
    f = np.fft.fftfreq(pad * N, 1.0 / N)
    keep = (f >= 0) & (f <= 41)
    half = pad // 2

    def envelope(w):
        p = np.abs(np.fft.fft(w, pad * N)) ** 2
        p = (p / p[0])[keep]
        return 10 * np.log10(np.maximum([p[max(0, i - half):i + half + 1].max() for i in range(p.size)], 1e-12))

    rect = envelope(np.ones(N))
    for ax, taper in zip(axes, (0.125, 0.25)):
        style(ax)
        ax.plot(f[keep], rect, **REF, label="rectangular")
        for shape in SHAPES:
            ax.plot(f[keep], envelope(window(N, taper, shape)), color=COLOUR[shape], linewidth=2, label=NAME[shape])
        ax.set_xlim(0, 40)
        ax.set_ylim(-100, 3)
        ax.set_title(f"{100 * taper:g}% tapers", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("frequency offset from the channel centre, channels", color=INK2)
    axes[0].set_ylabel("cross-power response (sidelobe envelope), dB", color=INK2)
    legend(axes[1], loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    finish(fig, path, "A channel's response to a narrow line: the taper sets it; overlap does not")


def sensitivity_png(path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, facecolor=SURFACE)
    taus = np.arange(0, 65)

    def curve(taper, shape):
        w = window(N, taper, shape)
        _, snr, _, _, _, _ = response(N, taper, taper, shape)
        return snr * np.array([(w[:N - t] * w[t:]).sum() / (w ** 2).sum() for t in taus])

    for ax, taper in zip(axes, (0.125, 0.25)):
        style(ax)
        ax.plot(taus, curve(0.0, "tukey"), **REF, label="rectangular, no overlap")
        for shape in SHAPES:
            ax.plot(taus, curve(taper, shape), color=COLOUR[shape], linewidth=2, label=NAME[shape])
        ax.set_xlim(0, 64)
        ax.set_title(f"{100 * taper:g}% tapers", loc="left", fontsize=10, color=INK)
        ax.set_xlabel(f"residual delay, samples ({N}-point FFT)", color=INK2)
    axes[0].set_ylabel("continuum sensitivity (rectangular at zero delay = 1)", color=INK2)
    legend(axes[0], loc="lower left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    finish(fig, path, "Continuum sensitivity against residual delay")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "plots"
    os.makedirs(out, exist_ok=True)
    names = []
    for name, fn in (("tapers", tapers_png), ("windows", windows_png), ("tradeoffs", tradeoffs_png),
                     ("response", response_png), ("sensitivity", sensitivity_png)):
        fn(os.path.join(out, name + ".png"))
        names.append(os.path.join(out, name + ".png"))
    print("wrote", ", ".join(names))
