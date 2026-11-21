#!/usr/bin/env python3
"""plot_two_stage.py - pictures of two-stage channelisation's sensitivity and leakage
(docs/two-stage.md; the numbers: two_stage.py).

Usage:  python3 plot_two_stage.py [output directory, default plots/]

Writes
  two-stage-offset.png       leakage against frequency offset, +-3 coarse channels: two stages (WOLA or
                             PFB stage 1) against one stage at the coarse and at the fine resolution
  two-stage-coarse.png       the stage-1 filter's response: WOLA windows and polyphase filter banks
  two-stage-fine.png         across one coarse channel: each fine channel's gain, and what it takes
                             from the alias images
  two-stage-line.png         a narrow line near a coarse channel's edge, as the fine channels see it
  two-stage-sensitivity.png  continuum sensitivity against residual delay
  two-stage-delayrate.png    coherence against the channel count at a high delay rate (one delay per
                             window for the fractional-sample correction)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import two_stage as T
from plot_windows import SURFACE, INK, INK2, GRID, style, legend


def finish(fig, path, title, note=None, right=None, bottom=0.17):
    """Title, a note of up to two lines below the axes (wrapped), and the file."""
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.subplots_adjust(bottom=bottom, top=0.88, **({"right": right} if right else {}))
    if note:
        import textwrap
        fig.text(0.01, 0.012, "\n".join(textwrap.wrap(note, 150)), fontsize=8.5, color=INK2, va="bottom")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)

# the reference palette's categorical slots, in order (dataviz skill); one stage (the reference) neutral
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
REF = dict(color=INK2, linewidth=1.5, linestyle=(0, (4, 2)))
L1 = T.L1


def pick(short):
    return next(d for d in T.designs() if d.short == short)


def coarse_png(path):
    fig, ax = plt.subplots(figsize=(11, 4.8), facecolor=SURFACE)
    style(ax)
    x = np.linspace(0, 3, 3001)                                   # coarse channels from the centre
    for i, (short, label) in enumerate([("WOLA 12.5%, O=8/7", "WOLA, Vorbis 12.5% taper"),
                                        ("WOLA 50%, O=2", "WOLA, Vorbis 50% taper"),
                                        ("PFB 8, O=1", "PFB 8 taps, critically sampled (O = 1)"),
                                        ("PFB 16, O=4/3", "PFB 16 taps, for O = 4/3"),
                                        ("PFB 8, O=2", "PFB 8 taps, for O = 2")]):
        d = pick(short)
        t = np.arange(len(d.h)) - (len(d.h) - 1) / 2
        H = np.abs(np.exp(2j * np.pi * np.outer(x / L1, t)) @ d.h) ** 2
        ax.plot(x, 10 * np.log10(np.maximum(H / H[0], 1e-14)), color=SLOT[i], linewidth=2, label=label)
    rect = np.abs(np.sinc(x)) ** 2
    ax.plot(x, 10 * np.log10(np.maximum(rect, 1e-14)), label="rectangular (no taper)", **REF)
    ax.axvspan(0, 0.5, color=GRID, alpha=0.6, linewidth=0)
    ax.text(0.25, -112, "kept", ha="center", fontsize=9, color=INK2)
    for O, lab in ((8 / 7, "O = 8/7"), (4 / 3, "4/3"), (2, "2")):
        ax.axvline(O - 0.5, color=INK2, linewidth=0.8, linestyle=":")
        ax.text(O - 0.5 + 0.03, -5, lab, fontsize=8.5, color=INK2, rotation=90, va="top")
    ax.set_xlim(0, 3)
    ax.set_ylim(-120, 3)
    ax.set_xlabel("frequency from the coarse channel's centre (coarse channels)", color=INK, fontsize=10)
    ax.set_ylabel("power response (dB)", color=INK, fontsize=10)
    legend(ax, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    finish(fig, path, "Stage 1: a coarse channel's response", right=0.7, note=
           "Shaded: the half of the coarse channel kept (each side).  Dotted: where the nearest alias image "
           "starts, O - 0.5 coarse channels out, for each oversampling O: the filter must be down by then.")


def fine_png(path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    cases = [("WOLA 12.5%, O=8/7", "WOLA 12.5%, O = 8/7"), ("WOLA 50%, O=2", "WOLA 50%, O = 2"),
             ("PFB 8, O=1", "PFB 8 taps, O = 1"), ("PFB 16, O=4/3", "PFB 16 taps, O = 4/3"),
             ("PFB 8, O=2", "PFB 8 taps, O = 2")]
    for i, (short, label) in enumerate(cases):
        d = pick(short)
        m = T.metrics(d)
        pos = m["k"] / d.K                                            # fine channel centre, coarse channels
        axes[0].plot(pos, m["gain_db"], color=SLOT[i], linewidth=2, label=label)
        axes[1].plot(pos, m["alias_db"], color=SLOT[i], linewidth=2, label=label)
    for ax, yl, lim in ((axes[0], "gain at the channel's centre (dB)", (-7, 0.5)),
                        (axes[1], "power from the alias images (dB, of its own)", (-130, 5))):
        style(ax)
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(*lim)
        ax.set_xlabel("fine channel's place in its coarse channel (coarse channels)", color=INK, fontsize=10)
        ax.set_ylabel(yl, color=INK, fontsize=10)
    legend(axes[0], loc="lower center", ncol=1)
    finish(fig, path, "Across one coarse channel: each fine channel's gain, and the power aliased into it", note=
           "Rectangular stage-2 windows; 64 fine bins a coarse channel before trimming.  At the coarse channel's "
           "edges the stage-1 filter's skirts fold in from beyond the next image.")


def line_png(path):
    """A narrow line at 0.4 coarse channels from a coarse channel's centre: power in every fine channel
    within +-1.5 coarse channels, as fine channel offset from the line."""
    fig, ax = plt.subplots(figsize=(9, 4.4), facecolor=SURFACE)
    style(ax)
    f0 = 0.4 / L1
    cases = [("WOLA 12.5%, O=8/7", "two stage, WOLA 12.5%, O = 8/7"), ("WOLA 50%, O=2", "two stage, WOLA 50%, O = 2"),
             ("PFB 16, O=4/3", "two stage, PFB 16 taps, O = 4/3")]
    for i, (short, label) in enumerate(cases):
        d = pick(short)
        t = np.arange(len(d.h)) - (len(d.h) - 1) / 2
        n = np.arange(d.N2) - (d.N2 - 1) / 2
        xs, ps = [], []
        for c in range(-2, 3):
            fr = f0 - c / L1                                         # relative to coarse channel c's centre
            P1 = np.abs(np.exp(2j * np.pi * fr * t) @ d.h) ** 2
            for k in d.ks():
                nu = fr * d.H - k / d.N2
                D = np.abs(np.exp(2j * np.pi * nu * n) @ d.w2) ** 2
                xs.append(c * d.K + k)
                ps.append(P1 * D)
        xs, ps = np.array(xs), np.array(ps)
        order = np.argsort(xs)
        x = (xs[order] - 0.4 * d.K) / d.K                           # from the line, coarse channels
        ax.plot(x, 10 * np.log10(np.maximum(ps[order] / ps.max(), 1e-14)), color=SLOT[i], linewidth=1.8,
                label=label, marker="o", markersize=2.5)
    # one stage, rectangular, at the PFB's fine spacing (48 channels a coarse channel): the line 0.4 x 48
    # = 19.2 channels from a coarse centre, so 0.2 of a channel from the nearest
    K = 48
    j = np.arange(-int(1.5 * K), int(1.5 * K) + 1)
    p = np.sinc(j - 0.2) ** 2
    ax.plot(j / K, 10 * np.log10(np.maximum(p / p.max(), 1e-14)), label="one stage, rectangular", **REF)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-100, 3)
    ax.set_xlabel("fine channel, from the line (coarse channels)", color=INK, fontsize=10)
    ax.set_ylabel("power (dB, of the peak)", color=INK, fontsize=10)
    legend(ax, loc="lower left")
    finish(fig, path, "A narrow line 0.4 coarse channels from a coarse centre: the fine channels' response", note=
           "With a WOLA first stage the line also appears O coarse channels away (an alias image: blue, at +-8/7); "
           "with a polyphase filter bank it does not.  The steps are the coarse channels' edges.")


def sensitivity_png(path):
    fig, ax = plt.subplots(figsize=(11, 4.8), facecolor=SURFACE)
    style(ax)
    fr = np.linspace(0, 1 / 8, 33)
    cases = [("1 stage rect", None, REF), ("1 stage WOLA", None, dict(color=INK2, linewidth=1.5, linestyle=":")),
             ("WOLA 12.5%, O=8/7", SLOT[0], None), ("WOLA 50%, O=2", SLOT[1], None), ("PFB 8, O=1", SLOT[2], None),
             ("PFB 16, O=4/3", SLOT[3], None), ("PFB 16, O=4/3, WOLA 2", SLOT[4], None)]
    labels = {"1 stage rect": "one stage, rectangular (mpifxcorr now)", "1 stage WOLA": "one stage, Vorbis WOLA 12.5%",
              "WOLA 12.5%, O=8/7": "two stage, WOLA 12.5%, O = 8/7", "WOLA 50%, O=2": "two stage, WOLA 50%, O = 2",
              "PFB 8, O=1": "two stage, PFB 8 taps, O = 1", "PFB 16, O=4/3": "two stage, PFB 16 taps, O = 4/3",
              "PFB 16, O=4/3, WOLA 2": "the same, stage 2 Vorbis WOLA 12.5%"}
    for short, col, st in cases:
        d, M = T.model(pick(short), 32)
        W = d.N2 * d.H if d.two else len(d.h)
        s = T.sensitivity(d, M, fr * W)
        if st:
            ax.plot(fr, s, label=labels[short], **st)
        else:
            ax.plot(fr, s, color=col, linewidth=2, label=labels[short])
        print(f"  {short}: {s[0]:.3f} .. {s[-1]:.3f}", flush=True)
    ax.set_xlim(0, 1 / 8)
    ax.set_ylim(0.55, 1.01)
    ax.set_xlabel("residual delay (fraction of the XMAC's window)", color=INK, fontsize=10)
    ax.set_ylabel("continuum sensitivity (of ideal)", color=INK, fontsize=10)
    legend(ax, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    finish(fig, path, "Continuum sensitivity against residual delay", right=0.68, note=
           "Exact, for white noise, summed over every channel; 64 coarse channels, 32 fine bins each before "
           "trimming (the PFBs' small zero-delay loss falls as 1 / fine channels a coarse channel).  Aliased images "
           "rotate with the delay and come back into phase at multiples of the stage-1 hop.")


def delayrate_png(path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    n = 2 ** np.arange(9, 21)
    for ax, lim in ((axes[0], (0, 1.02)), (axes[1], (0.98, 1.0005))):
        style(ax)
        for i, (r, lab) in enumerate(((3e-6, "Earth rotation, 3 \u00b5s/s"), (3e-5, "spacecraft, 30 \u00b5s/s"),
                                      (1e-4, "fast spacecraft, 100 \u00b5s/s"))):
            one = np.array([T.delay_rate_coherence(int(x), r) for x in n])
            ax.plot(n, one[:, 0], color=SLOT[i], linewidth=2, label=f"one stage, {lab}: top channel")
        for L1, st in ((16, (0, (4, 2))), (64, ":")):
            ax.axhline(1 - T.two_stage_fractional_loss(L1)[0], color=INK2, linewidth=1.5, linestyle=st,
                       label=f"two stages, {L1}-point stage 1: worst channel, any rate")
        ax.set_xscale("log", base=2)
        ax.set_ylim(*lim)
        ax.set_xlabel("channels per subband (any bandwidth)", color=INK, fontsize=10)
        ax.set_ylabel("coherence kept", color=INK, fontsize=10)
    axes[1].set_title("the top 2%", fontsize=10, color=INK2)
    legend(axes[0], loc="lower left")
    finish(fig, path, "The fractional-sample correction against the delay rate", note=
           "One stage: one delay per window (its middle), channel j's phase drifts by 2\u03c0 j \u00d7 rate across a "
           "window, whatever the bandwidth.  Two stages, an integer shift per coarse window: the fine channels are "
           "left 2\u03c0 u \u03b4 / L1 (u <= 1/2 coarse channel, \u03b4 <= 1/2 sample), whatever the rate.")


def offset_png(path):
    """Response against frequency offset, +-3 coarse channels: one stage at 64 channels (what one would
    have used before), one stage at 64 x 56, and two stages (WOLA or PFB stage 1, rectangular stage 2):
    for two stages the worst and the median, at each offset, over the 56 fine channels of a coarse
    channel (where a fine channel sits in its coarse channel matters), each peak-normalised; fine
    responses as their envelope over one fine channel."""
    coarse, fine, wola, pfb = T.offset_designs()
    K = 56
    os_ = 16
    x = (np.arange(-3 * K * os_, 3 * K * os_) + 0.5) / (K * os_)            # coarse channels
    fig, ax = plt.subplots(figsize=(11, 5), facecolor=SURFACE)
    style(ax)
    env = lambda r: np.repeat(r.reshape(-1, os_).max(axis=1), os_)        # the envelope over one fine channel
    db = lambda r: 10 * np.log10(np.maximum(r, 1e-14))
    for d, col, lab in ((wola, SLOT[0], "two stage, WOLA 12.5% stage 1"), (pfb, SLOT[2], "two stage, PFB 32 taps stage 1")):
        R = np.array([env(T.offset_response(d, k, x)) for k in d.ks()])
        ax.plot(x, db(R.max(axis=0)), color=col, linewidth=1.8, label=lab + ": worst fine channel")
        ax.plot(x, db(np.median(R, axis=0)), color=col, linewidth=1.2, alpha=0.55, label=lab + ": median")
    ax.plot(x, db(T.offset_response(coarse, 0, x)), label="one stage, 64 channels (the coarse resolution)", **REF)
    ax.plot(x, db(env(T.offset_response(fine, 0, x))), color=INK2, linewidth=1.3, linestyle=":",
            label="one stage, 64 x 56 channels (the fine resolution)")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-100, 3)
    ax.set_xlabel("frequency offset from the channel's centre (coarse channels)", color=INK, fontsize=10)
    ax.set_ylabel("power response (dB, of the channel's peak)", color=INK, fontsize=10)
    legend(ax, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    finish(fig, path, "Leakage against frequency offset: two stages against fewer channels in one stage", right=0.66,
           note="Vorbis 12.5% WOLA windows; 64 coarse channels, O = 8/7, 56 fine channels each.  Two stages: at each "
                "offset the worst and the median over a coarse channel's fine channels (envelope over one fine "
                "channel).  The WOLA first stage's broad response returns as an alias image O coarse channels from "
                "the edge fine channels.")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "plots"
    os.makedirs(out, exist_ok=True)
    for name, fn in (("two-stage-offset.png", offset_png), ("two-stage-coarse.png", coarse_png), ("two-stage-fine.png", fine_png),
                     ("two-stage-line.png", line_png), ("two-stage-delayrate.png", delayrate_png),
                     ("two-stage-sensitivity.png", sensitivity_png)):
        print(name, flush=True)
        fn(os.path.join(out, name))


if __name__ == "__main__":
    main()
