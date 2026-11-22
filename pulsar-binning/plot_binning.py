#!/usr/bin/env python3
"""plot_binning.py - plots/binning-approaches.svg: the pulsar-binning approaches (docs/approaches.md), each
drawn as what the XMAC multiplies for one channel group: a toy grid of 8 channels x windows, each
(window, channel) coloured by its bin, with dispersion shifting the bins from channel to channel.
The ratios are typed in from results-local/ar313-adeller-bins-20261122-140307/table.txt (run-bins.sh on
the RTX 2070 SUPER) and results-local/gina8-bins-20261122-142203/table.txt (the A100, OzSTAR job 21374617),
avg 1: each version's ms per subint over the unbinned staged XMAC's.

Usage: python3 plot_binning.py [out.svg]"""
import math
import sys
from xml.sax.saxutils import escape

W = 1400
FONT = "font-family='Helvetica, Arial, sans-serif'"
BIN = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948", "#b07aa1", "#ff9da7"]   # 8 toy bins
GREY, PAD, OFF = "#d9d9d9", "#f4f4f4", "#ffffff"
C = {"kernel": ("#eef2f7", "#5b6b80"), "f16": ("#d8f0dc", "#2e8540"), "cm": ("#fde4cc", "#c86a12"),
     "vis": ("#dbe9f7", "#2f6db0"), "chip": ("#fff8d6", "#b39200"), "no": ("#f6f6f6", "#999999")}
out = []
def add(s): out.append(s)

def text(x, y, lines, size=12, anchor="start", weight="normal", color="#1d1d1d", lh=None, italic=False):
    lh = lh or size * 1.25
    if isinstance(lines, str): lines = [lines]
    st = " font-style='italic'" if italic else ""
    add(f"<text x='{x}' y='{y}' {FONT} font-size='{size}' text-anchor='{anchor}' font-weight='{weight}' fill='{color}'{st}>")
    for i, l in enumerate(lines):
        add(f"<tspan x='{x}' dy='{0 if i == 0 else lh}'>{escape(l)}</tspan>")
    add("</text>")

def box(x, y, w, h, kind, title=None, lines=(), size=10.5, dash=False, rx=6):
    f, s = C[kind]
    d = " stroke-dasharray='5 3'" if dash else ""
    add(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='{rx}' fill='{f}' stroke='{s}' stroke-width='1.4'{d}/>")
    ty = y + 16
    if title:
        text(x + 8, ty, title, size=12, weight="bold")
        ty += 15
    if lines:
        text(x + 8, ty, list(lines), size=size, lh=size * 1.3)

def arrow(x1, y1, x2, y2, color="#444"):
    add(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' stroke-width='1.5' marker-end='url(#ah)'/>")

# ---- the toy pulsar: bin of (window w, channel c); a period of PW windows, NB bins, dispersion shifting
# channel c's pulse by c x SLOPE windows (higher channels earlier)
def binOf(w, c, PW, NB, slope, nb=None):
    ph = ((w + 0.5 + c * slope) / PW) % 1.0
    return int(ph * NB) % NB

def grid(x, y, nwin, cw, ch, colour, nchan=8, blocks=16, label_blocks=True, hatch=None):
    """colour(w, c) -> fill; vertical lines every `blocks` windows (the MMA's 16 windows)"""
    for c in range(nchan):
        for w in range(nwin):
            sw = "stroke='#ffffff' stroke-width='0.5'" if cw >= 5 else "stroke='none'"     # thin cells: no outline
            add(f"<rect x='{x + w * cw:.1f}' y='{y + c * ch:.1f}' width='{cw + (0 if cw >= 5 else 0.3):.1f}' height='{ch - (0 if cw >= 5 else 0.6):.1f}' "
                f"fill='{colour(w, c)}' {sw}/>")
    for b in range(0, nwin + 1, blocks):
        add(f"<line x1='{x + b * cw:.1f}' y1='{y - 3}' x2='{x + b * cw:.1f}' y2='{y + nchan * ch + 3}' stroke='#222' "
            f"stroke-width='1.6'/>")
    if hatch:
        for (w0, w1) in hatch:
            add(f"<rect x='{x + w0 * cw:.1f}' y='{y:.1f}' width='{(w1 - w0) * cw:.1f}' height='{nchan * ch:.1f}' "
                f"fill='url(#hatch)' stroke='none'/>")

# ---- the page
ROWH, GAP = 150, 14
rows = []
H = 1880
add(f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>")
add("<defs><marker id='ah' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'>"
    "<path d='M0,0 L10,5 L0,10 z' fill='#444'/></marker>"
    "<pattern id='hatch' width='6' height='6' patternUnits='userSpaceOnUse' patternTransform='rotate(45)'>"
    "<rect width='6' height='6' fill='#ffffff' fill-opacity='0.55'/><line x1='0' y1='0' x2='0' y2='6' stroke='#555' "
    "stroke-width='1.2'/></pattern></defs>")
add(f"<rect width='{W}' height='{H}' fill='#ffffff'/>")
text(16, 30, "pulsar-binning: the approaches, drawn as what the XMAC multiplies for one group of 8 channels", size=17,
     weight="bold")
text(16, 52, ["Each cell is one (FFT window, channel) of the station spectra, coloured by its pulse-phase bin (a toy: 8 bins). "
              "Black lines: the tensor-core XMAC's 16-window blocks (its MMA sums 16 windows of a channel at once).",
              "Dispersion delays the lower channels, so the bins slant from channel to channel: a 16-window block can hold "
              "several bins, and a different set in each channel."], size=12, color="#333")

# ---- the problem: three regimes
y0 = 92
text(16, y0 + 10, "The problem", size=15, weight="bold")
CW, CH = 9, 9
panels = [
    ("millisecond pulsar, many bins", "1.5 ms, 128 channels, 128 bins: a bin is 2.9 windows,",
     "adjacent channels 5.7 windows apart: every block mixes bins", lambda w, c: BIN[binOf(w, c, 24, 8, 1.3)]),
    ("millisecond pulsar, gate", "a 10% on-pulse window: most blocks hold no on-pulse",
     "window in any channel (white: weight 0)",
     lambda w, c: BIN[3] if 0.40 <= ((w - 8 + 0.5 + c * 1.0) / 40) % 1.0 < 0.60 else OFF),
    ("slow pulsar", "500 ms, 128 bins: a bin is 122-977 windows; a subint",
     "covers 5% of the period: long runs, few bins", lambda w, c: BIN[binOf(w, c, 400, 8, 1.3)]),
]
px = 16
for title, l1, l2, col in panels:
    text(px, y0 + 34, title, size=12, weight="bold")
    text(px, y0 + 50, [l1, l2], size=10.5, color="#333")
    grid(px, y0 + 76, 48, CW, CH, col)
    text(px, y0 + 76 + 8 * CH + 16, "→ windows (48 = three 16-window blocks); ↓ channels", size=10, color="#666", italic=True)
    px += 470

# ---- the approaches
y = y0 + 210
text(16, y, "The approaches", size=15, weight="bold")
text(W - 16, y, "binned / unbinned XMAC, ms per subint, 10 stations: RTX 2070 SUPER / A100", size=11, anchor="end",
     color="#555", weight="bold")
y += 14
LX, LW = 16, 200          # label
PX, PW_ = 226, 470        # pipeline box
GX = 716                  # picture
RX = 1150                 # ratios
CW2, CH2 = 12, 9.5
cases = ["1.5 ms, 128 ch, 128 bins", "500 ms, 1024 ch, 128 bins", "1.5 ms, 128 ch, 16 bins", "gate; 16 scr. (1.5 ms, 128)"]

def ratios(yy, vals, hl=None):
    for k, (case, v) in enumerate(zip(cases, vals)):
        if v is None: continue
        bold = "bold" if hl == k else "normal"
        text(RX, yy + 22 + k * 30, case, size=10.5, color="#555")
        dy = -15 if len(v) > 14 else 0             # a long value above its case (that row is empty)
        text(W - 20, yy + 22 + k * 30 + dy, v, size=12.5, anchor="end", weight=bold,
             color="#2e8540" if hl == k else "#24508a")

def row_frame(yy, name, outputs, hl=False):
    if hl:
        add(f"<rect x='8' y='{yy - 6}' width='{W - 16}' height='{ROWH + 12}' rx='10' fill='#f7fbf7' stroke='#2e8540' "
            "stroke-width='1.5'/>")
    text(LX, yy + 22, name, size=13, weight="bold")
    text(LX, yy + 40, outputs, size=11, color="#333", lh=14)

MS = lambda w, c: BIN[binOf(w, c, 24, 8, 1.3)]
gy = lambda yy: yy + 30

# 1. naive
row_frame(y, "1. naive port", ["mpifxcorr's loop", "every output kind"])
box(PX, y + 8, PW_, ROWH - 16, "kernel", "a thread per (baseline, product, channel)",
    ["walks the windows in time, reading both stations' spectra (CUDA cores, FP32)",
     "holds a sum while the bin stays the same; at each change adds it to that bin's",
     "output in global memory.  Dispersion: a warp's channels change bin at different",
     "windows, so those adds do not coalesce (18x faster at DM 0).",
     "The reference for the checks, not a candidate."])
grid(GX, gy(y), 32, CW2, CH2, MS)
for c in range(8):   # dots at the bin changes of each channel
    for w in range(1, 32):
        if binOf(w, c, 24, 8, 1.3) != binOf(w - 1, c, 24, 8, 1.3):
            add(f"<circle cx='{GX + w * CW2:.1f}' cy='{gy(y) + c * CH2 + CH2 / 2:.1f}' r='2.2' fill='#111'/>")
text(GX, gy(y) + 8 * CH2 + 16, "● a flush to global memory, per output, at every bin change", size=10, color="#555", italic=True)
ratios(y, ["189x / 52x", "8.1x / 7.8x", "6.9x / 11x", "5.1x / 8.0x"])
y += ROWH + GAP

# 2. weights in the stage
row_frame(y, "2. weights in the stage", ["(stage) the gate and", "scrunched output: one pass"], hl=True)
box(PX, y + 8, PW_, ROWH - 16, "kernel", "the staged XMAC, weighted (xmacTensorStagedWeighted)",
    ["the plain copy into shared memory, with each (window, channel)'s weight beside it",
     "(FP16, a thread each); each lane multiplies its A registers by its two windows'",
     "weights (4 HMUL2, the sign included).  Each 16-window block is classified first:",
     "skipped (all weights 0), as is (all 1), or weighted.  Sum over bins = one weighted",
     "sum: one pass.  Costs: mostly the classification (0.99-1.16x / 1.15-1.34x)."])
gate = lambda w, c: BIN[3] if 0.40 <= ((w - 8 + 0.5 + c * 1.0) / 40) % 1.0 < 0.60 else OFF
skip = []
for b0 in range(0, 48, 16):
    if not any(gate(w, c) != OFF for w in range(b0, b0 + 16) for c in range(8)):
        skip.append((b0, b0 + 16))
grid(GX, gy(y), 48, CW2 * 2 / 3, CH2, gate, hatch=skip)
text(GX, gy(y) + 8 * CH2 + 16, "the gate: hatched blocks (no on-pulse window in any channel) are skipped", size=10,
     color="#555", italic=True)
ratios(y, [None, None, None, "0.40; 1.15x / 0.67; 1.32x"], hl=3)
text(RX, y + 22 + 3 * 30 + 18, ["the gate does less than the unbinned XMAC:", "it skips the off-pulse blocks"],
     size=10, color="#2e8540", italic=True)
y += ROWH + GAP

# 3. a pass per bin
row_frame(y, "3. a pass per bin", ["(passes) separate bins"])
box(PX, y + 8, PW_, ROWH - 16, "kernel", "the weighted XMAC, once per bin, one-hot weights",
    ["each pass keeps one bin (weight 1, the rest 0) and skips the 16-window blocks",
     "without it in any of the block's channels.",
     "With dispersion a bin is in many blocks of an 8-channel group, so the passes",
     "barely skip: nbins x the XMAC, nearly.  Every pass also classifies every block",
     "(the slow pulsar's empty bins still cost a pass)."])
grid(GX, gy(y), 32, CW2, CH2, lambda w, c: BIN[2] if binOf(w, c, 24, 8, 1.3) == 2 else GREY)
text(GX, gy(y) + 8 * CH2 + 16, "the pass for one bin (colour); x nbins passes", size=10, color="#555", italic=True)
ratios(y, ["44x / 82x", "23x / 58x", "6.1x / 11x", None])
y += ROWH + GAP

# 4. sorted: the slots picture
def slots(w, c, PWt=24, NB=8, slope=1.3, nwin=106, pad=True):
    """each channel's windows (of the first nwin) sorted by bin, each bin padded to the most any channel of
    the group has, rounded to 16 slots: returns the fill of slot w in channel c"""
    cnt = [[0] * NB for _ in range(8)]
    for cc in range(8):
        for ww in range(nwin): cnt[cc][binOf(ww, cc, PWt, NB, slope)] += 1
    layout = []
    for b in range(NB):
        mx = max(cnt[cc][b] for cc in range(8))
        size = (mx + 15) // 16 * 16 if mx else 0
        layout.append(size)
    pos = 0
    for b in range(NB):
        if w < pos + layout[b]:
            return BIN[b] if w - pos < cnt[c][b] else PAD
        pos += layout[b]
    return OFF

row_frame(y, "4. sorted, permuted copy", ["(sorted) separate bins;", "best for short runs"], hl=True)
box(PX, y + 8, PW_, ROWH - 16, "kernel", "sort → permute → the XMAC over the slots",
    ["a stable counting sort per channel (bins computed on the fly); per channel group",
     "each bin gets the same slots, padded to 16 (the most windows any channel has);",
     "a permute writes the spectra in slot order (zeros in the padding); the staged",
     "XMAC runs a block per segment of a bin and writes that bin once.",
     "From channel-major spectra (a transpose in the binning; from window-major the",
     "permute is 4-byte gathers: 17-28 ms)."])
box(PX + PW_ - 186, y + ROWH - 42, 178, 26, "cm", None, rx=4)
text(PX + PW_ - 180, y + ROWH - 25, "in: [station][band][channel][window]", size=9.5)
SX = 0.5
grid(GX, gy(y), 128, CW2 * 0.28, CH2, lambda w, c: slots(w, c))
text(GX, gy(y) + 8 * CH2 + 16, "106 windows sorted into 128 slots: each bin a 16-slot block, the same across the group; "
     "grey = padding", size=10, color="#555", italic=True)
ratios(y, ["6.5x / 7.1x", "5.6x / 5.6x", "4.6x / 4.1x", None])
y += ROWH + GAP

# 5. gather
row_frame(y, "5. sorted, gathered", ["(gather) separate bins;", "best for long runs"], hl=True)
box(PX, y + 8, PW_, ROWH - 16, "kernel", "sort → the XMAC gathering the slots",
    ["the same sort and slots, no permuted copy: the XMAC's copy into its stage reads",
     "each slot's window through the slot table (a 16-B index load per channel and",
     "4 slots, then 4-byte loads) from the channel-major spectra in time order.",
     "Saves the permute's pass; with long runs the loads are nearly contiguous, with runs",
     "of 1-3 windows scattered.  Also from the usual window-major spectra (gatherwm, no",
     "copy at all): on the A100 it beats a transpose in 19 of 24 runs; not at 128 ch."])
grid(GX, gy(y), 128, CW2 * 0.28, CH2, lambda w, c: slots(w, c))
for k, (sl, src) in enumerate([(3, 1), (21, 25), (40, 49)]):    # a few index arrows, illustrative
    xs = GX + (sl + 0.5) * CW2 * 0.28
    add(f"<line x1='{xs:.1f}' y1='{gy(y) - 2}' x2='{xs:.1f}' y2='{gy(y) - 12}' stroke='#c86a12' stroke-width='1.2'/>")
text(GX, gy(y) - 16, "slot → window (index table) →", size=9.5, color="#c86a12")
text(GX, gy(y) + 8 * CH2 + 16, "the same slots, read where the windows are", size=10, color="#555", italic=True)
ratios(y, ["12.1x / 6.6x", "4.2x / 4.7x", "3.3x / 3.2x", None], hl=1)
y += ROWH + GAP

# 6. split (not built)
row_frame(y, "6. split at bin changes", ["not built"])
box(PX, y + 8, PW_, ROWH - 16, "no", "the window order kept; each 16-window block split",
    ["at the bins it holds (a masked MMA per bin present, per channel), the sums",
     "written to a bin whenever a channel's bin changes.",
     "Writing the sums inside the window loop (the first sorted kernel did) spilled",
     "0.5-1.9 KB of registers; for long runs gather already does the job, and for short",
     "runs it costs a pass per bin present in a block."], dash=True)
grid(GX, gy(y), 32, CW2, CH2, MS)
for c in range(8):
    for w in range(1, 32):
        if binOf(w, c, 24, 8, 1.3) != binOf(w - 1, c, 24, 8, 1.3):
            xx = GX + w * CW2
            add(f"<line x1='{xx:.1f}' y1='{gy(y) + c * CH2:.1f}' x2='{xx:.1f}' y2='{gy(y) + (c + 1) * CH2:.1f}' "
                "stroke='#111' stroke-width='1.6' stroke-dasharray='2 1'/>")
text(GX, gy(y) + 8 * CH2 + 16, "dashed: where each channel's MMA would have to split", size=10, color="#555", italic=True)
ratios(y, ["—", "—", "—", None])
y += ROWH + GAP

# ---- the decision, and the layouts
y += 6
text(16, y, "Which", size=15, weight="bold")
text(16, y + 20, ["The gate and scrunched output (any weights, negative too): weights in the stage, one pass: 0.31-1.16x the "
                  "unbinned XMAC on the 2070, 0.56-1.34x on the A100.",
                  "Separate bins: sorted per channel and gathered straight from the station kernel's usual window-major "
                  "spectra (gatherwm): 3.1-4.7x on the A100 for long runs, 10-11x",
                  "for the shortest (where a transpose to channel-major does better, 7.8x, and on the 2070 the permuted "
                  "copy, 6.5x).  Bins narrower than a window (1.5 ms,",
                  "1024 channels, 128 bins: 14x / 18x) sample rather than resolve the pulse: warn, do not optimise.  "
                  "A pass per bin and the naive port: not for many bins."], size=12, color="#333", lh=16)
y += 106
text(16, y, "The layouts, slowest index first (the last is contiguous)", size=14, weight="bold")
key = [
    ("f16", "FP16 spectra, window-major", "[station] [window] [band] [channel]   what the station kernel writes now; "
     "the stage and passes read it (16-B loads of 4 channels x 1 window)"),
    ("cm", "FP16 spectra, channel-major", "[station] [band] [channel] [window]   a channel's windows contiguous: what "
     "sorted and gather start from (a transpose in the binning; gatherwm reads window-major)"),
    ("cm", "slots (sorted, permuted copy)", "[station] [band] [channel] [slot]   each channel's windows in bin order, bins "
     "padded to 16 slots, the same slots across a channel group"),
    ("chip", "bins, slot table, segments", "bins [window][freq][chan] u8;  slotWin [freq][chan][slot] window or -1;  "
     "segs [freq][group][k] = (bin, first, last + 1 slot block, the bin's segments)"),
    ("vis", "binned visibilities", "[bin] [freq] [baseline] [product] [channel / avg]   float2; summed in the XMAC over "
     "avg channels, added over an integration with --accumulate"),
]
for k, (kind, name, desc) in enumerate(key):
    yy = y + 10 + k * 28
    f, s = C[kind]
    add(f"<rect x='16' y='{yy}' width='22' height='18' rx='3' fill='{f}' stroke='{s}' stroke-width='1.3'/>")
    text(48, yy + 13, name, size=12, weight="bold")
    idx, _, note = desc.partition("   ")
    add(f"<text x='300' y='{yy + 13}' {FONT} font-size='11.5' fill='#222'><tspan font-weight='bold'>{escape(idx)}</tspan>"
        f"<tspan dx='8'>— {escape(note)}</tspan></text>")
y += 10 + len(key) * 28 + 16
text(16, y, "Ratios: run-bins.sh, commit fafcd29: results-local/ar313-adeller-bins-20261122-140307 (2070) and "
     "gina8-bins-20261122-142203 (A100, job 21374617); each version's ms per 25.6 ms subint over the unbinned staged "
     "XMAC's (2070 1.19-1.50 ms, A100 0.32-0.40);", size=10.5, color="#555")
text(16, y + 15, "sorted counts the sort, permute and XMAC, not the channel-major spectra (as a transpose here: 1.4-1.5 ms "
     "on the 2070);", size=10.5, color="#555")
text(16, y + 30, "approach 2: c633b2e (2070 ar313-adeller-bins-20261122-164224, A100 job 21376562); gatherwm on the A100: gina5-bins-20261122-160540 (commit bc93212, job 21375824).", size=10.5, color="#555")
H2 = y + 55
out[0] = f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H2}' viewBox='0 0 {W} {H2}'>"
out[2] = f"<rect width='{W}' height='{H2}' fill='#ffffff'/>"
add("</svg>")
open(sys.argv[1] if len(sys.argv) > 1 else "plots/binning-approaches.svg", "w").write("\n".join(out))
