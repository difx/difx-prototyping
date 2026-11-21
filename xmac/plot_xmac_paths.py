#!/usr/bin/env python3
"""plot_xmac_paths.py - plots/xmac-paths.svg: the station + XMAC paths, the layout of each buffer, and
which kernel writes and reads it (docs/xmac-paths.md).  The times are typed in from
results-local/gina18-wide-20261120-140632/nbit2-nband8-nchan128-nant10.log (ms per 20 ms subint / 20).

Usage: python3 plot_xmac_paths.py [out.svg]"""
import sys
from xml.sax.saxutils import escape

W = 1400
FONT = "font-family='Helvetica, Arial, sans-serif'"
C = {   # fills, strokes
    "kernel": ("#eef2f7", "#5b6b80"),
    "f32": ("#fde4cc", "#c86a12"),      # FP32 spectra
    "f16": ("#d8f0dc", "#2e8540"),      # FP16 spectra
    "tens": ("#e6dcf5", "#6a46a8"),     # tensor layout
    "smp": ("#e8e8e8", "#777777"),      # FP32 samples
    "vis": ("#dbe9f7", "#2f6db0"),
    "chip": ("#fff8d6", "#b39200"),     # on-chip layout note
}
out = []
def add(s): out.append(s)

def text(x, y, lines, size=12, anchor="start", weight="normal", color="#1d1d1d", lh=None, italic=False):
    lh = lh or size * 1.25
    if isinstance(lines, str): lines = [lines]
    st = " font-style='italic'" if italic else ""
    add(f"<text x='{x}' y='{y}' {FONT} font-size='{size}' text-anchor='{anchor}' font-weight='{weight}' fill='{color}'{st}>")
    for i, l in enumerate(lines):
        add(f"<tspan x='{x}' dy='{0 if i == 0 else lh}'>{l}</tspan>")
    add("</text>")

def box(x, y, w, h, kind, title=None, lines=(), size=11, dash=False, rx=6):
    f, s = C[kind]
    d = " stroke-dasharray='5 3'" if dash else ""
    add(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='{rx}' fill='{f}' stroke='{s}' stroke-width='1.4'{d}/>")
    ty = y + 16
    if title:
        text(x + 8, ty, escape(title) if "<" not in title else title, size=12, weight="bold")
        ty += 15
    if lines:
        text(x + 8, ty, list(lines), size=size, lh=size * 1.3)

def arrow(x1, y1, x2, y2, label=None, color="#444", above=True):
    add(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' stroke-width='1.6' marker-end='url(#ah)'/>")
    if label:
        text((x1 + x2) / 2, (y1 + y2) / 2 + (-5 if above else 13), label, size=10, anchor="middle", color="#444",
             italic=True)

# ---- columns
LX, LW = 16, 170           # path label
SX, SW = 196, 250          # station kernel(s)
B1X, B1W = 466, 178        # DRAM buffer 1
TX, TW = 664, 140          # transpose kernel
B2X, B2W = 824, 170        # DRAM buffer 2
XX, XW = 1014, 262         # XMAC kernel
RX = 1290                  # chain time
ROWH, GAP, TOP = 104, 12, 118

rows = [
    dict(name="gpudifx today", status="the reference", tag="gdx",
         station=("gpudifx station processing", ["3 kernels, through DRAM: decode (FP64)", "→ FP32 samples → cuFFT → FP32", "spectra → fractional-sample rotation"]),
         b1="f32", b2=None, trans=False,
         xmac=("gpudifx XMAC", ["a thread per (baseline, product, channel);", "reads both stations' spectra every", "window (L1/L2), sums in registers"]),
         t="0.431", ts="0.291", tx="0.139"),
    dict(name="end-to-end", status="+ gpudifx XMAC", tag="e2e",
         station=None, b1="f32", b2=None, trans=False,
         xmac=("gpudifx XMAC", ["as above"]), t="0.176", ts="0.036", tx="0.139"),
    dict(name="end-to-end", status="+ tiled FP32 XMAC", tag="tiled",
         station=None, b1="f32", b2=None, trans=False,
         xmac=("tiled FP32 (CUDA cores)", ["a thread per (channel, 4 × 4 receiver", "pairs), sums in registers: each value", "loaded is used 4 times"]),
         t="0.067", ts="0.036", tx="0.029"),
    dict(name="end-to-end", status="+ transpose + tensor", tag="trans",
         station=None, b1="f32", b2="tens", trans=True,
         xmac=("tensor XMAC (WMMA 16×16×16)", ["fragments loaded straight from the", "layout (L1): Re += ArBrᵀ + AiBiᵀ,", "Im += AiBrᵀ − ArBiᵀ; no shared stage"]),
         t="0.100", ts="0.036", tt="0.048", tx="0.014"),
    dict(name="end-to-end → tensor layout", status="+ tensor  (“the layout path”)", tag="layout",
         station="layout", b1=None, b2="tens", trans=False,
         xmac=("tensor XMAC (WMMA 16×16×16)", ["as above: reads the layout ready-made"]),
         t="0.054", ts="0.040", tx="0.014"),
    dict(name="end-to-end → FP16 spectra", status="+ staged mma.sync  (THE path)", tag="staged", hl=True,
         station="half", b1="f16", b2=None, trans=False,
         xmac=("staged XMAC, mma.sync (default)", ["per 16 windows: 16-B loads (4 ch × 4 win),",
                                                    "register transpose, 16-B stores into the",
                                                    "stage; ldmatrix → mma.sync m16n8k16;",
                                                    "Im from A' = (im, −re); next tile prefetched"]),
         chip="stage: [channel][receiver row: 16 win × (re, im)], 64 B rows, XOR-swizzled",
         t="0.046", ts="0.028", tx="0.017"),
    dict(name="end-to-end → FP16 spectra", status="+ staged WMMA (comparison)", tag="wmma",
         station="half", b1="f16", b2=None, trans=False,
         xmac=("staged XMAC, WMMA", ["16-B loads (4 ch × 2 win) → __half2", "stores into the stage, then WMMA;", "load, barrier, products, barrier"]),
         chip="stage: [channel][plane re | im][receiver][16 win], rows 24 halves",
         t="—", ts="0.028"),
]

H = TOP + len(rows) * (ROWH + GAP) + 330
add(f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>")
add("<defs><marker id='ah' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'>"
    "<path d='M0,0 L10,5 L0,10 z' fill='#444'/></marker></defs>")
add(f"<rect width='{W}' height='{H}' fill='#ffffff'/>")

text(16, 30, "fft-fusion: the station + XMAC paths, their data layouts, and where each layout is written and read",
     size=17, weight="bold")
text(16, 52, ["Every path starts from packed VDIF in DRAM (1, 2, 4 or 8 bits a sample) and ends in visibilities "
              "[freq][baseline][product][channel] FP32 complex, summed over the subint's windows.",
              "Solid boxes: kernels (what they hold on chip in the yellow notes).  Coloured boxes: buffers in DRAM, "
              "written by the kernel to their left and read by the one to their right."], size=12, color="#333")

# column headers
hy = TOP - 12
for x, w, t in [(SX, SW, "station processing (kernel)"), (B1X, B1W, "DRAM"), (TX, TW, "rearrange"),
                (B2X, B2W, "DRAM"), (XX, XW, "XMAC (kernel)")]:
    text(x + w / 2, hy, t, size=12, anchor="middle", weight="bold", color="#555")
text(RX + 50, hy - 14, ["chain, ms per", "ms of data*"], size=11, anchor="middle", weight="bold", color="#555")

E2E = ("end-to-end kernel (fused, cuFFTDx)", ["decode, fringe rotation, FFT and", "fractional-sample correction in one;", "the FFT in shared memory"])
BUF = {
    "f32": ("FP32 spectra", ["[ant][win][band][fftchan]", "float2 (re, im): 8 B a value", "channel fastest"]),
    "f16": ("FP16 spectra", ["[ant][win][band][chan]", "__half2 (re, im): 4 B a value", "channel fastest"]),
    "tens": ("tensor layout (FP16)", ["[freq][chan][16-win block]", "[plane re | im][receiver][16 win]", "window fastest: 4 B a value"]),
}
for i, r in enumerate(rows):
    y = TOP + i * (ROWH + GAP)
    cy = y + ROWH / 2
    if r.get("hl"):
        add(f"<rect x='8' y='{y - 6}' width='{W - 16}' height='{ROWH + 12}' rx='10' fill='#f1faf2' stroke='#2e8540' stroke-width='2'/>")
    # label
    text(LX, y + 34, [escape(r["name"])], size=13, weight="bold")
    text(LX, y + 52, [escape(r["status"])], size=12, color="#333")
    # station
    st = r["station"]
    if st is None or st == "half" or st == "layout":
        t, l = E2E
        l = list(l)
        if st == "half":
            l = l + ["→ writes FP16 (re, im), station order"]
        if st == "layout":
            l = l[:2] + ["→ FP16 tile in shared memory, then", "16-B stores of 16-window runs"]
        box(SX, y, SW, ROWH, "kernel", t, l)
        if st == "layout":
            box(SX + 8, y + ROWH - 22, SW - 96, 18, "chip", None, rx=3)
            text(SX + 14, y + ROWH - 9, "tile: [plane][chan][W windows]", size=10)
    else:
        t, l = st
        box(SX, y, SW, ROWH, "kernel", t, l)
    # buffers and arrows
    bx_end = SX + SW
    if r["b1"]:
        t, l = BUF[r["b1"]]
        box(B1X, y + 8, B1W, ROWH - 16, r["b1"], t, l)
        arrow(bx_end, cy, B1X - 2, cy)
        bx_end = B1X + B1W
    if r["trans"]:
        box(TX, y, TW, ROWH, "kernel", "transpose", ["FP32 → FP16;", "via shared memory", "[8 recv][16 win][33]"])
        arrow(bx_end, cy, TX - 2, cy)
        bx_end = TX + TW
    if r["b2"]:
        t, l = BUF[r["b2"]]
        box(B2X, y + 8, B2W, ROWH - 16, r["b2"], t, l)
        arrow(bx_end, cy, B2X - 2, cy, "station kernel writes it directly" if r["tag"] == "layout" else None)
        bx_end = B2X + B2W
    elif r["b1"]:
        arrow(bx_end, cy, XX - 2, cy, "read as written: the XMAC rearranges on chip" if r["b1"] == "f16" else "read as written")
        bx_end = None
    if r["b2"]:
        arrow(bx_end, cy, XX - 2, cy)
    # xmac
    t, l = r["xmac"]
    box(XX, y, XW, ROWH, "kernel", t, l, size=10.5)
    if r.get("chip"):
        # the on-chip stage note sits under the arrow into the XMAC
        add(f"<rect x='{B1X + B1W + 12}' y='{y + ROWH - 30}' width='{XX - B1X - B1W - 24}' height='22' rx='3' "
            f"fill='{C['chip'][0]}' stroke='{C['chip'][1]}' stroke-width='1.2'/>")
        text(B1X + B1W + 18, y + ROWH - 15, escape(r["chip"]), size=10)
    for key, bx, bw in (("ts", SX, SW), ("tt", TX, TW), ("tx", XX, XW)):
        if r.get(key):
            text(bx + bw - 8, y + ROWH - 8, r[key] + " ms/ms", size=11, anchor="end", weight="bold", color="#24508a")
    # chain time
    text(RX + 50, cy + 5, r["t"], size=15 if r.get("hl") else 13, anchor="middle",
         weight="bold" if r.get("hl") else "normal")

# ---- the layouts key
ky = TOP + len(rows) * (ROWH + GAP) + 18
text(16, ky, "* RTX PRO 6000 Blackwell (gina18, 2026-11-20): 2 bits, 4 dual-pol subbands, 128 channels, 10 stations; "
     "best variant of each kernel.  Blue: each kernel's own time (they add up to the chain, with the phase and setup kernels).  Staged WMMA: xmacbench only.", size=11, color="#555")
ky += 30
text(16, ky, "The layouts, slowest index first (the last is contiguous in memory)", size=14, weight="bold")
ky += 10
key = [
    ("f32", "FP32 spectra (fused kernel → gpudifx, tiled, transpose)",
     "[ant] [win] [band] [fftchan]   float2: a window's channels of one band are contiguous; "
     "bands 2f, 2f+1 are the pols of freq f"),
    ("f16", "FP16 spectra (fused kernel → staged XMAC)",
     "[ant] [win] [band] [chan]   __half2: the same order at half the bytes, only the nchan channels used"),
    ("tens", "tensor layout (transpose or fused kernel → tensor XMAC)",
     "[freq] [chan] [16-window block] [plane: re | im] [receiver = 2 ant + pol] [16 windows]   half: "
     "each receiver's 16 windows (32 B) are a tensor-core row"),
    ("chip", "layout path's tile (fused kernel, shared memory)",
     "[plane] [channel] [W windows + 2 padding]   W = 16 (or 8) windows, gathered over passes of the "
     "kernel's FFTs, then written out as 16-B runs"),
    ("chip", "staged stage, mma.sync (XMAC, shared memory)",
     "[channel, 16 B apart past a 128-B multiple] [receiver row] [4 chunks of 4 windows, chunk q at q ^ ((row>>1)&3)] "
     "[window] [re, im]   64 B a row"),
    ("chip", "staged stage, WMMA (XMAC, shared memory)",
     "[channel] [plane: re | im] [receiver] [16 windows, rows 24 halves apart]   TensorLayout for one channel"),
    ("vis", "visibilities (every XMAC → DRAM)",
     "[freq] [baseline (a < b)] [product AA, BB, AB, BA] [chan]   float2, partial sums added atomically "
     "when window chunks split a sum"),
]
for k, (kind, name, desc) in enumerate(key):
    yy = ky + 8 + k * 30
    f, s = C[kind]
    add(f"<rect x='16' y='{yy}' width='22' height='18' rx='3' fill='{f}' stroke='{s}' stroke-width='1.3'/>")
    text(48, yy + 13, escape(name), size=12, weight="bold")
    idx, _, note = desc.partition("   ")
    add(f"<text x='430' y='{yy + 13}' {FONT} font-size='11.5' fill='#222'><tspan font-weight='bold'>{escape(idx)}</tspan>"
        f"<tspan dx='8'>— {escape(note)}</tspan></text>")
ky += 8 + len(key) * 30 + 14
text(16, ky, ["Where the rearrangement from channel-fastest (station order) to window-fastest per receiver "
              "(tensor-core rows) happens: in a separate transpose kernel, DRAM to DRAM (FP32 in, FP16 out);",
              "in the station kernel's shared-memory tile (the layout path); or in each XMAC block's copy into "
              "its shared-memory stage (the staged paths), so the spectra cross DRAM once each way at 4 B a value."],
     size=12, color="#333")
add("</svg>")
open(sys.argv[1] if len(sys.argv) > 1 else "plots/xmac-paths.svg", "w").write("\n".join(out))
