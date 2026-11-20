#!/usr/bin/env python3
"""matrix_summary.py - one CSV row per shape from run-matrix.sh's runs.csv (each shape's status) and
fusebench's --csv rows (raw.csv: every timing and check), for results/ in the repository.

Usage: python3 matrix_summary.py runs.csv raw.csv [--host H --commit C --date D --power W --clock MHz] > summary.csv

Columns (times in ms per ms of data):
  host, gpu, cc, power_w, sm_clock_lock_mhz, commit, date   where and what
  nbit, nsub, nband, nchan, nant, nwin, subint_ms, reps      the shape (nsub dual-pol subbands)
  status                  ok, fail (a check failed or a non-zero exit), oom, skip (did not fit),
                          unsupported (refused by fusebench: a window over the lean decode's 32 frames)
  exit, note              fusebench's exit code; why a shape was skipped
  checks, checks_failed   number of checks; the names of any that failed (;-separated)
  worst_check, worst_ratio  the check closest to its limit, and its rms / limit
  station_unfused, station_fp32, station_layout, station_fp16
                          the station processing alone: gpudifx's three kernels, and the best
                          end-to-end kernel variant writing FP32 spectra, the tensor layout, FP16 spectra
  xmac_gpudifx, xmac_tiled, xmac_transpose, xmac_tensor, xmac_staged   each XMAC stage alone
  chain_gpudifx_today, chain_gpudifx, chain_tiled, chain_transpose_tensor, chain_layout_tensor,
  chain_staged            station + XMAC as one timed chain (README.md, docs/xmac-paths.md)
  staged_over_best        chain_staged / the fastest of the other end-to-end chains (< 1: staged fastest)
  staged_over_layout      chain_staged / chain_layout_tensor
  gain_vs_gpudifx         chain_gpudifx_today / chain_staged
"""
import argparse
import csv
import math
import sys

CHAINS = {
    "station + XMAC: gpudifx today (both)": "chain_gpudifx_today",
    "station + XMAC: end-to-end + gpudifx XMAC": "chain_gpudifx",
    "station + XMAC: end-to-end + tiled": "chain_tiled",
    "station + XMAC: end-to-end + transpose + tensor": "chain_transpose_tensor",
    "station + XMAC: end-to-end -> tensor layout + tensor": "chain_layout_tensor",
    "station + XMAC: end-to-end -> FP16 spectra + staged tensor": "chain_staged",
}
XMACS = {
    "XMAC: gpudifx (FP32 spectra)": "xmac_gpudifx",
    "XMAC: tiled (FP32 spectra)": "xmac_tiled",
    "XMAC: transpose": "xmac_transpose",
    "XMAC: tensor": "xmac_tensor",
    "XMAC: staged tensor (FP16 spectra)": "xmac_staged",
}
STATIONS = [  # (column, name prefix): the best (lowest) of the variants whose name starts so
    ("station_fp32", "fused, "),
    ("station_layout", "end-to-end -> tensor layout, "),
    ("station_fp16", "end-to-end -> FP16 spectra, "),
]
COLS = (["host", "gpu", "cc", "power_w", "sm_clock_lock_mhz", "commit", "date",
         "nbit", "nsub", "nband", "nchan", "nant", "nwin", "subint_ms", "reps", "status", "exit", "note",
         "checks", "checks_failed", "worst_check", "worst_ratio",
         "station_unfused"] + [c for c, _ in STATIONS] + list(XMACS.values()) + list(CHAINS.values()) +
        ["staged_over_best", "staged_over_layout", "gain_vs_gpudifx"])


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def fmt(x, digits=4):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("raw")
    for o in ("host", "commit", "date", "power", "clock"):
        ap.add_argument("--" + o, default="")
    a = ap.parse_args()

    shapes = {}       # (nbit, nsub, nchan, nant) -> fusebench rows
    gpu = {}
    try:
        with open(a.raw, newline="") as f:
            for r in csv.DictReader(f):
                k = (r["nbit"], r["nsub"], r["nchan"], r["nant"])
                shapes.setdefault(k, []).append(r)
                gpu = {"gpu": r["gpu"], "cc": r["cc"]}
    except FileNotFoundError:
        pass

    w = csv.DictWriter(sys.stdout, COLS, lineterminator="\n")
    w.writeheader()
    with open(a.runs, newline="") as f:
        for run in csv.DictReader(f):
            k = (run["nbit"], run["nsub"], run["nchan"], run["nant"])
            rows = shapes.get(k, [])
            out = dict.fromkeys(COLS, "")
            out.update(host=a.host, commit=a.commit, date=a.date, power_w=a.power, sm_clock_lock_mhz=a.clock,
                       nbit=run["nbit"], nsub=run["nsub"], nband=str(2 * int(run["nsub"])), nchan=run["nchan"],
                       nant=run["nant"], nwin=run["nwin"], reps=run["reps"], status=run["status"],
                       exit=run["exit"], note=run["note"], **gpu)
            if rows:
                out["subint_ms"] = rows[0]["subint_ms"]
            checks = [r for r in rows if r["kind"] == "check"]
            out["checks"] = str(len(checks)) if rows else ""
            out["checks_failed"] = ";".join(r["name"] for r in checks if r["ok"] != "1")
            worst = max(checks, key=lambda r: num(r["rms"]) / num(r["limit"]) if num(r["limit"]) > 0 else -1,
                        default=None)
            if worst is not None and num(worst["limit"]) > 0:
                out["worst_check"] = worst["name"]
                out["worst_ratio"] = fmt(num(worst["rms"]) / num(worst["limit"]), 3)
            times = {}
            for r in rows:
                if r["kind"] == "time":
                    times.setdefault(r["name"], num(r["ms_per_ms"]))
            out["station_unfused"] = fmt(times.get("unfused total"))
            for col, prefix in STATIONS:
                v = [t for n, t in times.items() if n.startswith(prefix)]
                out[col] = fmt(min(v)) if v else ""
            for name, col in list(XMACS.items()) + list(CHAINS.items()):
                out[col] = fmt(times.get(name))
            st = times.get("station + XMAC: end-to-end -> FP16 spectra + staged tensor")
            others = [times[n] for n in CHAINS if n in times and "end-to-end" in n and "FP16" not in n]
            if st and others:
                out["staged_over_best"] = fmt(st / min(others), 3)
            lay = times.get("station + XMAC: end-to-end -> tensor layout + tensor")
            if st and lay:
                out["staged_over_layout"] = fmt(st / lay, 3)
            ref = times.get("station + XMAC: gpudifx today (both)")
            if st and ref:
                out["gain_vs_gpudifx"] = fmt(ref / st, 3)
            w.writerow(out)


if __name__ == "__main__":
    main()
