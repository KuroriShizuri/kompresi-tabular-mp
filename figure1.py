"""Build Figure 1 (accuracy change vs. batch-1024 speedup) from figure1_data.csv.

Usage: python figure1.py [--out Figure1.png] [--no-jitter]

Data come from the TOST summary tables printed by cell 8 of rancanganpublikasi.ipynb
(speedup: seed 42, one thread; accuracy change and CI95: mean over five seeds).
Horizontal offsets are display-only: each method is shifted by a fixed fraction of
the axis width, identical in every panel. Data values are not modified.
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
METHODS = [  # name, colour (Okabe-Ito), marker, offset as fraction of axis width
    ("Proposed (zero-shot)", "#D55E00", "o", -0.03),
    ("EVR, same budget", "#0072B2", "s", +0.02),
    ("Uniform rank", "#009E73", "^", -0.03),
    ("Magnitude pruning", "#E69F00", "D", -0.01),
    ("Small model from scratch", "#CC79A7", "P", +0.01),
]
DATASETS = ["Adult Income", "Bank Marketing", "Credit Default", "HIGGS (98k)"]
REGIMES = ["Standard", "Regularized"]
X_PAD = 0.075
MARKER_SIZE = 6.0


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["dataset"], r["regime"], r["method"]): [float(r[k]) for k in
                ("speedup_bs1024", "dacc_pp", "ci95_low", "ci95_high")]
                for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "figure1_data.csv"))
    ap.add_argument("--out", default=str(HERE / "Figure1.png"))
    ap.add_argument("--no-jitter", action="store_true")
    a = ap.parse_args()
    data = load(a.data)

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.9))
    for i, (ax, ds) in enumerate(zip(axes.ravel(), DATASETS)):
        pts = [data[(ds, r, m[0])] for r in REGIMES for m in METHODS]
        xs = [p[0] for p in pts] + [1.0]
        lo_x, hi_x = min(xs), max(xs)
        pad = (hi_x - lo_x) * X_PAD
        xlim = (lo_x - pad, hi_x + pad)
        width = xlim[1] - xlim[0]
        ys = [p[2] for p in pts] + [p[3] for p in pts] + [-0.5, 0.5]
        ypad = (max(ys) - min(ys)) * 0.06
        ax.set_xlim(*xlim)
        ax.set_ylim(min(ys) - ypad, max(ys) + ypad)
        ax.axhspan(-0.5, 0.5, color="#e0e0e0", zorder=0, lw=0)
        ax.axhline(0, color="#777777", lw=0.8, zorder=1)
        ax.axvline(1.0, color="#666666", lw=0.8, ls="--", zorder=1)
        ax.grid(True, color="#eeeeee", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for r in REGIMES:
            for name, colour, marker, off in METHODS:
                s, d, lo, hi = data[(ds, r, name)]
                x = s if a.no_jitter else s + off * width
                ax.vlines(x, lo, hi, color=colour, lw=1.2, alpha=0.55, zorder=2)
                filled = r == "Standard"
                ax.plot(x, d, ls="none", marker=marker, ms=MARKER_SIZE + (1 if marker == "P" else 0),
                        mfc=colour if filled else "white", mec=colour, mew=1.5, zorder=3)
        ax.set_title(f"({'abcd'[i]}) {ds}", loc="left", fontsize=9)
        ax.set_xlabel("Speedup at batch size 1024 (\u00d7)", fontsize=8.5)
        ax.set_ylabel("\u0394 accuracy vs. baseline (pp)", fontsize=8.5)
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    handles = [Line2D([], [], ls="none", marker=m, ms=7, mfc=c, mec=c, label=n)
               for n, c, m, _ in METHODS]
    handles += [Line2D([], [], ls="none", marker="o", ms=6.5, mfc="#333333", mec="#333333", label="Standard regime"),
                Line2D([], [], ls="none", marker="o", ms=6.5, mfc="white", mec="#333333", mew=1.5,
                       label="Regularized regime")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.5, columnspacing=1.1, handletextpad=0.4,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(a.out, dpi=300)


if __name__ == "__main__":
    main()



