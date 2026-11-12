"""Concept figure for the manuscript (pure matplotlib; no external artwork)."""
import os as _os
WORK = _os.environ.get("ECG_WORK", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "work"))  # data, runs and outputs live here
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
d = np.load("caches/mitbih_fs125_both.npz", allow_pickle=True); x = d["X"][np.where(d["y"] == 2)[0][40], 0]
fig, ax = plt.subplots(figsize=(7.2, 3.3)); ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")
BL, GR, OR, GY = "#1f4e79", "#2e7d5b", "#b85c1e", "#555555"
def box(x0, y0, w, h, txt, col, fs=7.2, fc="white"):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", lw=1.1, ec=col, fc=fc))
    ax.text(x0 + w / 2, y0 + h / 2, txt, ha="center", va="center", fontsize=fs, color="black", linespacing=1.25)
def arr(x0, y0, x1, y1, col=GY, txt=None, dy=1.6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=1.1, color=col))
    if txt: ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, txt, ha="center", va="bottom", fontsize=6.6, color=col, fontweight="bold")
# ECG trace (shared input)
t = np.linspace(2, 16, len(x)); 
for yc in (33.5, 10.5):
    ax.plot(t, yc + 2.0 * x / np.abs(x).max(), color="black", lw=0.5)
    ax.text(9, yc - 5.3, "10-s window, 2 leads\n5000 bytes raw", ha="center", va="top", fontsize=6.6)
ax.text(1, 44.5, "(a) Reconstruction-oriented compression", fontsize=8, fontweight="bold", color=OR, va="center")
ax.text(1, 21.5, "(b) Task-oriented compression (this work)", fontsize=8, fontweight="bold", color=BL, va="center")
box(21, 29.5, 15, 8, "Transform coder\n(wavelet / DCT)\nor autoencoder", OR); arr(16.6, 33.5, 20.6, 33.5)
box(47, 29.5, 15, 8, "Decoder:\nreconstructed\nwaveform", OR); arr(36.6, 33.5, 46.6, 33.5, OR, "16-1024 B")
box(73, 29.5, 14, 8, "Diagnostic\nmodel", OR); arr(62.6, 33.5, 72.6, 33.5)
arr(87.6, 33.5, 93, 33.5); ax.text(96.5, 33.5, "Class", ha="center", va="center", fontsize=7.2)
ax.text(54.5, 27.3, "judged by PRD", ha="center", va="top", fontsize=6.4, color=GY, style="italic")
box(20, 6.5, 16.6, 8, "Small encoder\nCNN + Transformer\n42k parameters", BL, fs=6.8); arr(16.2, 10.5, 19.6, 10.5)
box(42, 6.5, 12, 8, "Quantizer\n$D$ coeff.\n$\\times$ $b$ bits", BL); arr(36.6, 10.5, 41.6, 10.5)
box(73, 6.5, 14, 8, "Small decoder\n(perceptron)", BL); arr(54.6, 10.5, 72.6, 10.5, BL, "1 bit - 32 B")
arr(87.6, 10.5, 93, 10.5); ax.text(96.5, 10.5, "Class", ha="center", va="center", fontsize=7.2)
ax.text(63.5, 7.6, "no waveform is rebuilt;\njudged by the decision", ha="center", va="top", fontsize=6.4, color=GY, style="italic")
ax.plot([0, 100], [23.8, 23.8], color="#cccccc", lw=0.6, ls="--")
fig.savefig((WORK + "/paper/figs/fig_concept.pdf"), bbox_inches="tight"); fig.savefig((WORK + "/paper/figs/fig_concept.png"), dpi=170, bbox_inches="tight")
