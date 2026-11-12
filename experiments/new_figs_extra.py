"""Three additional figures: wavelet reconstructions of one window, per-seed results, external validation."""
import os as _os
WORK = _os.environ.get("ECG_WORK", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "work"))  # data, runs and outputs live here
import json, os, sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import rev_splits as S
FIG = (WORK + "/paper/figs"); E = json.load(open((WORK + "/new_eval2.json"))); C = E["cfg"]; L = E["lodo"]
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5, "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})
BL, OR, GY, PU, LB = "#1f4e79", "#d9822b", "#7a7a7a", "#8e44ad", "#5b9bd5"
def save(fig, n): fig.savefig(os.path.join(FIG, n + ".pdf"), bbox_inches="tight"); fig.savefig(os.path.join(FIG, n + ".png"), dpi=150, bbox_inches="tight"); plt.close(fig)
# ---- A. one DS2 window with ventricular ectopy and its wavelet reconstructions
d = np.load("caches/pool_all.npz", allow_pickle=True); g = d["groups"].astype(str); y = d["y"]
idx = np.where((g == "200") & (y == 2))[0][12]; x = d["X"][idx, 0]; t = np.arange(len(x)) / 125.0
fig, axs = plt.subplots(4, 1, figsize=(6.6, 4.2), sharex=True)
axs[0].plot(t, x, color="black", lw=0.6); axs[0].set_title("Original window, lead 1 (MIT-BIH record 200, DS2; raw size 5000 bytes for two leads)", fontsize=7, loc="left")
for ax, B in zip(axs[1:], (256, 64, 16)):
    xr = np.load("caches/pool_dwt_%d.npz" % B)["X"][idx, 0]; prd = 100 * np.sqrt(((x - xr) ** 2).sum() / (x ** 2).sum())
    ax.plot(t, x, color="#cccccc", lw=0.6); ax.plot(t, xr, color=OR, lw=0.7)
    ax.set_title("Wavelet coder, %d bytes per window (PRD of this lead %.0f%%); classifier AUROC on DS2 after decoding: %.3f" % (B, prd, E["wavelet"][str(B)]["deployed"][0]), fontsize=7, loc="left")
for ax in axs: ax.set_yticks([]); ax.spines["left"].set_visible(False)
axs[-1].set_xlabel("Time (s)"); fig.tight_layout(h_pad=0.4); save(fig, "fig_recon")
# ---- B. per-seed results: effect of training data and of code size
fig, ax = plt.subplots(figsize=(6.2, 2.6))
groups = [("ds1_cnn_D4", "DS1 only\nlight, 4 B", GY), ("main", "DS1 only\nTransformer, 32 B", GY), ("pool_all_cnn_D1", "3 databases\nlight, 1 B", LB), ("pool_all_cnn_D2", "3 databases\nlight, 2 B", LB),
          ("pool_all_cnn_D4", "3 databases\nlight, 4 B", BL), ("pool_all_cnn_D32", "3 databases\nlight, 32 B", LB), ("pool_dwt16_cnn", "3 databases\nwavelet 16 B", OR), ("pool_dwt256_cnn", "3 databases\nwavelet 256 B", OR)]
rng = np.random.default_rng(0)
for i, (k, nm, col) in enumerate(groups):
    v = np.array(C[k]["seeds"]); ax.scatter(i + rng.uniform(-0.13, 0.13, len(v)), v, s=14, color=col, alpha=0.75, zorder=3, edgecolor="none")
    ax.plot([i - 0.25, i + 0.25], [v.mean(), v.mean()], color="black", lw=1.2, zorder=4)
ax.set_xticks(range(len(groups))); ax.set_xticklabels([n for _, n, _ in groups], fontsize=6); ax.set_ylabel("AUROC on DS2"); ax.set_ylim(0.62, 0.94); ax.grid(axis="y", lw=0.3, color="#dddddd")
save(fig, "fig_seeds")
# ---- C. external validation: learned 4-byte code against the RR-interval code
fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.5), sharey=True)
P4 = C["pool_all_cnn_D4"]; dbs = ["DS2\n(test records)", "INCART\n(held out)", "Supraventricular\n(held out)"]
three = [(P4["auroc"], L["ds2_rr_pool_all"]["auroc"]), (L["incart"]["task_D4"]["auroc"], L["incart"]["rr_pool"]["auroc"]), (L["svdb"]["task_D4"]["auroc"], L["svdb"]["rr_pool"]["auroc"])]
binr = [([P4["per_class"][0], P4["per_class_sd"][0]], L["ds2_rr_pool_all"]["abn_auroc"]), (L["incart"]["task_D4"]["abn_auroc"], L["incart"]["rr_pool"]["abn_auroc"]), (L["svdb"]["task_D4"]["abn_auroc"], L["svdb"]["rr_pool"]["abn_auroc"])]
for ax, data, ttl in ((axs[0], three, "Three classes (macro AUROC)"), (axs[1], binr, "Abnormal versus normal (AUROC)")):
    xs = np.arange(3)
    ax.bar(xs - 0.19, [a[0][0] for a in data], 0.36, yerr=[a[0][1] for a in data], color=BL, capsize=2, label="Learned task code, 4 bytes")
    ax.bar(xs + 0.19, [a[1][0] for a in data], 0.36, yerr=[a[1][1] for a in data], color=PU, capsize=2, label="RR-interval code, 22 bytes")
    for i, a in enumerate(data):
        ax.text(i - 0.19, a[0][0] + a[0][1] + 0.008, "%.2f" % a[0][0], ha="center", fontsize=6); ax.text(i + 0.19, a[1][0] + a[1][1] + 0.008, "%.2f" % a[1][0], ha="center", fontsize=6)
    ax.set_xticks(xs); ax.set_xticklabels(dbs, fontsize=6.5); ax.set_title(ttl, fontsize=7.5); ax.set_ylim(0.5, 1.02)
axs[0].set_ylabel("AUROC"); axs[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(1.05, -0.2), ncol=2); save(fig, "fig_xdb")
print("ok")
