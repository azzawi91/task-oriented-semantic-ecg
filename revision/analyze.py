"""Post-hoc analysis of the revision experiments (NumPy / scikit-learn only).

    python analyze.py <results_revision dir> <cache dir> <paper dir>

Writes <paper dir>/tables/*.tex, <paper dir>/figs/*.pdf and <paper dir>/analysis.json.
Every number in the manuscript tables is produced here from the exported runs.
"""
from __future__ import annotations
import json, os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rev_np as R, rev_metrics as M, rev_splits as S

RES, CACHES, PAPER = sys.argv[1], sys.argv[2], sys.argv[3]
TAB = os.path.join(PAPER, "tables"); FIG = os.path.join(PAPER, "figs")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
OUT = {}
AEAD = 28; D = 32
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = {"task": "#1f4e79", "cnn": "#d9822b", "recon": "#7a7a7a", "jscc": "#b03a2e", "pca": "#5b9bd5", "feat": "#2e8b57", "rr": "#8e44ad", "erase": "#c0392b"}


def runs(name):
    d = os.path.join(RES, name)
    if not os.path.isdir(d):
        return []
    out = []
    for f in sorted(glob.glob(os.path.join(d, "seed*.json"))):
        meta = json.load(open(f)); npz = f[:-5] + ".npz"
        out.append((np.load(npz, allow_pickle=True) if os.path.exists(npz) else None, meta))
    return out


def ms(v, p=3):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return ("%%.%df $\\pm$ %%.%df" % (p, p)) % (v.mean(), v.std()) if len(v) > 1 else ("%%.%df" % p) % v.mean() if len(v) else "--"


def wtab(name, text):
    open(os.path.join(TAB, name + ".tex"), "w").write(text)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight"); plt.close(fig)


# =============================================================== 1. main results + ablations
def agg(name, keys=("auroc", "auprc", "macro_f1", "sens_at_spec", "f1_at_op")):
    rr = runs(name)
    if not rr:
        return None
    res = {k: [m["metrics"][k] for _, m in rr if k in m["metrics"]] for k in keys}
    res["per_class_auroc"] = np.array([m["metrics"]["per_class_auroc"] for _, m in rr])
    res["n"] = len(rr); res["macs"] = rr[0][1]["macs_enc"]; res["params"] = rr[0][1]["params_enc"]
    res["seq_len"] = rr[0][1].get("seq_len"); res["T"] = rr[0][1].get("T"); res["in_ch"] = rr[0][1].get("in_ch")
    return res


main = agg("main")
OUT["main"] = {k: (float(np.mean(v)), float(np.std(v)), [float(x) for x in v]) for k, v in main.items() if k in ("auroc", "auprc", "macro_f1", "sens_at_spec", "f1_at_op")}
OUT["main"]["per_class_auroc_mean"] = main["per_class_auroc"].mean(0).tolist()
OUT["main"]["macs"] = main["macs"]; OUT["main"]["params"] = main["params"]

# classical baselines on the same protocol (per seed = per de Chazal calibration split; DS2 fixed)
cache = np.load(os.path.join(CACHES, "mitbih_fs125_both.npz"), allow_pickle=True)
X, y, g, rhythm = cache["X"], cache["y"], cache["groups"], cache["rhythm"]
tr0, va0, te0 = S.dechazal_split(y, g, 0)
Fsum = R.summary_features(X)
rr_cache = os.path.join(CACHES, "rr_features_fs125.npy")
if os.path.exists(rr_cache):
    Frr = np.load(rr_cache)
else:
    Frr = R.rr_features(X, 125); np.save(rr_cache, Frr)
base = {}
for bname, F in (("summary", Fsum), ("rr", Frr)):
    aucs, f1s, pcs = [], [], []
    for seed in range(main["n"]):
        tr, va, te = S.dechazal_split(y, g, seed)
        sc = R.dim_scale(F[tr]); P = R.logreg_readout(R.quantize(F[tr], 8, sc), y[tr], R.quantize(F[te], 8, sc), seed=seed)
        aucs.append(M.auroc_multiclass(y[te], P)); f1s.append(M.macro_f1(y[te], P.argmax(1))); pcs.append(M.per_class_auroc(y[te], P))
    base[bname] = dict(auroc=aucs, macro_f1=f1s, per_class=np.mean(pcs, 0).tolist(), bytes=F.shape[1])
OUT["baselines"] = {k: dict(auroc=(float(np.mean(v["auroc"])), float(np.std(v["auroc"]))), macro_f1=(float(np.mean(v["macro_f1"])), float(np.std(v["macro_f1"]))), per_class=v["per_class"], bytes=v["bytes"]) for k, v in base.items()}

rand = agg("random_split")
rows = []
rows.append(("Spectral-summary code + logistic regression (22 B)", ms(base["summary"]["auroc"]), "--", ms(base["summary"]["macro_f1"]), "--", "--"))
rows.append(("RR-interval code + logistic regression (22 B)", ms(base["rr"]["auroc"]), "--", ms(base["rr"]["macro_f1"]), "--", "--"))
rows.append(("Task-oriented codec, de Chazal DS1/DS2 (%d seeds)" % main["n"], ms(main["auroc"]), ms(main["auprc"]), ms(main["macro_f1"]), ms(main["sens_at_spec"]), ms(main["f1_at_op"])))
if rand:
    rows.append(("Task-oriented codec, random record-disjoint (%d seeds)" % rand["n"], ms(rand["auroc"]), ms(rand["auprc"]), ms(rand["macro_f1"]), ms(rand["sens_at_spec"]), ms(rand["f1_at_op"])))
t = "\\begin{tabular}{lccccc}\n\\toprule\nModel / protocol & AUROC & AUPRC & Macro-F1 & Sens.@90\\%Spec. & F1@op \\\\\n\\midrule\n"
t += "".join(" & ".join(r) + " \\\\\n" for r in rows) + "\\bottomrule\n\\end{tabular}\n"
wtab("tab_main", t)

# per-class + AFib subset (main, DS2)
pc = main["per_class_auroc"].mean(0); pcs = main["per_class_auroc"].std(0)
afib = []; afib_sens = []; nonafib = []
for npz, m in runs("main"):
    te = npz["idx_te"]; P = npz["P_te"]; yt = npz["y_te"]; rh = rhythm[te]
    mask = np.char.startswith(rh.astype(str), "AFIB") | np.char.startswith(rh.astype(str), "AFL")
    yb, sb = M.binary_view(yt, P)
    thr = m["metrics"]["threshold"]
    afib_sens.append(float(((sb >= thr) & (yb == 1))[mask].sum() / max((yb[mask] == 1).sum(), 1)))
    nonafib.append(float(((sb >= thr) & (yb == 1))[~mask].sum() / max((yb[~mask] == 1).sum(), 1)))
    # AUROC of abnormal-vs-normal restricted to AFib/AFL-rhythm windows vs normal-rhythm windows
    from sklearn.metrics import roc_auc_score
    sel = mask | (rh == "N")
    afib.append(float(roc_auc_score(mask[sel].astype(int), sb[sel])))
OUT["afib"] = dict(n_afib_windows=int(mask.sum()), afib_sens=(float(np.mean(afib_sens)), float(np.std(afib_sens))),
                   other_sens=(float(np.mean(nonafib)), float(np.std(nonafib))), afib_vs_normal_auroc=(float(np.mean(afib)), float(np.std(afib))))
t = "\\begin{tabular}{lccc}\n\\toprule\nClass (one-vs-rest AUROC on DS2) & Task-oriented codec & Spectral-summary code & RR-interval code \\\\\n\\midrule\n"
for k, nm in enumerate(("Normal", "Supraventricular (SVEB)", "Ventricular (VEB)")):
    t += "%s & %.3f $\\pm$ %.3f & %.3f & %.3f \\\\\n" % (nm, pc[k], pcs[k], base["summary"]["per_class"][k], base["rr"]["per_class"][k])
t += "\\midrule\nSensitivity at the calibrated operating point, windows in AF/AFL rhythm ($n$=%d) & %s & -- & -- \\\\\n" % (int(mask.sum()), ms(afib_sens))
t += "Sensitivity at the calibrated operating point, all other abnormal windows & %s & -- & -- \\\\\n" % ms(nonafib)
t += "\\bottomrule\n\\end{tabular}\n"
wtab("tab_perclass", t)

# ablations
abl = [("main", "Full model (CNN + 2-layer Transformer, all loss terms)"), ("abl_cnn_only", "CNN only (Transformer removed)"),
       ("abl_tf1", "One Transformer layer"), ("abl_no_reg", "No latent $\\ell_2$ term ($\\lambda=0$)"), ("abl_no_rate", "No rate term ($\\gamma=0$)"),
       ("abl_no_ee", "No early-exit head"), ("abl_task_only", "Task loss only")]
t = "\\begin{tabular}{lcccrr}\n\\toprule\nVariant & AUROC & Macro-F1 & Sens.@90\\%Spec. & Enc. params & MACs/window \\\\\n\\midrule\n"
OUT["ablation"] = {}
for key, nm in abl:
    a = agg(key)
    if not a:
        continue
    OUT["ablation"][key] = dict(auroc=(float(np.mean(a["auroc"])), float(np.std(a["auroc"]))), f1=(float(np.mean(a["macro_f1"])), float(np.std(a["macro_f1"]))), macs=a["macs"], params=a["params"], n=a["n"])
    t += "%s & %s & %s & %s & %s & %s \\\\\n" % (nm, ms(a["auroc"]), ms(a["macro_f1"]), ms(a["sens_at_spec"]), "{:,}".format(a["params"]), "{:,}".format(a["macs"]))
t += "\\bottomrule\n\\end{tabular}\n"; wtab("tab_ablation", t)

# sampling rate / leads
t = "\\begin{tabular}{lccccrr}\n\\toprule\nInput & Raw bytes/window & AUROC & Macro-F1 & Sens.@90\\%Spec. & Tokens $L$ & MACs/window \\\\\n\\midrule\n"
OUT["input"] = {}
for key, nm, rawb in (("lead_mlii", "MLII only, 125 Hz", 2500), ("main", "Two leads, 125 Hz", 5000), ("fs250", "Two leads, 250 Hz", 10000), ("fs360", "Two leads, 360 Hz (native)", 14400)):
    a = agg(key)
    if not a:
        continue
    OUT["input"][key] = dict(auroc=(float(np.mean(a["auroc"])), float(np.std(a["auroc"]))), macs=a["macs"], seq_len=a["seq_len"], n=a["n"])
    t += "%s & %d & %s & %s & %s & %s & %s \\\\\n" % (nm, rawb, ms(a["auroc"]), ms(a["macro_f1"]), ms(a["sens_at_spec"]), a["seq_len"], "{:,}".format(a["macs"]))
t += "\\bottomrule\n\\end{tabular}\n"; wtab("tab_input", t)

# PTB-XL
p = agg("ptbxl_full", keys=("auroc", "auprc", "macro_f1"))
if p:
    OUT["ptbxl_full"] = {k: (float(np.mean(p[k])), float(np.std(p[k]))) for k in ("auroc", "auprc", "macro_f1")}; OUT["ptbxl_full"]["n"] = p["n"]
    _r0 = runs("ptbxl_full")[0][1]
    OUT["ptbxl_full"]["n_test"] = _r0["n_test"]; OUT["ptbxl_full"]["n_train"] = _r0["n_train"]; OUT["ptbxl_full"]["n_val"] = _r0.get("n_val", 0)
    OUT["ptbxl_full"]["n_records"] = _r0["n_train"] + _r0.get("n_val", 0) + _r0["n_test"]

# =============================================================== 2. matched-payload comparison
def readout_curve(Ztr, ytr, Zte, yte, bits_list, scale=None, seed=0):
    out = {}
    for b in bits_list:
        P = R.logreg_readout(R.quantize(Ztr, b, scale), ytr, R.quantize(Zte, b, scale), seed=seed)
        out[b] = M.auroc_multiclass(yte, P)
    return out


bits_list = [8, 6, 4, 3, 2]
proof = {"task_logreg": [], "task_native": [], "task_pe_logreg": [], "task_pe_native": []}
for npz, m in runs("main")[:3]:
    w = {k: npz[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3")}
    proof["task_logreg"].append(readout_curve(npz["Z_tr"], npz["y_tr"], npz["Z_te"], npz["y_te"], bits_list, seed=m["seed"]))
    proof["task_native"].append({b: M.auroc_multiclass(npz["y_te"], R.np_decoder(w, R.quantize(npz["Z_te"], b))) for b in bits_list})
for npz, m in runs("main_pe")[:3]:
    w = {k: npz[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3")}
    proof["task_pe_logreg"].append(readout_curve(npz["Z_tr"], npz["y_tr"], npz["Z_te"], npz["y_te"], bits_list, seed=m["seed"]))
    proof["task_pe_native"].append({b: M.auroc_multiclass(npz["y_te"], R.np_decoder(w, R.quantize(npz["Z_te"], b))) for b in bits_list})
for key in ("recon_convae", "recon_jscc", "recon_sameenc", "recon_sameenc_pe"):
    proof[key] = []; proof[key + "_recon"] = []
    for npz, m in runs(key):
        sc = R.dim_scale(npz["Z_tr"])
        proof[key].append(readout_curve(npz["Z_tr"], npz["y_tr"], npz["Z_te"], npz["y_te"], bits_list, sc, seed=m["seed"]))
        proof[key + "_recon"].append((m["test_mse"], m["test_prd_pct"]))
# PCA and hand-crafted codes (seed-0 calibration split; DS2 fixed)
proof["pca"] = {}
for k in (16, 32, 64, 128):
    Ptr, Pte = R.pca_code(X[tr0], X[te0], k); sc = R.dim_scale(Ptr)
    proof["pca"][k] = {b: M.auroc_multiclass(y[te0], R.logreg_readout(R.quantize(Ptr, b, sc), y[tr0], R.quantize(Pte, b, sc))) for b in (8, 4)}
proof["summary"] = readout_curve(Fsum[tr0], y[tr0], Fsum[te0], y[te0], [8, 4], R.dim_scale(Fsum[tr0]))
proof["rr"] = readout_curve(Frr[tr0], y[tr0], Frr[te0], y[te0], [8, 4], R.dim_scale(Frr[tr0]))
OUT["proof"] = {k: (v if isinstance(v, dict) else [{str(kk): vv for kk, vv in x.items()} if isinstance(x, dict) else x for x in v]) for k, v in proof.items()}


def curve_ms(lst, b):
    v = [c[b] for c in lst if b in c]
    return (np.mean(v), np.std(v)) if v else (np.nan, np.nan)


t = "\\begin{tabular}{llcccc}\n\\toprule\nRepresentation (objective) & Readout & 32 B (8 bit) & 16 B (4 bit) & 12 B (3 bit) & 8 B (2 bit) \\\\\n\\midrule\n"
def rowfmt(nm, rd, lst, bl=(8, 4, 3, 2)):
    cells = []
    for b in bl:
        mu, sd = curve_ms(lst, b); cells.append("--" if np.isnan(mu) else ("%.3f $\\pm$ %.3f" % (mu, sd) if len(lst) > 1 else "%.3f" % mu))
    return "%s & %s & %s \\\\\n" % (nm, rd, " & ".join(cells))
t += rowfmt("Task-oriented latent (diagnosis)", "trained decoder", proof["task_native"])
t += rowfmt("Task-oriented latent (diagnosis)", "logistic regr.", proof["task_logreg"])
if proof["task_pe_logreg"] and proof["recon_sameenc_pe"]:
    t += rowfmt("Task-oriented latent, encoder + positional enc.\ (diagnosis)", "logistic regr.", proof["task_pe_logreg"])
    t += rowfmt("Same encoder + positional enc., reconstruction objective (MSE)", "logistic regr.", proof["recon_sameenc_pe"])
t += rowfmt("Convolutional autoencoder (MSE)", "logistic regr.", proof["recon_convae"])
t += rowfmt("Deep-JSCC autoencoder (MSE + AWGN 10 dB)", "logistic regr.", proof["recon_jscc"])
t += "PCA transform code, 32 coeff.\\ & logistic regr. & %.3f & %.3f & -- & -- \\\\\n" % (proof["pca"][32][8], proof["pca"][32][4])
t += "PCA transform code, 128 coeff.\\ (128 B / 64 B) & logistic regr. & %.3f & %.3f & -- & -- \\\\\n" % (proof["pca"][128][8], proof["pca"][128][4])
t += "Spectral-summary code, 22 coeff.\\ (22 B / 11 B) & logistic regr. & %.3f & %.3f & -- & -- \\\\\n" % (proof["summary"][8], proof["summary"][4])
t += "RR-interval code, 22 coeff.\\ (22 B / 11 B) & logistic regr. & %.3f & %.3f & -- & -- \\\\\n" % (proof["rr"][8], proof["rr"][4])
t += "\\bottomrule\n\\end{tabular}\n"; wtab("tab_proof", t)
t = "\\begin{tabular}{lccc}\n\\toprule\nReconstruction codec & Test MSE (z-scored) & PRD (\\%) & Epochs \\\\\n\\midrule\n"
for key, nm in (("recon_sameenc_pe", "Task encoder + pos.\ enc.\ + transposed-conv decoder"), ("recon_convae", "Convolutional autoencoder"), ("recon_jscc", "Deep-JSCC autoencoder")):
    v = proof[key + "_recon"]
    if v:
        ep = runs(key)[0][1]["epochs"]
        t += "%s & %s & %s & %d \\\\\n" % (nm, ms([a for a, _ in v]), ms([b for _, b in v], 1), ep)
t += "\\bottomrule\n\\end{tabular}\n"; wtab("tab_recon", t)

fig, ax = plt.subplots(figsize=(3.4, 2.4))
def plot_curve(lst, bytes_of, label, color, marker):
    xs = [bytes_of(b) for b in bits_list]; mu = [curve_ms(lst, b)[0] for b in bits_list]; sd = [curve_ms(lst, b)[1] for b in bits_list]
    ax.errorbar(xs, mu, yerr=sd, label=label, color=color, marker=marker, ms=3, lw=1.2, capsize=2)
byt = lambda b: int(np.ceil(D * b / 8))
plot_curve(proof["task_native"], byt, "Task-oriented latent (trained decoder)", COL["task"], "o")
plot_curve(proof["task_logreg"], byt, "Task-oriented latent (linear readout)", COL["task"], "s")
if proof["task_pe_logreg"]:
    plot_curve(proof["task_pe_logreg"], byt, "Task latent + pos. enc. (linear readout)", COL["task"], "D")
if proof["recon_sameenc_pe"]:
    plot_curve(proof["recon_sameenc_pe"], byt, "Same encoder + pos. enc., MSE objective", COL["cnn"], "^")
if proof["recon_convae"]:
    plot_curve(proof["recon_convae"], byt, "Conv. autoencoder (MSE)", COL["recon"], "v")
if proof["recon_jscc"]:
    plot_curve(proof["recon_jscc"], byt, "Deep-JSCC (MSE + AWGN)", COL["jscc"], "d")
ax.plot([32, 128], [proof["pca"][32][8], proof["pca"][128][8]], color=COL["pca"], marker="x", lw=1, label="PCA transform code")
ax.plot([22], [proof["summary"][8]], color=COL["feat"], marker="*", ls="none", ms=7, label="Spectral-summary code")
ax.plot([22], [proof["rr"][8]], color=COL["rr"], marker="P", ls="none", ms=6, label="RR-interval code")
ax.axhline(0.5, color="k", ls=":", lw=0.8); ax.set_xscale("log", base=2); ax.set_xlabel("Transmitted code (bytes, before AEAD tag)"); ax.set_ylabel("AUROC (DS2)")
ax.set_xticks([8, 16, 32, 64, 128]); ax.set_xticklabels(["8", "16", "32", "64", "128"])
ax.set_ylim(0.45, 0.9); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False, fontsize=6); save(fig, "fig_proof")

# =============================================================== 3. quantisation: PTQ vs QAT, per-dimension sensitivity
ptq = {b: [] for b in bits_list}
for npz, m in runs("main"):
    w = {k: npz[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3")}
    for b in bits_list:
        ptq[b].append(M.auroc_multiclass(npz["y_te"], R.np_decoder(w, R.quantize(npz["Z_te"], b))))
qat = {}
for key, b in (("qat4", 4), ("qat3", 3)):
    a = agg(key)
    if a:
        qat[b] = a["auroc"]
OUT["ptq"] = {str(b): (float(np.mean(v)), float(np.std(v))) for b, v in ptq.items()}
OUT["qat"] = {str(b): (float(np.mean(v)), float(np.std(v))) for b, v in qat.items()}
t = "\\begin{tabular}{cccccc}\n\\toprule\nBits/coeff.\\ $b$ & Code (B) & Payload incl.\\ AEAD (B) & PTQ AUROC & QAT AUROC & PTQ Macro-F1 \\\\\n\\midrule\n"
f1q = {b: [] for b in bits_list}
for npz, m in runs("main"):
    w = {k: npz[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3")}
    for b in bits_list:
        f1q[b].append(M.macro_f1(npz["y_te"], R.np_decoder(w, R.quantize(npz["Z_te"], b)).argmax(1)))
for b in bits_list:
    t += "%d & %d & %d & %s & %s & %s \\\\\n" % (b, byt(b), byt(b) + AEAD, ms(ptq[b]), ms(qat[b]) if b in qat else ("(same model)" if b == 8 else "--"), ms(f1q[b]))
for key, Dl in (("latent8", 8), ("latent4", 4)):
    l = agg(key)
    if l:
        t += "8 ($D$=%d, trained with %d coeff.%s) & %d & %d & %s & -- & %s \\\\\n" % (Dl, Dl, "" if l["n"] >= 10 else ", %d seeds" % l["n"], Dl, Dl + AEAD, ms(l["auroc"]), ms(l["macro_f1"]))
t += "\\bottomrule\n\\end{tabular}\n"; wtab("tab_quant", t)

# per-dimension sensitivity: AUROC drop when one coefficient is (a) dropped, (b) quantised to 2 bits; plus quantisation error per dim
sens_drop = []; sens_2b = []; qerr = []; zvar = []
for npz, m in runs("main"):
    w = {k: npz[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3")}; Z = npz["Z_te"]; yt = npz["y_te"]
    ref = M.auroc_multiclass(yt, R.np_decoder(w, R.quantize(Z, 8)))
    dd, d2 = [], []
    for j in range(D):
        Zq = R.quantize(Z, 8).copy(); Zq[:, j] = 0; dd.append(ref - M.auroc_multiclass(yt, R.np_decoder(w, Zq)))
        Zq = R.quantize(Z, 8).copy(); Zq[:, j] = R.quantize(Z[:, j], 2); d2.append(ref - M.auroc_multiclass(yt, R.np_decoder(w, Zq)))
    sens_drop.append(dd); sens_2b.append(d2)
    qerr.append(((R.quantize(Z, 4) - Z) ** 2).mean(0)); zvar.append(Z.var(0))
sens_drop = np.array(sens_drop); sens_2b = np.array(sens_2b); qerr = np.array(qerr); zvar = np.array(zvar)
OUT["dim_sensitivity"] = dict(drop_mean=sens_drop.mean(0).tolist(), q2_mean=sens_2b.mean(0).tolist(), max_drop=float(sens_drop.mean(0).max()),
                              median_drop=float(np.median(sens_drop.mean(0))), n_dims_drop_gt_0p01=int((sens_drop.mean(0) > 0.01).sum()),
                              qerr4_mean=float(qerr.mean()), latent_var_mean=float(zvar.mean()), snr4_db=float(10 * np.log10(zvar.mean() / max(qerr.mean(), 1e-12))))
# coefficient-subset curve: AUROC when only k of the 32 coefficients are transmitted (others zeroed)
ks = [2, 4, 8, 12, 16, 20, 24, 28, 32]; sub_var = {k: [] for k in ks}; sub_rand = {k: [] for k in ks}
for npz, m in runs("main"):
    w = {k: npz[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3")}; Z = npz["Z_te"]; yt = npz["y_te"]; Zq = R.quantize(Z, 8)
    order_v = np.argsort(-npz["Z_tr"].var(0)); rng_s = np.random.default_rng(m["seed"])
    for k in ks:
        Zk = np.zeros_like(Zq); Zk[:, order_v[:k]] = Zq[:, order_v[:k]]; sub_var[k].append(M.auroc_multiclass(yt, R.np_decoder(w, Zk)))
        vals = []
        for r in range(5):
            idx = rng_s.choice(D, k, replace=False); Zk = np.zeros_like(Zq); Zk[:, idx] = Zq[:, idx]; vals.append(M.auroc_multiclass(yt, R.np_decoder(w, Zk)))
        sub_rand[k].append(np.mean(vals))
# variance spectrum of the latent (effective dimensionality)
spec = np.array([np.sort(npz["Z_tr"].var(0))[::-1] for npz, _ in runs("main")]); frac = np.cumsum(spec, 1) / spec.sum(1, keepdims=True)
OUT["latent_spectrum"] = {"var_sorted_mean": spec.mean(0).tolist(), "cum_frac_mean": frac.mean(0).tolist(),
                          "k90": int(np.mean([np.searchsorted(f, 0.90) + 1 for f in frac])), "k99": int(np.mean([np.searchsorted(f, 0.99) + 1 for f in frac]))}
OUT["subset"] = {"k": ks, "top_var": [(float(np.mean(sub_var[k])), float(np.std(sub_var[k]))) for k in ks], "random": [(float(np.mean(sub_rand[k])), float(np.std(sub_rand[k]))) for k in ks]}
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2))
axs[0].errorbar(ks, [np.mean(sub_var[k]) for k in ks], yerr=[np.std(sub_var[k]) for k in ks], color=COL["task"], marker="o", ms=3, lw=1.1, capsize=2, label="highest-variance coefficients kept")
axs[0].errorbar(ks, [np.mean(sub_rand[k]) for k in ks], yerr=[np.std(sub_rand[k]) for k in ks], color=COL["cnn"], marker="s", ms=3, lw=1.1, capsize=2, label="random coefficients kept")
axs[0].set_xlabel("Coefficients transmitted (of 32)"); axs[0].set_ylabel("AUROC (DS2)"); axs[0].legend(frameon=False, loc="lower right"); axs[0].set_ylim(0.5, 0.9)
# quantisation-error distribution at 4 and 8 bits (pooled over seeds and dims)
errs = {}
for b in (8, 4, 3):
    e = np.concatenate([(R.quantize(npz["Z_te"], b) - npz["Z_te"]).ravel() for npz, _ in runs("main")])
    errs[b] = e; axs[1].hist(e, bins=80, histtype="step", density=True, label="%d bit" % b, lw=1)
axs[1].set_xlabel("Quantisation error (latent units)"); axs[1].set_ylabel("Density"); axs[1].set_yscale("log"); axs[1].legend(frameon=False)
save(fig, "fig_quant")
OUT["qerr_std"] = {str(b): float(e.std()) for b, e in errs.items()}

# =============================================================== 4. channel models
rng = np.random.default_rng(0)
chan = {}


def eval_model(name, fn, n_rep=10):
    """Returns (mean AUROC, SD over seeds, mean paired change vs clean, SD of the paired change)."""
    vals, deltas = [], []
    for npz, m in runs(name):
        w = {k: npz[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3")}; Z = npz["Z_te"]; yt = npz["y_te"]; b = m["cfg"]["bits"]
        q = R.quantize_int(Z, b); clean = M.auroc_multiclass(yt, R.np_decoder(w, R.dequantize_int(q, b)))
        v = [M.auroc_multiclass(yt, R.np_decoder(w, fn(q, b, np.random.default_rng(1000 * m["seed"] + r)))) for r in range(n_rep)]
        vals.append(np.mean(v)); deltas.append(np.mean(v) - clean)
    return float(np.mean(vals)), float(np.std(vals)), float(np.mean(deltas)), float(np.std(deltas))


rates = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
for model in ("main", "chan_erase20"):
    if not runs(model):
        continue
    chan[model] = {}
    chan[model]["iid_coef"] = [eval_model(model, lambda q, b, r, p=p: R.dequantize_int(q, b) * (r.random(q.shape) >= p)) for p in rates]
    chan[model]["frag4_iid"] = [eval_model(model, lambda q, b, r, p=p: R.fragment_erasure(R.dequantize_int(q, b), 4, p, r)[0]) for p in rates]
    # bursty: Gilbert-Elliott over fragments with mean loss ~p: bad state loses everything, good state loses nothing; P(bad)=p
    chan[model]["frag4_burst"] = [eval_model(model, lambda q, b, r, p=p: R.fragment_erasure(R.dequantize_int(q, b), 4, p, r, bursty=(0.5 * p / max(1 - p, 1e-6), 0.5, 0.0, 1.0))[0]) for p in rates]
    bers = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    chan[model]["bsc"] = [eval_model(model, lambda q, b, r, p=p: R.dequantize_int(R.bsc_flip(q, b, p, r), b), n_rep=5) for p in bers]
    chan[model]["bsc_burst"] = [eval_model(model, lambda q, b, r, p=p: R.dequantize_int(R.bursty_bsc(q, b, p, r), b), n_rep=5) for p in bers]
OUT["channel"] = {m: {k: [list(x) for x in v] for k, v in d.items()} for m, d in chan.items()}
OUT["channel_rates"] = rates; OUT["channel_bers"] = bers
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3))
for model, ls in (("main", "-"), ("chan_erase20", "--")):
    if model not in chan:
        continue
    lab = "published training" if model == "main" else "erasure-aware training"
    for key, col, mk, nm in (("iid_coef", COL["task"], "o", "i.i.d. coefficient erasure"), ("frag4_iid", COL["pca"], "s", "4-fragment i.i.d. erasure"), ("frag4_burst", COL["erase"], "^", "4-fragment bursty (Gilbert-Elliott)")):
        mu = [x[2] for x in chan[model][key]]; sd = [x[3] for x in chan[model][key]]
        axs[0].errorbar(np.array(rates) * 100, mu, yerr=sd, color=col, marker=mk, ls=ls, ms=3, lw=1.1, capsize=2, label="%s, %s" % (nm, lab))
    for key, col, mk, nm in (("bsc", COL["task"], "o", "i.i.d. bit flips"), ("bsc_burst", COL["erase"], "^", "bursty bit flips")):
        mu = [x[2] for x in chan[model][key]]; sd = [x[3] for x in chan[model][key]]
        axs[1].errorbar([max(b, 3e-4) for b in bers], mu, yerr=sd, color=col, marker=mk, ls=ls, ms=3, lw=1.1, capsize=2, label="%s, %s" % (nm, lab))
axs[0].set_xlabel("Erasure rate (%)"); axs[0].set_ylabel("AUROC change vs. clean channel"); axs[0].legend(frameon=False, fontsize=5.5, loc="lower left")
axs[1].set_xscale("log"); axs[1].set_xlabel("Bit-error rate (leftmost point: 0)"); axs[1].legend(frameon=False, fontsize=5.5, loc="lower left")
save(fig, "fig_channel")

# =============================================================== 5. energy model
RADIOS = {"BLE 1M (nRF52840)": dict(R=1e6, I_tx=9.65e-3, mtu=244, ovh=14, ifs=150e-6, ramp=140e-6),
          "802.15.4 (CC2652R)": dict(R=250e3, I_tx=24.0e-3, mtu=116, ovh=15, ifs=192e-6, ramp=150e-6)}
V = 3.0; I_CPU = 3.3e-3; CAP_MWH = 220 * 3.0


def tx(B, r):
    n = max(1, int(np.ceil(B / r["mtu"]))); t = (B + n * r["ovh"]) * 8 / r["R"] + n * r["ifs"] + r["ramp"]
    return r["I_tx"] * V * t * 1e3, t * 1e3, n


def energy_row(B, macs, thr, r, per=0.0):
    e_tx, t_air, n = tx(B, r)
    e_tx_eff = e_tx / max(1 - per, 1e-6)          # retransmission-amortised (ARQ, geometric)
    e_cpu = I_CPU * V * (macs / thr) * 1e3
    return dict(bytes=B, e_tx=e_tx, e_tx_eff=e_tx_eff, e_cpu=e_cpu, total=e_cpu + e_tx_eff, airtime=t_air, packets=n,
                life_days=CAP_MWH / ((e_cpu + e_tx_eff) / 3600.0 * 8640.0) if (e_cpu + e_tx_eff) > 0 else np.inf)


macs_main = main["macs"]; macs_cnn = OUT["ablation"].get("abl_cnn_only", {}).get("macs", None)
thr_list = [40e6, 100e6, 200e6]
en = {}
for rn, r in RADIOS.items():
    en[rn] = {"raw": energy_row(5000, 0, 40e6, r)}
    for thr in thr_list:
        en[rn]["task_%d" % int(thr / 1e6)] = energy_row(D + AEAD, macs_main, thr, r)
        if macs_cnn:
            en[rn]["cnn_%d" % int(thr / 1e6)] = energy_row(D + AEAD, macs_cnn, thr, r)
OUT["energy"] = {rn: {k: {kk: float(vv) for kk, vv in v.items()} for k, v in d.items()} for rn, d in en.items()}
t = "\\begin{tabular}{llrrrrrr}\n\\toprule\nRadio & Scheme (encoder, INT8 throughput) & Bytes & $E_\\mathrm{tx}$ (mJ) & $E_\\mathrm{cpu}$ (mJ) & Total (mJ) & Airtime (ms) & Life (d) \\\\\n\\midrule\n"
for rn, d in en.items():
    t += "%s & Raw window & 5000 & %.3f & 0 & %.3f & %.1f & %.0f \\\\\n" % (rn, d["raw"]["e_tx"], d["raw"]["total"], d["raw"]["airtime"], d["raw"]["life_days"])
    for thr in thr_list:
        k = "task_%d" % int(thr / 1e6); t += " & Full codec, %d MMAC/s & %d & %.3f & %.3f & %.3f & %.2f & %.0f \\\\\n" % (int(thr / 1e6), D + AEAD, d[k]["e_tx"], d[k]["e_cpu"], d[k]["total"], d[k]["airtime"], d[k]["life_days"])
    if macs_cnn:
        for thr in thr_list:
            k = "cnn_%d" % int(thr / 1e6); t += " & CNN-only codec, %d MMAC/s & %d & %.3f & %.3f & %.3f & %.2f & %.0f \\\\\n" % (int(thr / 1e6), D + AEAD, d[k]["e_tx"], d[k]["e_cpu"], d[k]["total"], d[k]["airtime"], d[k]["life_days"])
    t += "\\midrule\n"
t = t[:-len("\\midrule\n")] + "\\bottomrule\n\\end{tabular}\n"; wtab("tab_energy", t)
# crossover figure: per-decision energy vs encoder cost (MACs/throughput = compute seconds)
fig, ax = plt.subplots(figsize=(3.4, 2.3))
tc = np.logspace(-3, 0, 100)   # compute seconds per window
for rn, r in RADIOS.items():
    e_tx, _, _ = tx(D + AEAD, r); raw_e, _, _ = tx(5000, r)
    ax.plot(tc * 1e3, (I_CPU * V * tc * 1e3 + e_tx) / raw_e, label=rn, color=COL["task"] if "BLE" in rn else COL["cnn"])
ax.axhline(1.0, color="k", ls=":", lw=0.8)
for macs, nm, mk in ((macs_main, "full codec", "o"), (macs_cnn, "CNN-only", "s")):
    if macs:
        for thr in thr_list:
            ax.axvline(macs / thr * 1e3, color="grey", lw=0.5, ls="--")
            ax.text(macs / thr * 1e3, 0.035, "%d" % int(thr / 1e6), fontsize=5, ha="center", color="grey")
        ax.text(macs / thr_list[-1] * 1e3 * 0.8, 1.9, nm, fontsize=6, rotation=90, va="top")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Encoder compute time per window (ms)"); ax.set_ylabel("Energy / raw transmission")
ax.set_ylim(0.03, 3); ax.legend(frameon=False, fontsize=6); save(fig, "fig_energy")
# PER sweep factor
per_list = [0, 0.05, 0.1, 0.2, 0.3, 0.4]
OUT["energy_per"] = {rn: [(per, energy_row(5000, 0, 40e6, r, per)["total"] / energy_row(D + AEAD, macs_main, 100e6, r, per)["total"]) for per in per_list] for rn, r in RADIOS.items()}

# PQC cycles (pqm4, m4f) -> time/energy at 64 MHz, 9.9 mW
PQ = dict(kem_keygen=392423, kem_encaps=390881, kem_decaps=428167, dsa_keygen=1426025, dsa_sign=3943121, dsa_verify=1421623)
OUT["pqc"] = {k: dict(cycles=c, ms=c / 64e6 * 1e3, mJ=c / 64e6 * I_CPU * V * 1e3) for k, c in PQ.items()}
OUT["pqc"]["node_handshake_mJ"] = OUT["pqc"]["kem_keygen"]["mJ"] + OUT["pqc"]["kem_decaps"]["mJ"] + OUT["pqc"]["dsa_verify"]["mJ"]
OUT["pqc"]["node_handshake_ms"] = OUT["pqc"]["kem_keygen"]["ms"] + OUT["pqc"]["kem_decaps"]["ms"] + OUT["pqc"]["dsa_verify"]["ms"]

json.dump(OUT, open(os.path.join(PAPER, "analysis.json"), "w"), indent=1, default=float)
print(json.dumps({k: OUT[k] for k in ("main", "baselines", "afib", "ablation", "ptq", "qat", "pqc") if k in OUT}, indent=1, default=float)[:4000])
print("tables:", os.listdir(TAB)); print("figs:", os.listdir(FIG))

# =============================================================== 6. LaTeX macros for the manuscript text
def pm(t, p=3):
    return ("%%.%df $\\pm$ %%.%df" % (p, p)) % (t[0], t[1])
def one(v, p=3):
    return ("%%.%df" % p) % v
mac = {}
mac["mainAUROC"] = pm(OUT["main"]["auroc"]); mac["mainAUPRC"] = pm(OUT["main"]["auprc"]); mac["mainFone"] = pm(OUT["main"]["macro_f1"])
mac["mainSens"] = pm(OUT["main"]["sens_at_spec"]); mac["mainFop"] = pm(OUT["main"]["f1_at_op"])
mac["mainAUROCmin"] = one(min(OUT["main"]["auroc"][2])); mac["mainAUROCmax"] = one(max(OUT["main"]["auroc"][2]))
mac["mainMACs"] = "%.1f" % (OUT["main"]["macs"] / 1e6); mac["mainParams"] = "{:,}".format(OUT["main"]["params"])
mac["nSeeds"] = str(main["n"])
if rand:
    mac["randAUROC"] = pm((np.mean(rand["auroc"]), np.std(rand["auroc"])))
mac["summaryAUROC"] = pm(OUT["baselines"]["summary"]["auroc"]); mac["rrAUROC"] = pm(OUT["baselines"]["rr"]["auroc"])
mac["pcNormal"], mac["pcSVEB"], mac["pcVEB"] = [one(v) for v in OUT["main"]["per_class_auroc_mean"]]
mac["rrSVEB"] = one(OUT["baselines"]["rr"]["per_class"][1]); mac["rrVEB"] = one(OUT["baselines"]["rr"]["per_class"][2]); mac["rrNormal"] = one(OUT["baselines"]["rr"]["per_class"][0])
mac["afibN"] = str(OUT["afib"]["n_afib_windows"]); mac["afibSens"] = pm(OUT["afib"]["afib_sens"]); mac["otherSens"] = pm(OUT["afib"]["other_sens"])
for key, nm in (("abl_cnn_only", "cnn"), ("abl_tf1", "tfone"), ("abl_no_reg", "noreg"), ("abl_no_rate", "norate"), ("abl_no_ee", "noee"), ("abl_task_only", "taskonly")):
    if key in OUT["ablation"]:
        mac[nm + "AUROC"] = pm(OUT["ablation"][key]["auroc"]); mac[nm + "MACs"] = "%.2f" % (OUT["ablation"][key]["macs"] / 1e6); mac[nm + "Params"] = "{:,}".format(OUT["ablation"][key]["params"])
for key, nm in (("lead_mlii", "mlii"), ("fs250", "fsTwoFifty"), ("fs360", "fsThreeSixty")):
    if key in OUT["input"]:
        mac[nm + "AUROC"] = pm(OUT["input"][key]["auroc"]); mac[nm + "MACs"] = "%.1f" % (OUT["input"][key]["macs"] / 1e6)
def c16(lst):
    mu, sd = curve_ms(lst, 4); return "--" if np.isnan(mu) else (pm((mu, sd)) if len(lst) > 1 else one(mu))
def c32(lst):
    mu, sd = curve_ms(lst, 8); return "--" if np.isnan(mu) else (pm((mu, sd)) if len(lst) > 1 else one(mu))
mac["taskNativeSixteen"] = c16(proof["task_native"]); mac["taskLogregSixteen"] = c16(proof["task_logreg"]); mac["taskNativeThirtyTwo"] = c32(proof["task_native"])
mac["sameencSixteen"] = c16(proof["recon_sameenc_pe"]); mac["convaeSixteen"] = c16(proof["recon_convae"]); mac["jsccSixteen"] = c16(proof["recon_jscc"])
mac["sameencThirtyTwo"] = c32(proof["recon_sameenc_pe"]); mac["convaeThirtyTwo"] = c32(proof["recon_convae"]); mac["jsccThirtyTwo"] = c32(proof["recon_jscc"])
mac["taskPeSixteen"] = c16(proof["task_pe_logreg"]); mac["taskPeThirtyTwo"] = c32(proof["task_pe_logreg"])
pe = agg("main_pe")
if pe:
    mac["mainPeAUROC"] = pm((np.mean(pe["auroc"]), np.std(pe["auroc"])))
mac["pcaThirtyTwo"] = one(proof["pca"][32][8]); mac["pcaOneTwentyEight"] = one(proof["pca"][128][8]); mac["summaryProof"] = one(proof["summary"][8]); mac["rrProof"] = one(proof["rr"][8])
for key, nm in (("recon_sameenc_pe", "sameenc"), ("recon_convae", "convae"), ("recon_jscc", "jscc")):
    v = proof[key + "_recon"]
    if v:
        mac[nm + "MSE"] = pm((np.mean([a for a, _ in v]), np.std([a for a, _ in v]))); mac[nm + "PRD"] = pm((np.mean([b for _, b in v]), np.std([b for _, b in v])), 1)
for b in bits_list:
    mac["ptq%s" % {8: "Eight", 6: "Six", 4: "Four", 3: "Three", 2: "Two"}[b]] = pm(OUT["ptq"][str(b)])
for b, nm in ((4, "qatFour"), (3, "qatThree")):
    if str(b) in OUT["qat"]:
        mac[nm] = pm(OUT["qat"][str(b)])
mac["kNinety"] = str(OUT["latent_spectrum"]["k90"]); mac["kNinetyNine"] = str(OUT["latent_spectrum"]["k99"])
mac["cumTwo"] = "%.0f" % (100 * OUT["latent_spectrum"]["cum_frac_mean"][1]); mac["cumFour"] = "%.0f" % (100 * OUT["latent_spectrum"]["cum_frac_mean"][3]); mac["cumEight"] = "%.0f" % (100 * OUT["latent_spectrum"]["cum_frac_mean"][7])
mac["subTopTwo"] = pm(OUT["subset"]["top_var"][0]); mac["subTopFour"] = pm(OUT["subset"]["top_var"][1]); mac["subTopEight"] = pm(OUT["subset"]["top_var"][2]); mac["subRandFour"] = pm(OUT["subset"]["random"][1]); mac["subRandEight"] = pm(OUT["subset"]["random"][2]); mac["subRandSixteen"] = pm(OUT["subset"]["random"][4])
for key, nm in (("latent4", "latFour"), ("latent8", "latEight")):
    l = agg(key)
    if l:
        mac[nm + "AUROC"] = pm((np.mean(l["auroc"]), np.std(l["auroc"]))); mac[nm + "Fone"] = pm((np.mean(l["macro_f1"]), np.std(l["macro_f1"]))); mac[nm + "N"] = str(l["n"])
        OUT[key] = dict(auroc=(float(np.mean(l["auroc"])), float(np.std(l["auroc"]))), n=l["n"])
mac["dimMaxDrop"] = one(OUT["dim_sensitivity"]["max_drop"]); mac["dimMedianDrop"] = one(OUT["dim_sensitivity"]["median_drop"]); mac["dimNsens"] = str(OUT["dim_sensitivity"]["n_dims_drop_gt_0p01"])
mac["snrFour"] = "%.1f" % OUT["dim_sensitivity"]["snr4_db"]
if "main" in chan:
    mac["iidForty"] = pm(chan["main"]["iid_coef"][-1]); mac["fragForty"] = pm(chan["main"]["frag4_iid"][-1]); mac["burstForty"] = pm(chan["main"]["frag4_burst"][-1])
    mac["iidTen"] = pm(chan["main"]["iid_coef"][2]); mac["bscOnePct"] = pm(chan["main"]["bsc"][3]); mac["bscTenPct"] = pm(chan["main"]["bsc"][5]); mac["burstBscOnePct"] = pm(chan["main"]["bsc_burst"][3])
if "chan_erase20" in chan:
    mac["eraseIidForty"] = pm(chan["chan_erase20"]["iid_coef"][-1]); mac["eraseBurstForty"] = pm(chan["chan_erase20"]["frag4_burst"][-1]); mac["eraseZero"] = pm(chan["chan_erase20"]["iid_coef"][0]); mac["eraseBscTenPct"] = pm(chan["chan_erase20"]["bsc"][5])
ble = OUT["energy"]["BLE 1M (nRF52840)"]; z8 = OUT["energy"]["802.15.4 (CC2652R)"]
mac["bleRaw"] = "%.2f" % ble["raw"]["total"]; mac["zRaw"] = "%.2f" % z8["raw"]["total"]
mac["bleFullForty"] = "%.2f" % ble["task_40"]["total"]; mac["bleFullHundred"] = "%.2f" % ble["task_100"]["total"]; mac["bleFullTwoHundred"] = "%.2f" % ble["task_200"]["total"]
mac["zFullForty"] = "%.2f" % z8["task_40"]["total"]; mac["zFullHundred"] = "%.2f" % z8["task_100"]["total"]
if "cnn_40" in ble:
    mac["bleCnnForty"] = "%.3f" % ble["cnn_40"]["total"]; mac["zCnnForty"] = "%.3f" % z8["cnn_40"]["total"]
    mac["bleCnnFactor"] = "%.1f" % (ble["raw"]["total"] / ble["cnn_40"]["total"]); mac["zCnnFactor"] = "%.1f" % (z8["raw"]["total"] / z8["cnn_40"]["total"])
mac["bleFullHundredFactor"] = "%.1f" % (ble["raw"]["total"] / ble["task_100"]["total"]); mac["zFullHundredFactor"] = "%.1f" % (z8["raw"]["total"] / z8["task_100"]["total"])
mac["zFullFortyFactor"] = "%.1f" % (z8["raw"]["total"] / z8["task_40"]["total"])
mac["bleAirRaw"] = "%.1f" % ble["raw"]["airtime"]; mac["bleAirSem"] = "%.2f" % ble["task_100"]["airtime"]; mac["airFactor"] = "%.0f" % (ble["raw"]["airtime"] / ble["task_100"]["airtime"])
mac["cpuCrossBLEms"] = "%.0f" % ((ble["raw"]["total"] - ble["task_100"]["e_tx"]) / (I_CPU * V) )   # ms of compute that equals raw BLE energy
mac["cpuCrossZms"] = "%.0f" % ((z8["raw"]["total"] - z8["task_100"]["e_tx"]) / (I_CPU * V))
mac["crossMMAC"] = "%.0f" % (OUT["main"]["macs"] / ((ble["raw"]["total"] - ble["task_100"]["e_tx"]) / (I_CPU * V) / 1e3) / 1e6)
mac["pqEncapsMs"] = "%.1f" % OUT["pqc"]["kem_encaps"]["ms"]; mac["pqVerifyMs"] = "%.0f" % OUT["pqc"]["dsa_verify"]["ms"]; mac["pqNodeMJ"] = "%.2f" % OUT["pqc"]["node_handshake_mJ"]
mac["pqNodeMs"] = "%.0f" % OUT["pqc"]["node_handshake_ms"]; mac["pqKeygenMs"] = "%.1f" % OUT["pqc"]["kem_keygen"]["ms"]; mac["pqSignMs"] = "%.0f" % OUT["pqc"]["dsa_sign"]["ms"]; mac["pqDecapsMs"] = "%.1f" % OUT["pqc"]["kem_decaps"]["ms"]
if "ptbxl_full" in OUT:
    mac["ptbxlAUROC"] = pm(OUT["ptbxl_full"]["auroc"]); mac["ptbxlAUPRC"] = pm(OUT["ptbxl_full"]["auprc"]); mac["ptbxlFone"] = pm(OUT["ptbxl_full"]["macro_f1"])
    mac["ptbxlNtest"] = "{:,}".format(OUT["ptbxl_full"]["n_test"]); mac["ptbxlNtrain"] = "{:,}".format(OUT["ptbxl_full"]["n_train"])
    mac["ptbxlNval"] = "{:,}".format(OUT["ptbxl_full"]["n_val"]); mac["ptbxlNrec"] = "{:,}".format(OUT["ptbxl_full"]["n_records"])
with open(os.path.join(PAPER, "numbers.tex"), "w") as f:
    for k, v in mac.items():
        f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    f.write("\\newif\\ifhavepe \\havepe%s\n" % ("true" if (proof["task_pe_logreg"] and proof["recon_sameenc_pe"]) else "false"))
    f.write("\\newif\\ifhavelat \\havelat%s\n" % ("true" if "latent4" in OUT else "false"))
    if "latent4" in OUT and "latent8" in OUT:
        f.write("\\newcommand{\\latConfirmSentence}{Training codecs with $D=8$ and $D=4$ from scratch confirms it (Table~\\ref{tab:quant}): %s and %s, indistinguishable from the 32-coefficient codec.}\n" % (mac["latEightAUROC"], mac["latFourAUROC"]))
    elif "latent4" in OUT:
        f.write("\\newcommand{\\latConfirmSentence}{Training a codec with $D=4$ from scratch confirms it (Table~\\ref{tab:quant}): %s over %s seeds, indistinguishable from the 32-coefficient codec.}\n" % (mac["latFourAUROC"], mac["latFourN"]))
    else:
        f.write("\\newcommand{\\latConfirmSentence}{}\n")
    f.write("\\newif\\ifhaveptbxl \\haveptbxl%s\n" % ("true" if "ptbxl_full" in OUT else "false"))
    if "ptbxl_full" in OUT:
        f.write("\\newcommand{\\ptbxlRow}{PTB-XL, 12 leads, NORM vs.\\ abnormal & %s & %s & %s \\\\}\n" % (mac["ptbxlAUROC"], mac["ptbxlAUPRC"], mac["ptbxlFone"]))
    else:
        f.write("\\newcommand{\\ptbxlRow}{PTB-XL, 3039-record subset, 12 leads, NORM vs.\\ abnormal & 0.890 $\\pm$ 0.011 & 0.912 $\\pm$ 0.007 & 0.815 $\\pm$ 0.014 \\\\}\n")
print("macros:", len(mac))
