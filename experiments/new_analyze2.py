"""Tables, figures and LaTeX macros of the manuscript from new_eval2.json (+ new_eval.json, paper_old/analysis.json).
    python new_analyze2.py <paper dir>"""
import os as _os
WORK = _os.environ.get("ECG_WORK", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "work"))  # data, runs and outputs live here
import json, os, sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
PAPER = sys.argv[1]; TAB = os.path.join(PAPER, "tables"); FIG = os.path.join(PAPER, "figs"); os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
E = json.load(open((WORK + "/new_eval2.json"))); E1 = json.load(open((WORK + "/new_eval.json"))); A = json.load(open((WORK + "/paper_old/analysis.json")))
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5, "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})
BL, OR, GY, GR, RD, PU = "#1f4e79", "#d9822b", "#7a7a7a", "#2e8b57", "#b03a2e", "#8e44ad"
C = E["cfg"]; W = E["width"]; Wv = E["wavelet"]; L = E["lodo"]; mac = {}
def pm(v, p=3): return ("%%.%df $\\pm$ %%.%df" % (p, p)) % (v[0], v[1])
def f3(v): return "%.3f" % v
def wt(name, text): open(os.path.join(TAB, name + ".tex"), "w").write(text)
NUM = {1: "One", 2: "Two", 4: "Four", 8: "Eight", 16: "Sixteen", 32: "ThirtyTwo", 64: "SixtyFour", 128: "OneTwentyEight", 256: "TwoFiftySix"}
P4 = C["pool_all_cnn_D4"]; P32 = C["pool_all_cnn_D32"]
# ------------------------------------------------ main table
rows = [("Task code, light encoder, 4 B", "3 databases", "pool_all_cnn_D4"), ("Task code, light encoder, 32 B", "3 databases", "pool_all_cnn_D32"), ("Task code, light encoder, 1 B", "3 databases", "pool_all_cnn_D1"),
        ("Task code, light encoder, 4 B", "DS1 only", "ds1_cnn_D4"), ("Task code, light encoder, 32 B", "DS1 only", "abl_cnn_only"), ("Task code, Transformer encoder, 32 B", "DS1 only", "main"), ("Task code, Transformer encoder, 4 B", "DS1 only", "latent4")]
t = "\\begin{tabular}{llcccccc}\n\\toprule\nCode (bytes per 10-s window) & Training data & Seeds & AUROC & AUPRC & Macro-F1 & Sens.\\ @ 90\\% spec. & F1 @ op.\\ point \\\\\n\\midrule\n"
for nm, tr, k in rows:
    if k in C:
        c = C[k]; t += "%s & %s & %d & %s & %s & %s & %s & %s \\\\\n" % (nm, tr, c["auroc"][2], pm(c["auroc"]), pm(c["auprc"]), pm(c["macro_f1"]), pm(c["sens_at_spec"]), pm(c["f1_at_op"]))
    if k == "pool_all_cnn_D1": t += "\\midrule\n"
t += "\\midrule\nRR-interval code + logistic regression, 22 B & 3 databases & 5 & %s & -- & -- & -- & -- \\\\\n" % pm(L["ds2_rr_pool_all"]["auroc"])
t += "RR-interval code + logistic regression, 22 B & DS1 only & 5 & %s & -- & -- & -- & -- \\\\\n" % pm(L["ds2_rr_ds1only"]["auroc"])
t += "Spectral-summary code + logistic regression, 22 B & DS1 only & 10 & %s & -- & -- & -- & -- \\\\\n" % pm(A["baselines"]["summary"]["auroc"])
t += "\\bottomrule\n\\end{tabular}\n"; wt("tab_main", t)
for k, nm in (("pool_all_cnn_D4", "pFour"), ("pool_all_cnn_D32", "pThirtyTwo"), ("pool_all_cnn_D1", "pOne"), ("pool_all_cnn_D2", "pTwo"), ("pool_all_cnn_D8", "pEight"), ("pool_all_cnn_D16", "pSixteen"), ("ds1_cnn_D4", "dsLightFour"), ("abl_cnn_only", "dsLight"), ("main", "dsFull"), ("latent4", "dsFullFour"),
              ("pool_all_full_D4", "pFullFour"), ("pool_all_cnn_D4_1lead", "pOneLead"), ("pool_all_cnn_D4_noreg", "pNoreg"), ("pool_all_cnn_D4_norate", "pNorate"), ("pool_all_cnn_D4_noee", "pNoee"), ("pool_all_cnn_D4_taskonly", "pTaskonly")):
    if k in C:
        c = C[k]; mac[nm + "AUROC"] = pm(c["auroc"]); mac[nm + "N"] = str(c["auroc"][2]); mac[nm + "Fone"] = pm(c["macro_f1"]); mac[nm + "AUPRC"] = pm(c["auprc"]); mac[nm + "Sens"] = pm(c["sens_at_spec"]); mac[nm + "Fop"] = pm(c["f1_at_op"])
        mac[nm + "Params"] = "{:,}".format(c["params"]); mac[nm + "MACs"] = "%.2f" % (c["macs"] / 1e6); mac[nm + "Min"] = f3(min(c["seeds"])); mac[nm + "Max"] = f3(max(c["seeds"]))
        mac[nm + "PcN"], mac[nm + "PcS"], mac[nm + "PcV"] = [f3(v) for v in c["per_class"]]; mac[nm + "Ntrain"] = "{:,}".format(c["n_train"])
mac["pFourParamsDec"] = "{:,}".format(P4["params_dec"])
mac["rrPoolDsTwo"] = pm(L["ds2_rr_pool_all"]["auroc"]); mac["rrDsOneDsTwo"] = pm(L["ds2_rr_ds1only"]["auroc"]); mac["rrPoolDsTwoAbn"] = pm(L["ds2_rr_pool_all"]["abn_auroc"]); mac["rrDsOneDsTwoAbn"] = pm(L["ds2_rr_ds1only"]["abn_auroc"])
mac["summaryAUROCb"] = pm(A["baselines"]["summary"]["auroc"])
D4d = E["detail"]["pool_all_cnn_D4"]; D32d = E["detail"].get("pool_all_cnn_D32")
mac["afN"] = str(D4d["n_af"]); mac["afSens"] = pm(D4d["af_sens"], 2); mac["otherSensB"] = pm(D4d["other_sens"], 2)
mac["eraseTwentyFour"] = "%.3f" % abs(D4d["erase20"][0]); mac["eraseFortyFour"] = "%.3f" % abs(D4d["erase40"][0])
if D32d: mac["eraseTwentyThirtyTwo"] = "%.3f" % abs(D32d["erase20"][0]); mac["eraseFortyThirtyTwo"] = "%.3f" % abs(D32d["erase40"][0])
# per-class table
t = "\\begin{tabular}{lccc}\n\\toprule\nOne-vs-rest AUROC on DS2 & Normal & SVEB & VEB \\\\\n\\midrule\n"
for nm, k in (("Task code, 4 B, 3 databases", "pool_all_cnn_D4"), ("Task code, 32 B, 3 databases", "pool_all_cnn_D32"), ("Task code, 4 B, DS1 only", "ds1_cnn_D4")):
    if k in C: t += "%s & %s \\\\\n" % (nm, " & ".join("%.3f $\\pm$ %.3f" % (a, b) for a, b in zip(C[k]["per_class"], C[k]["per_class_sd"])))
t += "\\bottomrule\n\\end{tabular}\n"; wt("tab_perclass", t)
# ------------------------------------------------ width x bits
t = "\\begin{tabular}{lrrrrr}\n\\toprule\n & \\multicolumn{5}{c}{Bits per coefficient after post-training requantization} \\\\\n\\cmidrule(l){2-6}\nCoefficients $D$ (seeds) & 8 & 4 & 3 & 2 & 1 \\\\\n\\midrule\n"
for D in (32, 16, 8, 4, 2, 1):
    if str(D) in W:
        r = dict(W[str(D)]); r["8"] = C["pool_all_cnn_D%d" % D]["auroc"]; t += "%d (%d) & " % (D, r["8"][2]) + " & ".join("%.3f $\\pm$ %.3f" % (r[b][0], r[b][1]) for b in ("8", "4", "3", "2", "1")) + " \\\\\n"
        for b, bn in (("8", "Eight"), ("4", "Four"), ("3", "Three"), ("2", "Two"), ("1", "One")): mac["w%sb%s" % (NUM[D], bn)] = pm(r[b])
t += "\\bottomrule\n\\end{tabular}\n"; wt("tab_width", t)
# ------------------------------------------------ wavelet table + figure
BUD = [b for b in (4, 8, 16, 32, 64, 128, 256) if str(b) in Wv]
t = "\\begin{tabular}{l" + "r" * len(BUD) + "}\n\\toprule\n & \\multicolumn{%d}{c}{Bytes per 10-s window} \\\\\n\\cmidrule(l){2-%d}\nWavelet coder (db4, ideal entropy coding) & " % (len(BUD), len(BUD) + 1) + " & ".join(str(b) for b in BUD) + " \\\\\n\\midrule\n"
t += "Non-zero coefficients (mean) & " + " & ".join("%.0f" % json.load(open((WORK + "/classic_prd.json")))["dwt_%d" % b]["mean_nonzero"] for b in BUD) + " \\\\\n"
t += "PRD on DS2 (\\%) & " + " & ".join("%.1f" % Wv[str(b)]["prd"] for b in BUD) + " \\\\\n"
t += "AUROC, deployed classifier & " + " & ".join("%.3f $\\pm$ %.3f" % tuple(Wv[str(b)]["deployed"][:2]) for b in BUD) + " \\\\\n"
t += "AUROC, matched classifier & " + " & ".join(("%.3f $\\pm$ %.3f" % tuple(Wv[str(b)]["matched"][:2])) if "matched" in Wv[str(b)] else "--" for b in BUD) + " \\\\\n\\midrule\n"
t += "Task code of the same size & " + " & ".join(("%.3f $\\pm$ %.3f" % tuple(C["pool_all_cnn_D%d" % b]["auroc"][:2])) if "pool_all_cnn_D%d" % b in C else "--" for b in BUD) + " \\\\\n"
t += "\\bottomrule\n\\end{tabular}\n"; wt("tab_wavelet", t)
for b in BUD:
    mac["wvDep" + NUM[b]] = pm(Wv[str(b)]["deployed"]); mac["wvPRD" + NUM[b]] = "%.0f" % Wv[str(b)]["prd"]
    if "matched" in Wv[str(b)]: mac["wvMat" + NUM[b]] = pm(Wv[str(b)]["matched"]); mac["wvMatN"] = str(Wv[str(b)]["matched"][2])
def best(b): return max(Wv[str(b)]["deployed"][0], Wv[str(b)].get("matched", [0])[0])
refU = P32["auroc"][0]; mac["refAUROC"] = pm(P32["auroc"])
okr = [b for b in BUD if best(b) >= refU - 0.01]; mac["wvReachRef"] = str(okr[0]) if okr else "more than 256"
okt = [D for D in (1, 2, 4, 8, 16, 32) if "pool_all_cnn_D%d" % D in C and C["pool_all_cnn_D%d" % D]["auroc"][0] >= refU - 0.01]; mac["taskReachRef"] = str(okt[0]); mac["reachFactor"] = str(okr[0] // okt[0]) if okr else "--"
okh = [D for D in (1, 2, 4) if W[str(D)]["4"][0] >= refU - 0.01]; mac["taskReachRefHalf"] = "%g" % (okh[0] / 2.0) if okh else "--"
for k_, n_ in (("pool_all_cnn_D4", "pFour"), ("ds1_cnn_D4", "dsLightFour")): mac[n_ + "SD"] = "%.3f" % C[k_]["auroc"][1]
ref4 = P4["auroc"][0]
ok = [b for b in BUD if best(b) >= ref4 - 0.01]; mac["wvReachFour"] = str(ok[0]) if ok else "more than 256"; mac["wvReachFourFactor"] = str(ok[0] // 4) if ok else "more than 64"
ok1 = [b for b in BUD if best(b) >= C["pool_all_cnn_D1"]["auroc"][0] - 0.01] if "pool_all_cnn_D1" in C else []; mac["wvReachOne"] = str(ok1[0]) if ok1 else "more than 256"
fig, ax = plt.subplots(figsize=(5.4, 3.1))
Ds = [D for D in (1, 2, 4, 8, 16, 32) if "pool_all_cnn_D%d" % D in C]
ax.errorbar(Ds, [C["pool_all_cnn_D%d" % D]["auroc"][0] for D in Ds], yerr=[C["pool_all_cnn_D%d" % D]["auroc"][1] for D in Ds], color=BL, marker="o", ms=4.5, lw=1.5, capsize=2, zorder=5, label="Task-oriented code ($D$ coefficients $\\times$ 8 bits)")
sub = [(D * 4 / 8.0, W[str(D)]["4"][0], W[str(D)]["4"][1]) for D in (1, 2, 4) if str(D) in W]
ax.errorbar([s[0] for s in sub], [s[1] for s in sub], yerr=[s[2] for s in sub], color=BL, marker="o", mfc="white", ms=4, lw=1.0, ls=":", capsize=2, zorder=4, label="Task-oriented code requantized to 4 bits ($D$ = 1, 2, 4)")
ax.errorbar(BUD, [Wv[str(b)]["deployed"][0] for b in BUD], yerr=[Wv[str(b)]["deployed"][1] for b in BUD], color=OR, marker="s", ms=4, lw=1.3, capsize=2, label="Wavelet coder, decode then diagnose (deployed classifier)")
mb = [b for b in BUD if "matched" in Wv[str(b)]]
ax.errorbar(mb, [Wv[str(b)]["matched"][0] for b in mb], yerr=[Wv[str(b)]["matched"][1] for b in mb], color=OR, marker="s", mfc="white", ms=4, lw=1.3, ls="--", capsize=2, label="Wavelet coder, decode then diagnose (matched classifier)")
ax.plot([22], [L["ds2_rr_pool_all"]["auroc"][0]], color=PU, marker="*", ms=9, ls="none", label="RR-interval code + logistic regression (22 B)")
ax.axhline(P32["auroc"][0], color=GY, lw=0.7, ls=":"); ax.text(0.42, 0.935, "dotted line: same classifier on uncompressed windows", fontsize=6, color=GY, ha="left", va="center")
ax.set_xscale("log", base=2); ax.set_xticks([0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]); ax.set_xticklabels(["4 bit", "1", "2", "4", "8", "16", "32", "64", "128", "256"], fontsize=6.5)
ax.set_xlabel("Bytes per 10-s window (raw window: 5000 bytes)"); ax.set_ylabel("AUROC on DS2"); ax.set_ylim(0.6, 0.95)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False, fontsize=6.3)
fig.savefig(os.path.join(FIG, "fig_rd.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(FIG, "fig_rd.png"), dpi=160, bbox_inches="tight"); plt.close(fig)
# ------------------------------------------------ leave-one-database-out
names = {"svdb": "MIT-BIH Supraventricular", "incart": "St Petersburg INCART"}
t = "\\begin{tabular}{llcccccc}\n\\toprule\nHeld-out database & Code and training data & AUROC & Normal & SVEB & VEB & Abn.\\ vs.\\ normal & Sens.\\ / Spec. \\\\\n\\midrule\n"
for db in ("svdb", "incart"):
    r = L[db]; first = names[db]; oth = "INCART" if db == "svdb" else "SVDB"
    for key, nm in (("task_D4", "Task code 4 B; DS1 + %s" % oth), ("task_D32", "Task code 32 B; DS1 + %s" % oth), ("task_ds1only_D4", "Task code 4 B; DS1 only")):
        if key in r:
            q = r[key]; t += "%s & %s & %s & %.3f & %.3f & %.3f & %s & %.2f / %.2f \\\\\n" % (first, nm, pm(q["auroc"]), q["per_class"][0], q["per_class"][1], q["per_class"][2], pm(q["abn_auroc"]), q["sens"][0], q["spec"][0]); first = ""
            s = db + {"task_D4": "Four", "task_D32": "ThirtyTwo", "task_ds1only_D4": "DsOnly"}[key]
            mac[s + "AUROC"] = pm(q["auroc"]); mac[s + "Abn"] = pm(q["abn_auroc"]); mac[s + "SVEB"] = f3(q["per_class"][1]); mac[s + "VEB"] = f3(q["per_class"][2]); mac[s + "Sens"] = "%.2f" % q["sens"][0]; mac[s + "Spec"] = "%.2f" % q["spec"][0]
    for key, nm in (("rr_pool", "RR-interval code 22 B; DS1 + %s" % oth), ("rr_ds1only", "RR-interval code 22 B; DS1 only")):
        q = r[key]; t += " & %s & %s & -- & -- & -- & %s & -- \\\\\n" % (nm, pm(q["auroc"]), pm(q["abn_auroc"])); s = db + ("RrPool" if key == "rr_pool" else "RrDs"); mac[s + "AUROC"] = pm(q["auroc"]); mac[s + "Abn"] = pm(q["abn_auroc"])
    mac[db + "N"] = "{:,}".format(r["n"]); mac[db + "Rec"] = str(r["records"]); mac[db + "Cls"] = "/".join("{:,}".format(c) for c in r["classes"]); t += "\\midrule\n"
t = t[:-len("\\midrule\n")] + "\\bottomrule\n\\end{tabular}\n"; wt("tab_xdb", t)
mac["pFourAbn"] = pm([P4["per_class"][0], P4["per_class_sd"][0]])
# ------------------------------------------------ ablation (pooled light D=4) + DS1-only encoder comparison
t = "\\begin{tabular}{lcccrr}\n\\toprule\nVariant & Seeds & AUROC & Macro-F1 & Enc.\\ params & MACs / window \\\\\n\\midrule\n"
for nm, k in (("Light encoder, all loss terms (reference)", "pool_all_cnn_D4"), ("\\quad no latent $\\ell_2$ term ($\\lambda=0$)", "pool_all_cnn_D4_noreg"), ("\\quad no rate term ($\\gamma=0$)", "pool_all_cnn_D4_norate"), ("\\quad no auxiliary head", "pool_all_cnn_D4_noee"),
               ("\\quad cross-entropy only", "pool_all_cnn_D4_taskonly"), ("\\quad single lead", "pool_all_cnn_D4_1lead"), ("Transformer encoder (2 layers)", "pool_all_full_D4")):
    if k in C: c = C[k]; t += "%s & %d & %s & %s & %s & %s \\\\\n" % (nm, c["auroc"][2], pm(c["auroc"]), pm(c["macro_f1"]), "{:,}".format(c["params"]), "{:,}".format(c["macs"]))
t += "\\midrule\n\\multicolumn{6}{l}{\\emph{Trained on DS1 only, 32 coefficients}} \\\\\n"
for nm, k in (("Light encoder", "abl_cnn_only"), ("Transformer encoder (2 layers)", "main")):
    if k in C: c = C[k]; t += "%s & %d & %s & %s & %s & %s \\\\\n" % (nm, c["auroc"][2], pm(c["auroc"]), pm(c["macro_f1"]), "{:,}".format(c["params"]), "{:,}".format(c["macs"]))
t += "\\bottomrule\n\\end{tabular}\n"; wt("tab_ablation", t)
# ------------------------------------------------ paired tests
PN = {"pool_all_cnn_D4|ds1_cnn_D4": "pPoolVsDs", "pool_all_cnn_D4|pool_dwt4_cnn": "pWvFour", "pool_all_cnn_D8|pool_dwt8_cnn": "pWvEight", "pool_all_cnn_D16|pool_dwt16_cnn": "pWvSixteen", "pool_all_cnn_D4|pool_dwt16_cnn": "pFourVsWvSixteen",
      "pool_all_cnn_D4|pool_dwt64_cnn": "pFourVsWvSixtyFour", "pool_all_cnn_D4|pool_dwt256_cnn": "pFourVsWvTwoFiftySix", "pool_all_cnn_D4|pool_all_cnn_D32": "pFourVsThirtyTwo", "pool_all_cnn_D4|pool_all_cnn_D1": "pFourVsOne",
      "pool_all_cnn_D4|pool_all_full_D4": "pLightVsFull", "pool_all_cnn_D4|pool_all_cnn_D4_1lead": "pTwoVsOneLead", "pool_all_cnn_D4|pool_all_cnn_D4_taskonly": "pTaskonly", "pool_all_cnn_D4|pool_all_cnn_D4_noreg": "pNoreg",
      "pool_all_cnn_D4|pool_all_cnn_D4_norate": "pNorate", "pool_all_cnn_D4|pool_all_cnn_D4_noee": "pNoee", "main|abl_cnn_only": "pDsFullVsLight", "latent4|cls_dwt_4": "pDsWvFour", "main|cls_dwt_16": "pDsWvSixteen"}
for k, v in E["paired"].items():
    if k in PN: mac["pv" + PN[k][1:]] = ("%.3f" % v["p"]) if v["p"] < 0.1 else ("%.2f" % v["p"]); mac["df" + PN[k][1:]] = "%.3f" % abs(v["diff"]); mac["nn" + PN[k][1:]] = str(v["n"])
# ------------------------------------------------ DS1-only extras from the first evaluation
A2 = E1.get("recon_decode", {})
for k, nm in (("recon_convae", "convaeDecode"), ("recon_sameenc_pe", "sameencDecode")):
    if k in A2: mac[nm] = pm(A2[k]["auroc_decode"]); mac[nm + "PRD"] = "%.0f" % A2[k]["prd"][0]
for k, nm in (("cls_dwt_4", "dsWvMatFour"), ("cls_dwt_8", "dsWvMatEight"), ("cls_dwt_16", "dsWvMatSixteen")):
    if k in C: mac[nm] = pm(C[k]["auroc"]); mac[nm + "N"] = str(C[k]["auroc"][2])
for b in (16, 32, 64, 256):
    if "dwt_%d" % b in E1["classic"]: mac["dsWvDep" + NUM[b]] = pm(E1["classic"]["dwt_%d" % b]["auroc_orig_model"])
for k, nm in (("ptbxl_cnn_D4", "ptbFour"), ("ptbxl_cnn_D32", "ptbThirtyTwo")):
    if k in C: c = C[k]; mac[nm + "AUROC"] = pm(c["auroc"]); mac[nm + "AUPRC"] = pm(c["auprc"]); mac[nm + "Fone"] = pm(c["macro_f1"]); mac[nm + "N"] = str(c["auroc"][2]); mac["ptbNtrain"] = "{:,}".format(c["n_train"]); mac["ptbNtest"] = "{:,}".format(c["n_test"]); mac["ptbNrec"] = "{:,}".format(c["n_train"] + c["n_val"] + c["n_test"])
with open(os.path.join(PAPER, "numbers_new.tex"), "w") as f:
    for k, v in mac.items(): f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
print("macros", len(mac)); print("\n".join("%s = %s" % kv for kv in mac.items()))
