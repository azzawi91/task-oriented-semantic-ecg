"""Post-hoc evaluations added in the 2026-11 revision (no training):
 A. diagnosis after classical (wavelet / DCT) compression, B. zero-shot cross-database tests,
 C. linear probes of what the code keeps, D. width x bit-depth sweep, E. paired statistics."""
import os as _os
WORK = _os.environ.get("ECG_WORK", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "work"))  # data, runs and outputs live here
import os, sys, json, glob, numpy as np, torch
from scipy.stats import wilcoxon
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, r2_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rev_np as R, rev_metrics as M, rev_splits as S
from rev_model import Encoder, Decoder, quantize_ste
RES = (WORK + "/res"); CD = os.path.join(HERE, "caches"); OUT = {}
torch.set_num_threads(2)
def load_models(name):
    ms = []
    for f in sorted(glob.glob(os.path.join(RES, name, "seed*.pt"))):
        ck = torch.load(f, map_location="cpu"); c = ck["cfg"]
        enc = Encoder(ck["in_ch"], use_transformer=c["use_transformer"], n_layers=c["n_layers"], bits=c["bits"], n_classes=c["n_classes"],
                      pos_enc=c.get("pos_enc", False), latent=c.get("latent", 32)); dec = Decoder(latent=c.get("latent", 32), n_classes=c["n_classes"])
        enc.load_state_dict(ck["enc"]); dec.load_state_dict(ck["dec"]); enc.eval(); dec.eval()
        ms.append((enc, dec, json.load(open(f[:-3] + ".json"))))
    return ms
def probs(enc, dec, X):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            z = enc(torch.tensor(X[i:i + 256]))[3]; out.append(dec(quantize_ste(z, 8, False)).softmax(-1).numpy())
    return np.concatenate(out)
def msd(v): return [float(np.mean(v)), float(np.std(v)), len(v)]
d = np.load(os.path.join(CD, "mitbih_fs125_both.npz"), allow_pickle=True); X, y, g = d["X"], d["y"], d["groups"]
te = np.where(np.isin(g, S.DS2))[0]
main = load_models("main"); mlii = load_models("lead_mlii")
print("models", len(main), len(mlii), flush=True)

# ---------------- A. classical codecs: diagnosis after decompression (models trained on originals)
if os.path.exists((WORK + "/classic_prd.json")):
    prd = json.load(open((WORK + "/classic_prd.json"))); A = {}
    for key, info in prd.items():
        p = os.path.join(CD, "cls_%s.npz" % key)
        if not os.path.exists(p): continue
        Xr = np.load(p)["X"][te]
        A[key] = dict(info, auroc_orig_model=msd([M.auroc_multiclass(y[te], probs(e, dc, Xr)) for e, dc, _ in main]))
        mt = [json.load(open(f))["metrics"]["auroc"] for f in sorted(glob.glob(os.path.join(RES, "cls_" + key, "seed*.json")))]
        if mt: A[key]["auroc_matched_model"] = msd(mt)
        print("A", key, A[key], flush=True)
    OUT["classic"] = A

# ---------------- A2. learned reconstruction codecs evaluated the same way (decode, then diagnose)
A2 = {}
for nm in ("recon_convae", "recon_jscc", "recon_sameenc_pe"):
    v = []; prds = []
    for f in sorted(glob.glob(os.path.join(RES, nm, "seed*.npz"))):
        z = np.load(f, allow_pickle=True)
        if "X_rec_te" not in z: continue
        Xr = z["X_rec_te"].astype(np.float32); yt = z["y_te"]
        v.append(float(np.mean([M.auroc_multiclass(yt, probs(e, dc, Xr)) for e, dc, _ in main])))
        prds.append(json.load(open(f[:-4] + ".json"))["test_prd_pct"])
    if v: A2[nm] = dict(auroc_decode=msd(v), prd=msd(prds))
OUT["recon_decode"] = A2; print("A2", A2, flush=True)

# ---------------- B. zero-shot cross-database
B = {}
for db in ("svdb", "incart"):
    p = os.path.join(CD, db + ".npz")
    if not os.path.exists(p): continue
    e = np.load(p, allow_pickle=True); Xe, ye = e["X"], e["y"]; r = {"n": int(len(ye)), "classes": np.bincount(ye).tolist(), "records": int(len(set(e["groups"])))}
    for nm, models, Xin in (("two_lead", main, Xe), ("single_lead", mlii, Xe[:, :1])):
        au, pc, ab, se, sp = [], [], [], [], []
        for enc, dec, meta in models:
            P = probs(enc, dec, Xin); au.append(M.auroc_multiclass(ye, P)); pc.append(M.per_class_auroc(ye, P))
            yb, sb = M.binary_view(ye, P); ab.append(roc_auc_score(yb, sb)); thr = meta["metrics"]["threshold"]; pr = sb >= thr
            se.append(float((pr & (yb == 1)).sum() / (yb == 1).sum())); sp.append(float(((~pr) & (yb == 0)).sum() / (yb == 0).sum()))
        if au: r[nm] = dict(auroc=msd(au), per_class=np.mean(pc, 0).tolist(), abn_auroc=msd(ab), sens=msd(se), spec=msd(sp))
    # hand-crafted RR code trained on DS1, tested on the external database
    rc = os.path.join(CD, "rr_%s.npy" % db)
    Fe = np.load(rc) if os.path.exists(rc) else R.rr_features(Xe, 125); np.save(rc, Fe)
    rcm = os.path.join(CD, "rr_features_fs125.npy"); Fm = np.load(rcm) if os.path.exists(rcm) else R.rr_features(X, 125); np.save(rcm, Fm)
    au, ab = [], []
    for seed in range(10):
        tr, va, _ = S.dechazal_split(y, g, seed); sc = R.dim_scale(Fm[tr])
        P = R.logreg_readout(R.quantize(Fm[tr], 8, sc), y[tr], R.quantize(Fe, 8, sc), seed=seed)
        au.append(M.auroc_multiclass(ye, P)); yb, sb = M.binary_view(ye, P); ab.append(roc_auc_score(yb, sb))
    r["rr_code"] = dict(auroc=msd(au), abn_auroc=msd(ab)); B[db] = r; print("B", db, json.dumps(r)[:600], flush=True)
OUT["xdb"] = B

# ---------------- C. linear probes: what the transmitted code retains
fz = np.load(os.path.join(CD, "mitbih_feats.npz"), allow_pickle=True); F, names = fz["F"], [str(n) for n in fz["names"]]
def probe(Ztr, Zte, itr, ite):
    sc = StandardScaler().fit(Ztr); out = []
    for j in range(F.shape[1]):
        t = F[:, j]; mu, sd = t[itr].mean(), t[itr].std() + 1e-9
        m = Ridge(alpha=1.0).fit(sc.transform(Ztr), (t[itr] - mu) / sd); out.append(max(-1.0, r2_score((t[ite] - mu) / sd, m.predict(sc.transform(Zte)))))
    return out
C = {"names": names}
for nm in ("main", "recon_convae", "recon_sameenc_pe", "abl_cnn_only"):
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, nm, "seed*.npz"))):
        z = np.load(f, allow_pickle=True); rows.append(probe(z["Z_tr"], z["Z_te"], z["idx_tr"], z["idx_te"]))
    if rows: C[nm] = dict(mean=np.mean(rows, 0).tolist(), sd=np.std(rows, 0).tolist(), n=len(rows))
rows = []
for seed in range(3):
    tr, va, _ = S.dechazal_split(y, g, seed); Ptr, Pte = R.pca_code(X[tr], X[te], 32); rows.append(probe(Ptr, Pte, tr, te))
C["pca32"] = dict(mean=np.mean(rows, 0).tolist(), sd=np.std(rows, 0).tolist(), n=3); OUT["probe"] = C; print("C", json.dumps(C)[:900], flush=True)

# ---------------- D. width x bit-depth sweep (native decoder)
def q1(Z, Ztr): c = np.abs(Ztr).mean(0); return np.sign(Z) * c
Dres = {}
for nm, Dl in (("latent1", 1), ("latent2", 2), ("latent4", 4), ("latent8", 8), ("latent16", 16), ("main", 32)):
    fs = sorted(glob.glob(os.path.join(RES, nm, "seed*.npz")))
    if not fs: continue
    row = {}
    for b in (8, 4, 3, 2, 1):
        v = []
        for f in fs:
            z = np.load(f, allow_pickle=True); w = {k: z[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3")}
            Zq = R.quantize(z["Z_te"], b) if b > 1 else q1(z["Z_te"], z["Z_tr"]); v.append(M.auroc_multiclass(z["y_te"], R.np_decoder(w, Zq)))
        row[str(b)] = msd(v)
    Dres[str(Dl)] = row; print("D", Dl, {k: round(v[0], 3) for k, v in row.items()}, flush=True)
OUT["width"] = Dres

# ---------------- E. paired statistics across seeds (same calibration split per seed)
def aucs(nm): return [json.load(open(f))["metrics"]["auroc"] for f in sorted(glob.glob(os.path.join(RES, nm, "seed*.json")))]
E = {}; a = aucs("main")
for nm in ("abl_cnn_only", "abl_tf1", "abl_task_only", "lead_mlii", "latent4", "abl_no_reg", "abl_no_rate", "abl_no_ee", "fs250", "fs360", "main_pe"):
    b = aucs(nm); n = min(len(a), len(b))
    if n >= 5:
        try: p = float(wilcoxon(a[:n], b[:n]).pvalue)
        except ValueError: p = 1.0
        E[nm] = dict(n=n, mean_diff=float(np.mean(np.array(b[:n]) - np.array(a[:n]))), p=p)
OUT["paired"] = E; print("E", E, flush=True)
json.dump(OUT, open((WORK + "/new_eval.json"), "w"), indent=1)
