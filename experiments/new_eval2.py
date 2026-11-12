"""Evaluation of the multi-database (pooled) experiments. Writes <work>/new_eval2.json"""
import os as _os
WORK = _os.environ.get("ECG_WORK", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "work"))  # data, runs and outputs live here
import os, sys, json, glob, numpy as np, torch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rev_np as R, rev_metrics as M, rev_splits as S
from rev_model import Encoder, Decoder, quantize_ste
RES = (WORK + "/res"); CD = os.path.join(HERE, "caches"); OUT = {}; torch.set_num_threads(2)
def msd(v): return [float(np.mean(v)), float(np.std(v)), len(v)]
def metas(name): return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, name, "seed*.json")))]
def agg(name):
    ms = metas(name)
    if not ms: return None
    r = {k: msd([m["metrics"][k] for m in ms]) for k in ("auroc", "auprc", "macro_f1", "sens_at_spec", "f1_at_op", "spec_at_op") if k in ms[0]["metrics"]}
    r["per_class"] = np.mean([m["metrics"]["per_class_auroc"] for m in ms], 0).tolist(); r["per_class_sd"] = np.std([m["metrics"]["per_class_auroc"] for m in ms], 0).tolist()
    r["params_dec"] = ms[0]["params_dec"]; r["params"] = ms[0]["params_enc"]; r["macs"] = ms[0]["macs_enc"]; r["n_train"] = ms[0]["n_train"]; r["seeds"] = [m["metrics"]["auroc"] for m in ms]; return r
def load_models(name):
    out = []
    for f in sorted(glob.glob(os.path.join(RES, name, "seed*.pt"))):
        ck = torch.load(f, map_location="cpu"); c = ck["cfg"]
        enc = Encoder(ck["in_ch"], use_transformer=c["use_transformer"], n_layers=c["n_layers"], bits=c["bits"], n_classes=c["n_classes"], pos_enc=c.get("pos_enc", False), latent=c.get("latent", 32))
        dec = Decoder(latent=c.get("latent", 32), n_classes=c["n_classes"]); enc.load_state_dict(ck["enc"]); dec.load_state_dict(ck["dec"]); enc.eval(); dec.eval()
        out.append((enc, dec, json.load(open(f[:-3] + ".json"))))
    return out
def probs(enc, dec, X):
    o = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            o.append(dec(quantize_ste(enc(torch.tensor(X[i:i + 512]))[3], 8, False)).softmax(-1).numpy())
    return np.concatenate(o)
pool = np.load(os.path.join(CD, "pool_all.npz"), allow_pickle=True); X, y, g = pool["X"], pool["y"], pool["groups"].astype(str)
is_sv = np.char.startswith(g, "svdb_"); is_in = np.char.startswith(g, "incart_"); te = np.where(np.isin(g, S.DS2))[0]
mit = np.load(os.path.join(CD, "mitbih_fs125_both.npz"), allow_pickle=True); rhythm = mit["rhythm"].astype(str)

# 1. configurations summarised from their own DS2 metrics
names = ["pool_all_cnn_D%d" % D for D in (1, 2, 4, 8, 16, 32)] + ["pool_noSV_cnn_D4", "pool_noSV_cnn_D32", "pool_noIN_cnn_D4", "pool_noIN_cnn_D32", "ds1_cnn_D4", "abl_cnn_only", "main", "latent4",
         "pool_all_full_D4", "pool_all_cnn_D4_1lead", "pool_all_cnn_D4_noreg", "pool_all_cnn_D4_norate", "pool_all_cnn_D4_noee", "pool_all_cnn_D4_taskonly"] + ["pool_dwt%d_cnn" % B for B in (4, 8, 16, 32, 64, 128, 256)] + ["cls_dwt_%d" % B for B in (4, 8, 16, 32, 64, 128, 256)]
OUT["cfg"] = {n: agg(n) for n in names if agg(n)}
for n in ("ptbxl_cnn_D4", "ptbxl_cnn_D32"):
    ms = metas(n)
    if ms: OUT["cfg"][n] = {k: msd([m["metrics"][k] for m in ms]) for k in ("auroc", "auprc", "macro_f1")} | dict(n_train=ms[0]["n_train"], n_val=ms[0]["n_val"], n_test=ms[0]["n_test"], params=ms[0]["params_enc"])
print("cfg", {k: (round(v["auroc"][0], 3), round(v["auroc"][1], 3), v["auroc"][2]) for k, v in OUT["cfg"].items()}, flush=True)

# 2. width x bit-depth (post-training quantisation), pooled light codec
def q1(Z, Ztr): return np.sign(Z) * np.abs(Ztr).mean(0)
Wd = {}
for D in (1, 2, 4, 8, 16, 32):
    fs = sorted(glob.glob(os.path.join(RES, "pool_all_cnn_D%d" % D, "seed*.npz")))
    if not fs: continue
    row = {}
    for b in (8, 4, 3, 2, 1):
        v = []
        for f in fs:
            z = np.load(f, allow_pickle=True); w = {k: z[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3")}
            sc = R.dim_scale(z["Z_tr"])          # fixed per-coefficient scale from the training codes
            v.append(M.auroc_multiclass(z["y_te"], R.np_decoder(w, R.quantize(z["Z_te"], b, sc) if b > 1 else q1(z["Z_te"], z["Z_tr"]))))
        row[str(b)] = msd(v)
    Wd[str(D)] = row
OUT["width"] = Wd; print("width", {D: {b: round(v[0], 3) for b, v in r.items()} for D, r in Wd.items()}, flush=True)

# 3. wavelet coder, deployed model = pooled light classifier (D=32) trained on original windows
dep = load_models("pool_all_cnn_D32"); Wv = {}
for B in (4, 8, 16, 32, 64, 128, 256):
    p = os.path.join(CD, "pool_dwt_%d.npz" % B)
    if not os.path.exists(p) or not dep: continue
    Xr = np.load(p)["X"][te]; Wv[str(B)] = dict(deployed=msd([M.auroc_multiclass(y[te], probs(e, d, Xr)) for e, d, _ in dep]),
                                               prd=float(100 * np.sqrt(((X[te] - Xr) ** 2).sum() / (X[te] ** 2).sum())))
    if "pool_dwt%d_cnn" % B in OUT["cfg"]: Wv[str(B)]["matched"] = OUT["cfg"]["pool_dwt%d_cnn" % B]["auroc"]
OUT["wavelet"] = Wv; print("wavelet", Wv, flush=True)

# 4. leave-one-database-out: models that never saw the target database
def ext_eval(models, idx):
    au, pc, ab, se, sp = [], [], [], [], []
    for enc, dec, meta in models:
        P = probs(enc, dec, X[idx]); ye = y[idx]; au.append(M.auroc_multiclass(ye, P)); pc.append(M.per_class_auroc(ye, P)); yb, sb = M.binary_view(ye, P); ab.append(roc_auc_score(yb, sb))
        pr = sb >= meta["metrics"]["threshold"]; se.append(float((pr & (yb == 1)).sum() / (yb == 1).sum())); sp.append(float(((~pr) & (yb == 0)).sum() / (yb == 0).sum()))
    return dict(auroc=msd(au), per_class=np.mean(pc, 0).tolist(), abn_auroc=msd(ab), sens=msd(se), spec=msd(sp))
Frr = {"mit": np.load(os.path.join(CD, "rr_features_fs125.npy")), "svdb": np.load(os.path.join(CD, "rr_svdb.npy")), "incart": np.load(os.path.join(CD, "rr_incart.npy"))}
Fall = np.concatenate([Frr["mit"], Frr["svdb"], Frr["incart"]])
def rr_eval(train_ext_mask, test_idx):
    au, ab = [], []
    for seed in range(5):
        tr, va, _ = S.dechazal_split(y, g, seed); tr = np.concatenate([tr, np.where(train_ext_mask)[0]]); sc = R.dim_scale(Fall[tr])
        P = R.logreg_readout(R.quantize(Fall[tr], 8, sc), y[tr], R.quantize(Fall[test_idx], 8, sc), seed=seed)
        au.append(M.auroc_multiclass(y[test_idx], P)); yb, sb = M.binary_view(y[test_idx], P); ab.append(roc_auc_score(yb, sb))
    return dict(auroc=msd(au), abn_auroc=msd(ab))
L = {}
for db, mask, key, other in (("svdb", is_sv, "noSV", is_in), ("incart", is_in, "noIN", is_sv)):
    idx = np.where(mask)[0]; r = {"n": int(len(idx)), "classes": np.bincount(y[idx]).tolist(), "records": int(len(set(g[idx])))}
    for D in (4, 32):
        ms = load_models("pool_%s_cnn_D%d" % (key, D))
        if ms: r["task_D%d" % D] = ext_eval(ms, idx)
    ms = load_models("ds1_cnn_D4")
    if ms: r["task_ds1only_D4"] = ext_eval(ms, idx)
    r["rr_pool"] = rr_eval(other, idx); r["rr_ds1only"] = rr_eval(np.zeros(len(y), bool), idx); L[db] = r; print("LODO", db, json.dumps(r)[:1500], flush=True)
L["ds2_rr_pool_all"] = rr_eval(is_sv | is_in, te); L["ds2_rr_ds1only"] = rr_eval(np.zeros(len(y), bool), te)
OUT["lodo"] = L; print("rr ds2", L["ds2_rr_pool_all"], L["ds2_rr_ds1only"], flush=True)

# 5. per-class, AF/AFL sensitivity, erasure tolerance for the pooled light codec
def detail(name):
    afs, oth, er = [], [], {0.2: [], 0.4: []}
    for f in sorted(glob.glob(os.path.join(RES, name, "seed*.npz"))):
        z = np.load(f, allow_pickle=True); m = json.load(open(f[:-4] + ".json")); it = z["idx_te"]; P = z["P_te"]; yt = z["y_te"]; rh = rhythm[it]
        mask = np.char.startswith(rh, "AFIB") | np.char.startswith(rh, "AFL"); yb, sb = M.binary_view(yt, P); pr = sb >= m["metrics"]["threshold"]
        afs.append(float((pr & (yb == 1))[mask].sum() / max((yb[mask] == 1).sum(), 1))); oth.append(float((pr & (yb == 1))[~mask].sum() / max((yb[~mask] == 1).sum(), 1)))
        w = {k: z[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3")}; Zq = R.quantize(z["Z_te"], 8); clean = M.auroc_multiclass(yt, R.np_decoder(w, Zq))
        for p in er:
            rng = np.random.default_rng(m["seed"]); er[p].append(np.mean([M.auroc_multiclass(yt, R.np_decoder(w, Zq * (rng.random(Zq.shape) >= p))) for _ in range(10)]) - clean)
    return dict(af_sens=msd(afs), other_sens=msd(oth), n_af=int(mask.sum()), erase20=msd(er[0.2]), erase40=msd(er[0.4]))
OUT["detail"] = {n: detail(n) for n in ("pool_all_cnn_D4", "pool_all_cnn_D32") if glob.glob(os.path.join(RES, n, "seed*.npz"))}; print("detail", OUT["detail"], flush=True)

# 6. paired tests (same seeds -> same calibration split and initialisation seed)
def pt(a, b):
    A, B = OUT["cfg"].get(a), OUT["cfg"].get(b)
    if not A or not B: return None
    n = min(len(A["seeds"]), len(B["seeds"])); x, z = np.array(A["seeds"][:n]), np.array(B["seeds"][:n])
    try: p = float(wilcoxon(x, z).pvalue)
    except ValueError: p = 1.0
    return dict(n=n, diff=float((x - z).mean()), p=p)
pairs = [("pool_all_cnn_D4", "ds1_cnn_D4"), ("pool_all_cnn_D4", "pool_dwt4_cnn"), ("pool_all_cnn_D8", "pool_dwt8_cnn"), ("pool_all_cnn_D16", "pool_dwt16_cnn"), ("pool_all_cnn_D4", "pool_dwt16_cnn"), ("pool_all_cnn_D4", "pool_dwt64_cnn"),
         ("pool_all_cnn_D4", "pool_dwt256_cnn"), ("pool_all_cnn_D4", "pool_all_cnn_D32"), ("pool_all_cnn_D4", "pool_all_cnn_D1"), ("pool_all_cnn_D4", "pool_all_full_D4"), ("pool_all_cnn_D4", "pool_all_cnn_D4_1lead"),
         ("pool_all_cnn_D4", "pool_all_cnn_D4_taskonly"), ("pool_all_cnn_D4", "pool_all_cnn_D4_noreg"), ("pool_all_cnn_D4", "pool_all_cnn_D4_norate"), ("pool_all_cnn_D4", "pool_all_cnn_D4_noee"), ("main", "abl_cnn_only"), ("latent4", "cls_dwt_4"), ("main", "cls_dwt_16")]
OUT["paired"] = {"%s|%s" % p: pt(*p) for p in pairs if pt(*p)}; print("paired", OUT["paired"], flush=True)
json.dump(OUT, open((WORK + "/new_eval2.json"), "w"), indent=1)
