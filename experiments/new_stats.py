"""Uncertainty for the main comparisons: paired two-level bootstrap (test records x seeds).
Usage: python new_stats.py <part>   part in {ds2, ext, late}"""
import os as _os
WORK = _os.environ.get("ECG_WORK", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "work"))  # data, runs and outputs live here
import os, sys, json, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RES = (WORK + "/res"); CD = os.path.join(HERE, "caches"); NB = 2000; MARGIN = 0.05
C = np.load(os.path.join(CD, "pool_all.npz"), allow_pickle=True); G = C["groups"].astype(str); Y = C["y"]
import rev_splits as _S
GR = G                                   # record names (used to select databases)
_pid = _S.patient_ids(G); G = np.array([p if p is not None else "rec:" + r for p, r in zip(_pid, GR)])   # bootstrap clusters = patients
rng = np.random.default_rng(20261127)
def auc_bin(s, y):
    o = np.argsort(s, kind="mergesort"); ss = s[o]
    _, inv, cnt = np.unique(ss, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); avg = (cs - cnt + cs + 1) / 2.0; rk = np.empty(len(s)); rk[o] = avg[inv]
    n1 = y.sum(); n0 = len(y) - n1
    return np.nan if n1 == 0 or n0 == 0 else (rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def macro(P, y): return np.nanmean([auc_bin(P[:, k].astype(np.float64), (y == k).astype(int)) for k in range(P.shape[1])])
def abn(P, y): return auc_bin(1.0 - P[:, 0].astype(np.float64), (y != 0).astype(int))
def load(cfg):
    out = {}
    for f in sorted(glob.glob("%s/%s/seed*.npz" % (RES, cfg))):
        z = np.load(f); out[int(f.split("seed")[1].split(".")[0])] = (z["P_te"], z["idx_te"])
    return out
def boot(PA, PB, idx, metric=macro, nb=NB):
    """PA, PB: dict seed -> probabilities aligned with idx (global window indices). Returns summary dict."""
    ss = sorted(set(PA) & set(PB)); y = Y[idx]; g = G[idx]; recs = np.unique(g); mem = {r: np.where(g == r)[0] for r in recs}
    a = np.mean([metric(PA[s], y) for s in ss]); b = np.mean([metric(PB[s], y) for s in ss]); d = []
    for _ in range(nb):
        sel = np.concatenate([mem[r] for r in rng.choice(recs, len(recs))]); sd = rng.choice(ss, len(ss)); ys = y[sel]
        d.append(np.mean([metric(PA[s][sel], ys) - metric(PB[s][sel], ys) for s in sd]))
    d = np.array(d); d = d[~np.isnan(d)]
    return dict(a=float(a), b=float(b), diff=float(a - b), ci95=[float(v) for v in np.percentile(d, [2.5, 97.5])],
                ci90=[float(v) for v in np.percentile(d, [5, 95])], p=float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
                seeds=len(ss), records=int(len(recs)), windows=int(len(idx)))
def P(cfg): return {s: v[0] for s, v in load(cfg).items()}
OUTF = (WORK + "/new_stats.json")
OUT = json.load(open(OUTF)) if os.path.exists(OUTF) else {}
def put(k, v):
    OUT[k] = v; json.dump(OUT, open(OUTF, "w"), indent=1); print(k, json.dumps(v), flush=True)
part = sys.argv[1]
base = load("pool_all_cnn_D4"); te = base[0][1]
if part == "ds2":
    put("pop_D4_vs_ds1", boot(P("pool_all_cnn_D4"), P("ds1_cnn_D4"), te))
    for D in (1, 2, 4, 8, 16): put("rate_D%d_vs_D32" % D, boot(P("pool_all_cnn_D%d" % D), P("pool_all_cnn_D32"), te))
    put("rate_D4_vs_D1", boot(P("pool_all_cnn_D4"), P("pool_all_cnn_D1"), te))
    for B in (4, 8, 16, 64, 256):
        put("task4_vs_wavelet%d" % B, boot(P("pool_all_cnn_D4"), P("pool_dwt%d_cnn" % B), te))
        put("task1_vs_wavelet%d" % B, boot(P("pool_all_cnn_D1"), P("pool_dwt%d_cnn" % B), te))
    # on-device classifier sending its quantised posterior (two probabilities, b bits each) or the hard label
    ref = P("pool_all_cnn_D32")
    def q(Pm, b):
        L = 2 ** b - 1; Q = np.round(Pm[:, :2] * L) / L; return np.concatenate([Q, np.clip(1 - Q.sum(1, keepdims=True), 0, 1)], 1)
    def lab(Pm): return np.eye(3)[Pm.argmax(1)]
    for b in (8, 4, 2, 1): put("post%db_vs_ref" % b, boot({s: q(v, b) for s, v in ref.items()}, ref, te))
    put("label_vs_ref", boot({s: lab(v) for s, v in ref.items()}, ref, te))
    put("code1B_vs_post4b", boot(P("pool_all_cnn_D1"), {s: q(v, 4) for s, v in ref.items()}, te))
    put("code4B_vs_post4b", boot(P("pool_all_cnn_D4"), {s: q(v, 4) for s, v in ref.items()}, te))
    put("code4B_vs_label", boot(P("pool_all_cnn_D4"), {s: lab(v) for s, v in ref.items()}, te))
if part == "ext":
    import torch, rev_np as R, rev_splits as S
    from rev_model import Encoder, Decoder, quantize_ste
    torch.set_num_threads(1); X = C["X"]
    def models(name):
        out = {}
        for f in sorted(glob.glob(os.path.join(RES, name, "seed*.pt"))):
            ck = torch.load(f, map_location="cpu"); c = ck["cfg"]
            enc = Encoder(ck["in_ch"], use_transformer=c["use_transformer"], n_layers=c["n_layers"], bits=c["bits"], n_classes=c["n_classes"],
                          pos_enc=c.get("pos_enc", False), latent=c.get("latent", 32)); dec = Decoder(latent=c.get("latent", 32), n_classes=c["n_classes"])
            enc.load_state_dict(ck["enc"]); dec.load_state_dict(ck["dec"]); enc.eval(); dec.eval(); out[int(f.split("seed")[1].split(".")[0])] = (enc, dec)
        return out
    def probs(m, idx):
        enc, dec = m; out = []
        with torch.no_grad():
            for i in range(0, len(idx), 256):
                z = enc(torch.tensor(X[idx[i:i + 256]]))[3]; out.append(dec(quantize_ste(z, 8, False)).softmax(-1).numpy())
        return np.concatenate(out)
    Fall = np.concatenate([np.load(os.path.join(CD, "rr_features_fs125.npy")), np.load(os.path.join(CD, "rr_svdb.npy")), np.load(os.path.join(CD, "rr_incart.npy"))])
    is_sv = np.char.startswith(GR, "svdb_"); is_in = np.char.startswith(GR, "incart_")
    order_ok = bool((np.where(~(is_sv | is_in))[0].max() < np.where(is_sv)[0].min()) and (np.where(is_sv)[0].max() < np.where(is_in)[0].min()))
    assert order_ok and len(Fall) == len(Y)
    def rr(train_ext, idx, seeds):
        out = {}
        for s in seeds:
            tr, va, _ = S.dechazal_split(Y, C["groups"], s); tr = np.concatenate([tr, np.where(train_ext)[0]]); sc = R.dim_scale(Fall[tr])
            out[s] = R.logreg_readout(R.quantize(Fall[tr], 8, sc), Y[tr], R.quantize(Fall[idx], 8, sc), seed=s)
        return out
    for db, mask, key, other in (("svdb", is_sv, "noSV", is_in), ("incart", is_in, "noIN", is_sv)):
        idx = np.where(mask)[0]; ms = models("pool_%s_cnn_D4" % key); PL = {s: probs(m, idx) for s, m in ms.items()}
        np.savez_compressed((WORK + "/ext_pred_%s.npz") % db, idx=idx, **{"s%d" % s: v for s, v in PL.items()})
        PR = rr(other, idx, sorted(PL))
        put("ext_%s_learned_vs_rr_macro" % db, boot(PL, PR, idx, macro, 1000))
        put("ext_%s_learned_vs_rr_abn" % db, boot(PL, PR, idx, abn, 1000))
    # DS2: learned vs RR, same resampling
    PR = rr(is_sv | is_in, te, range(10))
    put("ds2_learned_vs_rr_macro", boot(P("pool_all_cnn_D4"), PR, te, macro)); put("ds2_learned_vs_rr_abn", boot(P("pool_all_cnn_D4"), PR, te, abn))
if part == "late":
    put("arch_tf_vs_cnn", boot(P("pool_all_full_D4"), P("pool_all_cnn_D4"), te))
    mx = load("pool_mixed_cnn_D4"); idxm = mx[0][1]; pos = {v: i for i, v in enumerate(te)}; sel = np.array([pos[v] for v in idxm])
    put("protocol_mixed_vs_inter", boot({s: v[0] for s, v in mx.items()}, {s: v[sel] for s, v in P("pool_all_cnn_D4").items()}, idxm))
if part == "cvlc":
    # ---- learning curve on DS2 (equal number of training windows processed)
    for a, b in (("lc_ds1_18_matched", "ds1_cnn_D4"), ("lc_mix18", "lc_ds1_18_matched"), ("lc_mix36", "lc_mix18"), ("lc_mix72", "lc_mix36"),
                 ("pool_all_cnn_D4", "lc_mix72"), ("pool_all_cnn_D4", "lc_ds1_18_matched"), ("pool_all_cnn_D4", "lc_mix18")):
        ("lc_%s_vs_%s" % (a, b)) in OUT or put("lc_%s_vs_%s" % (a, b), boot(P(a), P(b), te))
    # ---- patient-level five-fold cross-validation: out-of-fold predictions pooled per seed
    def oof(prefix):
        out = {}
        for s in range(3):
            Ps, Is = [], []
            for k in range(5):
                f = "%s/%s_f%d/seed%d.npz" % (RES, prefix, k, s)
                if not os.path.exists(f): break
                z = np.load(f); Ps.append(z["P_te"]); Is.append(z["idx_te"])
            else:
                idx = np.concatenate(Is); o = np.argsort(idx); out[s] = (np.concatenate(Ps)[o], idx[o])
        return out
    A4, A1, A32, M4 = oof("cv_all_D4"), oof("cv_all_D1"), oof("cv_all_D32"), oof("cv_mit_D4")
    idx = A4[0][1]; d = lambda O: {s: v[0] for s, v in O.items()}
    put("cv_D1_vs_D32", boot(d(A1), d(A32), idx, macro, 1000)); put("cv_D4_vs_D32", boot(d(A4), d(A32), idx, macro, 1000)); put("cv_D4_vs_D1", boot(d(A4), d(A1), idx, macro, 1000))
    mit = np.array([not (g.startswith("svdb") or g.startswith("incart")) for g in GR[idx]])
    put("cv_pop_all_vs_mit_on_mit", boot({s: v[0][mit] for s, v in A4.items()}, {s: v[0][mit] for s, v in M4.items()}, idx[mit], macro, 1000))
    for db in ("svdb", "incart"):
        m = np.array([g.startswith(db) for g in GR[idx]])
        put("cv_shift_%s_in_vs_out" % db, boot({s: v[0][m] for s, v in A4.items()}, {s: v[0][m] for s, v in M4.items()}, idx[m], macro, 1000))
