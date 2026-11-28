"""Repeated patient-level five-fold cross-validation (10 repetitions, each with its own partition and seed).
Usage: python new_cvrep.py hyb|rate|pop     Fast bootstrap over the 153 patients and the repetitions."""
import os as _os
WORK = _os.environ.get("ECG_WORK", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "work"))  # data, runs and outputs live here
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fastboot as FB, rev_splits as S, rev_np as R
RES = (WORK + "/res"); CD = os.path.join(HERE, "caches"); OUTF = (WORK + "/new_stats.json")
C = np.load(os.path.join(CD, "pool_all.npz"), allow_pickle=True); GR = C["groups"].astype(str); Y = C["y"]
pid = S.patient_ids(GR); PID = np.array([p if p is not None else "none" for p in pid])
def path(prefix, rep, k): return "%s/%s_f%d/seed0.npz" % (RES, prefix, k) if rep == 0 else "%s/%s_r%d_f%d/seed%d.npz" % (RES, prefix, rep, k, rep)
def put(k, v):
    OUT = json.load(open(OUTF)); OUT[k] = v; json.dump(OUT, open(OUTF, "w"), indent=1)
    print("%-36s a %.3f b %.3f d %+.3f [%+.3f,%+.3f] 90lo %+.3f p %.3f reps %d pat %d" % (k, v["a"], v["b"], v["diff"], *v["ci95"], v["ci90"][0], v["p"], v["seeds"], v["records"]), flush=True)
def oof(prefix, extra=None):
    out, IDX = {}, None
    for rep in range(10):
        if not all(os.path.exists(path(prefix, rep, k)) for k in range(5)): continue
        Ps, Is, Ex = [], [], []
        for k in range(5):
            z = np.load(path(prefix, rep, k)); Ps.append(z["P_te"]); Is.append(z["idx_te"])
            if extra: Ex.append(extra(z, rep))
        idx = np.concatenate(Is); o = np.argsort(idx); IDX = idx[o]; out[rep] = (np.concatenate(Ps)[o], [np.concatenate([e[j] for e in Ex])[o] for j in range(len(Ex[0]))] if extra else None)
    return out, IDX
part = sys.argv[1]
if part == "hyb":
    Fall = np.concatenate([np.load(os.path.join(CD, "rr_features_fs125.npy")), np.load(os.path.join(CD, "rr_svdb.npy")), np.load(os.path.join(CD, "rr_incart.npy"))])
    qz = lambda Z: np.clip(np.round(Z * 127), -128, 127) / 127.0
    def extra(z, rep):
        tr, te = z["idx_tr"], z["idx_te"]; sc = R.dim_scale(Fall[tr]); Ftr = R.quantize(Fall[tr], 8, sc); Fte = R.quantize(Fall[te], 8, sc)
        return (R.logreg_readout(np.concatenate([qz(z["Z_tr"]), Ftr[:, :10]], 1), Y[tr], np.concatenate([qz(z["Z_te"]), Fte[:, :10]], 1), seed=rep), R.logreg_readout(Ftr, Y[tr], Fte, seed=rep))
    A4, idx = oof("cv_all_D4", extra); A32, idx2 = oof("cv_all_D32"); assert np.array_equal(idx, idx2)
    y = Y[idx]; cl = PID[idx]; H = {r: v[1][0] for r, v in A4.items()}; RRp = {r: v[1][1] for r, v in A4.items()}; L4 = {r: v[0] for r, v in A4.items()}; L32 = {r: v[0] for r, v in A32.items()}
    np.savez_compressed((WORK + "/hyb_pred_cvrep.npz"), idx=idx, **{"h_%d" % r: v for r, v in H.items()}, **{"rr_%d" % r: v for r, v in RRp.items()})
    for kind in ("macro", "abn"):
        for nm, B in (("learned32", L32), ("learned", L4), ("rr", RRp)): put("hyb14_cv_vs_%s_%s" % (nm, kind), FB.boot(H, B, y, cl, kind, 2000))
    put("cv_learned_vs_rr_macro", FB.boot(L4, RRp, y, cl, "macro", 2000))
    for kind in ("clsS", "clsV"): put("hyb14_cv_vs_learned_%s" % kind, FB.boot(H, L4, y, cl, kind, 2000))
    for db in ("mit", "svdb", "incart"):
        m = np.char.startswith(cl, db + ":"); put("hyb14_cv%s_vs_learned32_macro" % db, FB.boot({r: v[m] for r, v in H.items()}, {r: v[m] for r, v in L32.items()}, y[m], cl[m], "macro", 2000))
if part == "rate":
    A4, idx = oof("cv_all_D4"); A32, _ = oof("cv_all_D32"); A1, _ = oof("cv_all_D1"); y = Y[idx]; cl = PID[idx]; d = lambda O: {r: v[0] for r, v in O.items()}
    put("cv_D4_vs_D32", FB.boot(d(A4), d(A32), y, cl)); put("cv_D1_vs_D32", FB.boot(d(A1), d(A32), y, cl)); put("cv_D4_vs_D1", FB.boot(d(A4), d(A1), y, cl))
if part == "pop":
    A4, idx = oof("cv_all_D4"); M4, _ = oof("cv_mit_D4"); y = Y[idx]; cl = PID[idx]
    for key, db in (("cv_pop_all_vs_mit_on_mit", "mit"), ("cv_shift_svdb_in_vs_out", "svdb"), ("cv_shift_incart_in_vs_out", "incart")):
        m = np.char.startswith(cl, db + ":"); put(key, FB.boot({r: v[0][m] for r, v in A4.items()}, {r: v[0][m] for r, v in M4.items()}, y[m], cl[m]))
