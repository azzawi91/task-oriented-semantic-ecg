"""Hybrid code under patient-level five-fold cross-validation (153 patients): out-of-fold predictions of the
14-byte hybrid code, the 22-byte RR code, and the learned 4- and 32-byte codes."""
import os as _os
WORK = _os.environ.get("ECG_WORK", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "work"))  # data, runs and outputs live here
import os, sys, json, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "new_stats.py")).read(); sys.argv = [sys.argv[0], "none"]; exec(src.split("part = sys.argv[1]")[0])
import rev_np as R
Fall = np.concatenate([np.load(os.path.join(CD, "rr_features_fs125.npy")), np.load(os.path.join(CD, "rr_svdb.npy")), np.load(os.path.join(CD, "rr_incart.npy"))])
def qz(Z): return np.clip(np.round(Z * 127), -128, 127) / 127.0
H, RRp, L4, L32 = {}, {}, {}, {}
for s in range(3):
    ph, pr, p4, p32, ii = [], [], [], [], []
    for k in range(5):
        z = np.load("%s/cv_all_D4_f%d/seed%d.npz" % (RES, k, s)); tr, te_ = z["idx_tr"], z["idx_te"]
        sc = R.dim_scale(Fall[tr]); Ftr = R.quantize(Fall[tr], 8, sc); Fte = R.quantize(Fall[te_], 8, sc)
        ph.append(R.logreg_readout(np.concatenate([qz(z["Z_tr"]), Ftr[:, :10]], 1), Y[tr], np.concatenate([qz(z["Z_te"]), Fte[:, :10]], 1), seed=s))
        pr.append(R.logreg_readout(Ftr, Y[tr], Fte, seed=s)); p4.append(z["P_te"]); ii.append(te_)
        z32 = np.load("%s/cv_all_D32_f%d/seed%d.npz" % (RES, k, s)); assert np.array_equal(z32["idx_te"], te_); p32.append(z32["P_te"])
    idx = np.concatenate(ii); o = np.argsort(idx); idx = idx[o]
    H[s], RRp[s], L4[s], L32[s] = (np.concatenate(v)[o] for v in (ph, pr, p4, p32))
np.savez_compressed((WORK + "/hyb_pred_cv.npz"), idx=idx, **{"h10_%d" % s: v for s, v in H.items()}, **{"rr_%d" % s: v for s, v in RRp.items()})
for nm, B in (("learned32", L32), ("learned", L4), ("rr", RRp)):
    put("hyb14_cv_vs_%s_macro" % nm, boot(H, B, idx, macro, 1000)); put("hyb14_cv_vs_%s_abn" % nm, boot(H, B, idx, abn, 1000))
put("cv_learned_vs_rr_macro", boot(L4, RRp, idx, macro, 1000))
for db in ("mit", "svdb", "incart"):
    m = np.array([(g.startswith(db + "_") if db != "mit" else not (g.startswith("svdb") or g.startswith("incart"))) for g in GR[idx]])
    put("hyb14_cv%s_vs_learned32_macro" % db, boot({s: v[m] for s, v in H.items()}, {s: v[m] for s, v in L32.items()}, idx[m], macro, 1000))
