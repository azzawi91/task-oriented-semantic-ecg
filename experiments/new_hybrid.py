"""Hybrid code: learned 4-byte code concatenated with quantised RR-interval statistics, read by the same
class-balanced logistic regression as the RR baseline. DS2 and the two held-out databases."""
import os as _os
WORK = _os.environ.get("ECG_WORK", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "work"))  # data, runs and outputs live here
import os, sys, json, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "new_stats.py")).read(); sys.argv = [sys.argv[0], "none"]; exec(src.split("part = sys.argv[1]")[0])
import torch, rev_np as R, rev_splits as S
from rev_model import Encoder, Decoder
torch.set_num_threads(1); X = C["X"]
Fall = np.concatenate([np.load(os.path.join(CD, "rr_features_fs125.npy")), np.load(os.path.join(CD, "rr_svdb.npy")), np.load(os.path.join(CD, "rr_incart.npy"))])
is_sv = np.char.startswith(GR, "svdb_"); is_in = np.char.startswith(GR, "incart_")
def qz(Z): return np.clip(np.round(Z * 127), -128, 127) / 127.0
def enc_of(f):
    ck = torch.load(f, map_location="cpu"); c = ck["cfg"]
    e = Encoder(ck["in_ch"], use_transformer=c["use_transformer"], n_layers=c["n_layers"], bits=c["bits"], n_classes=c["n_classes"], pos_enc=c.get("pos_enc", False), latent=c.get("latent", 32))
    e.load_state_dict(ck["enc"]); e.eval(); return e
def lat(e, idx):
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), 256): out.append(e(torch.tensor(X[idx[i:i + 256]]))[3].numpy())
    return np.concatenate(out)
RRS = list(range(10))            # the ten RR-interval statistics (10 bytes); all 22 features = 22 bytes
def run(cfg, test_idx, tag):
    Pl, Ph10, Ph22, Pr, Pz = {}, {}, {}, {}, {}
    for f in sorted(glob.glob("%s/%s/seed*.npz" % (RES, cfg))):
        s = int(f.split("seed")[1].split(".")[0]); z = np.load(f); tr = z["idx_tr"]; Ztr = qz(z["Z_tr"])
        if np.array_equal(test_idx, z["idx_te"]): Zte = qz(z["Z_te"]); Pl[s] = z["P_te"]
        else:
            e = enc_of(f[:-4] + ".pt"); Zte = qz(lat(e, test_idx)); ext = np.load((WORK + "/ext_pred_%s.npz") % tag); Pl[s] = ext["s%d" % s]
        sc = R.dim_scale(Fall[tr]); Ftr = R.quantize(Fall[tr], 8, sc); Fte = R.quantize(Fall[test_idx], 8, sc); ytr = Y[tr]
        Pr[s] = R.logreg_readout(Ftr, ytr, Fte, seed=s)
        Pz[s] = R.logreg_readout(Ztr, ytr, Zte, seed=s)
        Ph10[s] = R.logreg_readout(np.concatenate([Ztr, Ftr[:, RRS]], 1), ytr, np.concatenate([Zte, Fte[:, RRS]], 1), seed=s)
        Ph22[s] = R.logreg_readout(np.concatenate([Ztr, Ftr], 1), ytr, np.concatenate([Zte, Fte], 1), seed=s)
    np.savez_compressed((WORK + "/hyb_pred_%s.npz") % tag, idx=test_idx, **{"h10_%d" % s: v for s, v in Ph10.items()}, **{"h22_%d" % s: v for s, v in Ph22.items()}, **{"rr_%d" % s: v for s, v in Pr.items()})
    for nm, H in (("hyb14", Ph10), ("hyb26", Ph22)):
        for mname, m in (("macro", macro), ("abn", abn)):
            put("%s_%s_vs_learned_%s" % (nm, tag, mname), boot(H, Pl, test_idx, m, 1000))
            put("%s_%s_vs_rr_%s" % (nm, tag, mname), boot(H, Pr, test_idx, m, 1000))
    put("zlogreg_%s_vs_learned_macro" % tag, boot(Pz, Pl, test_idx, macro, 1000))
base = load("pool_all_cnn_D4"); te = base[0][1]
run("pool_all_cnn_D4", te, "ds2")
run("pool_noSV_cnn_D4", np.where(is_sv)[0], "svdb")
run("pool_noIN_cnn_D4", np.where(is_in)[0], "incart")
