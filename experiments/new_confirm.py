"""Confirmatory test on the European ST-T Database, as fixed in CONFIRMATION_PLAN.md. Run once."""
import os as _os
WORK = _os.environ.get("ECG_WORK", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "work"))  # data, runs and outputs live here
import os, sys, json, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "new_stats.py")).read(); sys.argv = [sys.argv[0], "none"]; exec(src.split("part = sys.argv[1]")[0])
import torch, rev_np as R
from rev_model import Encoder, Decoder, quantize_ste
torch.set_num_threads(1)
E = np.load(os.path.join(CD, "edb.npz"), allow_pickle=True); XE, YE, SE, FE = E["X"], E["y"], E["subjects"].astype(str), E["F"]
Fall = np.concatenate([np.load(os.path.join(CD, "rr_features_fs125.npy")), np.load(os.path.join(CD, "rr_svdb.npy")), np.load(os.path.join(CD, "rr_incart.npy"))])
def qz(Z): return np.clip(np.round(Z * 127), -128, 127) / 127.0
def run_model(f):
    ck = torch.load(f, map_location="cpu"); c = ck["cfg"]
    enc = Encoder(ck["in_ch"], use_transformer=c["use_transformer"], n_layers=c["n_layers"], bits=c["bits"], n_classes=c["n_classes"], pos_enc=c.get("pos_enc", False), latent=c.get("latent", 32))
    dec = Decoder(latent=c.get("latent", 32), n_classes=c["n_classes"]); enc.load_state_dict(ck["enc"]); dec.load_state_dict(ck["dec"]); enc.eval(); dec.eval(); Z, Pp = [], []
    with torch.no_grad():
        for i in range(0, len(XE), 512):
            z = enc(torch.tensor(XE[i:i + 512]))[3]; Z.append(z.numpy()); Pp.append(dec(quantize_ste(z, 8, False)).softmax(-1).numpy())
    return np.concatenate(Z), np.concatenate(Pp)
H, L4, L32, RRp = {}, {}, {}, {}
for s in range(10):
    f = "%s/pool_all_cnn_D4/seed%d" % (RES, s); z = np.load(f + ".npz"); tr = z["idx_tr"]; Zte, L4[s] = run_model(f + ".pt")
    sc = R.dim_scale(Fall[tr]); Ftr = R.quantize(Fall[tr], 8, sc); Fte = R.quantize(FE, 8, sc)
    H[s] = R.logreg_readout(np.concatenate([qz(z["Z_tr"]), Ftr[:, :10]], 1), Y[tr], np.concatenate([qz(Zte), Fte[:, :10]], 1), seed=s)
    RRp[s] = R.logreg_readout(Ftr, Y[tr], Fte, seed=s)
    _, L32[s] = run_model("%s/pool_all_cnn_D32/seed%d.pt" % (RES, s)); print("seed", s, flush=True)
np.savez_compressed((WORK + "/confirm_pred_edb.npz"), y=YE, subjects=SE, **{"h_%d" % s: H[s] for s in H}, **{"l4_%d" % s: L4[s] for s in H}, **{"l32_%d" % s: L32[s] for s in H}, **{"rr_%d" % s: RRp[s] for s in H})
# bootstrap over subjects of the new database: reuse boot() by swapping the global label and cluster arrays
Y = YE; G = SE; idx = np.arange(len(YE))
def cls(k): return lambda P, y: auc_bin(P[:, k].astype(np.float64), (y == k).astype(int))
put("confirm_edb_hyb14_vs_learned32_macro", boot(H, L32, idx, macro, 2000))
for nm, B in (("learned", L4), ("rr", RRp)): put("confirm_edb_hyb14_vs_%s_macro" % nm, boot(H, B, idx, macro, 2000))
for nm, B in (("learned32", L32), ("learned", L4), ("rr", RRp)): put("confirm_edb_hyb14_vs_%s_abn" % nm, boot(H, B, idx, abn, 2000))
for k, c in enumerate("NSV"):
    for nm, B in (("learned32", L32), ("rr", RRp)): put("confirm_edb_hyb14_vs_%s_cls%s" % (nm, c), boot(H, B, idx, cls(k), 1000))
