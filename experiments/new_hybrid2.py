"""Controls for the hybrid code: (i) the same number of bytes spent on learned coefficients (32-byte learned code),
(ii) per-class AUROC of learned, RR and hybrid codes. Uses predictions saved by new_hybrid.py."""
import os as _os
WORK = _os.environ.get("ECG_WORK", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "work"))  # data, runs and outputs live here
import os, sys, json, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "new_stats.py")).read(); sys.argv = [sys.argv[0], "none"]; exec(src.split("part = sys.argv[1]")[0])
import torch
from rev_model import Encoder, Decoder, quantize_ste
torch.set_num_threads(1); X = C["X"]
def probs32(cfg, idx):
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, cfg, "seed*.pt"))):
        s = int(f.split("seed")[1].split(".")[0]); z = np.load(f[:-3] + ".npz")
        if np.array_equal(z["idx_te"], idx): out[s] = z["P_te"]; continue
        ck = torch.load(f, map_location="cpu"); c = ck["cfg"]
        enc = Encoder(ck["in_ch"], use_transformer=c["use_transformer"], n_layers=c["n_layers"], bits=c["bits"], n_classes=c["n_classes"], pos_enc=c.get("pos_enc", False), latent=c.get("latent", 32))
        dec = Decoder(latent=c.get("latent", 32), n_classes=c["n_classes"]); enc.load_state_dict(ck["enc"]); dec.load_state_dict(ck["dec"]); enc.eval(); dec.eval(); o = []
        with torch.no_grad():
            for i in range(0, len(idx), 256): o.append(dec(quantize_ste(enc(torch.tensor(X[idx[i:i + 256]]))[3], 8, False)).softmax(-1).numpy())
        out[s] = np.concatenate(o)
    return out
def cls(k): return lambda P, y: auc_bin(P[:, k].astype(np.float64), (y == k).astype(int))
for tag, c32, c4 in (("ds2", "pool_all_cnn_D32", "pool_all_cnn_D4"), ("svdb", "pool_noSV_cnn_D32", "pool_noSV_cnn_D4"), ("incart", "pool_noIN_cnn_D32", "pool_noIN_cnn_D4")):
    h = np.load((WORK + "/hyb_pred_%s.npz") % tag); idx = h["idx"]; seeds = sorted(int(k[4:]) for k in h.files if k.startswith("h10_"))
    H = {s: h["h10_%d" % s] for s in seeds}; RRp = {s: h["rr_%d" % s] for s in seeds}; L32 = probs32(c32, idx)
    if tag == "ds2": L4 = P(c4)
    else: e = np.load((WORK + "/ext_pred_%s.npz") % tag); L4 = {s: e["s%d" % s] for s in seeds}
    put("hyb14_%s_vs_learned32_macro" % tag, boot(H, L32, idx, macro, 1000)); put("hyb14_%s_vs_learned32_abn" % tag, boot(H, L32, idx, abn, 1000))
    for k, nm in enumerate(("N", "S", "V")):
        put("hyb14_%s_vs_learned_cls%s" % (tag, nm), boot(H, L4, idx, cls(k), 1000)); put("hyb14_%s_vs_rr_cls%s" % (tag, nm), boot(H, RRp, idx, cls(k), 1000))
