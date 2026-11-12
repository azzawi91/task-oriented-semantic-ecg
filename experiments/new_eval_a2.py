import os as _os
WORK = _os.environ.get("ECG_WORK", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "work"))  # data, runs and outputs live here
import os, sys, json, glob, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rev_metrics as M
from rev_model import Encoder, Decoder, quantize_ste
RES = (WORK + "/res"); torch.set_num_threads(1)
def load_models(name):
    ms = []
    for f in sorted(glob.glob(os.path.join(RES, name, "seed*.pt"))):
        ck = torch.load(f, map_location="cpu"); c = ck["cfg"]
        enc = Encoder(ck["in_ch"], use_transformer=c["use_transformer"], n_layers=c["n_layers"], bits=c["bits"], n_classes=c["n_classes"], pos_enc=c.get("pos_enc", False), latent=c.get("latent", 32)); dec = Decoder(latent=c.get("latent", 32), n_classes=c["n_classes"])
        enc.load_state_dict(ck["enc"]); dec.load_state_dict(ck["dec"]); enc.eval(); dec.eval(); ms.append((enc, dec))
    return ms
def probs(enc, dec, X):
    with torch.no_grad(): return np.concatenate([dec(quantize_ste(enc(torch.tensor(X[i:i + 256]))[3], 8, False)).softmax(-1).numpy() for i in range(0, len(X), 256)])
def msd(v): return [float(np.mean(v)), float(np.std(v)), len(v)]
A2 = {}
for cls_name, tag in (("main", "auroc_decode"), ("abl_cnn_only", "auroc_decode_light")):
    models = load_models(cls_name)[:3] if cls_name == "main" else load_models(cls_name)
    for nm in ("recon_convae", "recon_sameenc_pe"):
        v, prds = [], []
        for f in sorted(glob.glob(os.path.join(RES, nm, "seed*.npz"))):
            z = np.load(f, allow_pickle=True)
            if "X_rec_te" not in z: continue
            Xr = z["X_rec_te"].astype(np.float32); v.append(float(np.mean([M.auroc_multiclass(z["y_te"], probs(e, d, Xr)) for e, d in models]))); prds.append(json.load(open(f[:-4] + ".json"))["test_prd_pct"])
        if v: A2.setdefault(nm, {})[tag] = msd(v); A2[nm]["prd"] = msd(prds)
E = json.load(open((WORK + "/new_eval.json"))); E["recon_decode"] = A2; json.dump(E, open((WORK + "/new_eval.json"), "w"), indent=1); print(A2)
