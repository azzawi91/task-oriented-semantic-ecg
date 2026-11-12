"""Wavelet-decoded copies of the pooled cache at each byte budget (same codec as new_classic.py)."""
import os as _os
WORK = _os.environ.get("ECG_WORK", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "work"))  # data, runs and outputs live here
import sys, json, numpy as np
from new_classic import codec, prd
from rev_splits import DS2
d = np.load("caches/pool_all.npz", allow_pickle=True); X = d["X"]; te = np.isin(d["groups"], DS2); res = {}
for B in [int(v) for v in sys.argv[1:]]:
    Xr, mb, mk = codec(X, "dwt", B); Xr = Xr.astype(np.float32)
    np.savez("caches/pool_dwt_%d.npz" % B, X=Xr, y=d["y"], groups=d["groups"])
    res[str(B)] = dict(mean_bytes=mb, mean_nonzero=mk, prd_ds2=prd(X[te], Xr[te])); print(B, res[str(B)], flush=True)
    json.dump(res, open((WORK + "/pool_dwt_prd_%s.json") % sys.argv[1], "w"), indent=1)
