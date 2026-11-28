"""Window cache, RR features and subject grouping for the European ST-T Database (confirmatory test)."""
import os as _os
WORK = _os.environ.get("ECG_WORK", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "work"))  # data, runs and outputs live here
import os, sys, glob, numpy as np, wfdb
from scipy.signal import resample
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rev_data import window_label, BEAT_SYMBOLS
import rev_np as R
root = (WORK + "/data/edb/1.0.0"); recs = sorted(os.path.basename(f)[:-4] for f in glob.glob(root + "/*.dat"))
X, Y, G, SUB, LEADS = [], [], [], {}, {}
for r in recs:
    rec = wfdb.rdrecord(os.path.join(root, r)); ann = wfdb.rdann(os.path.join(root, r), "atr"); fs = rec.fs; sig = rec.p_signal[:, :2]
    c = [x.strip() for x in rec.comments]; key = "|".join(x for x in c if x and not x.lower().startswith(("medication", "recorder")))
    SUB[r] = key if key else "rec:" + r; LEADS[r] = "+".join(rec.sig_name[:2])
    samp = np.asarray(ann.sample); sym = np.asarray(ann.symbol); keep = np.array([s in BEAT_SYMBOLS for s in sym], bool); bs, by = samp[keep], sym[keep]
    spw = int(round(10.0 * fs))
    for w in range(sig.shape[0] // spw):
        a, b = w * spw, (w + 1) * spw; lab = window_label(by[(bs >= a) & (bs < b)])
        if lab < 0: continue
        ch = []
        for k in range(2):
            x = resample(np.nan_to_num(sig[a:b, k].astype(np.float64)), 1250); sd = x.std(); ch.append(((x - x.mean()) / sd if sd > 1e-6 else x - x.mean()).astype(np.float32))
        X.append(np.stack(ch, 0)); Y.append(lab); G.append(r)
X = np.stack(X); Y = np.asarray(Y); G = np.asarray(G)
keys = sorted(set(SUB.values())); sid = {k: "edb:s%02d" % i for i, k in enumerate(keys)}; S = np.array([sid[SUB[g]] for g in G])
F = R.rr_features(X, 125)
np.savez_compressed(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "caches", "edb.npz"), X=X, y=Y, groups=G, subjects=S, F=F)
print(X.shape, dict(zip(*[a.tolist() for a in np.unique(Y, return_counts=True)])), len(recs), "records", len(keys), "subjects"); print(sorted(set(LEADS.values())))
