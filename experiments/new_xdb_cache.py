"""Window caches for external databases (SVDB, INCART) with the same windowing/labelling as rev_data."""
import os as _os
WORK = _os.environ.get("ECG_WORK", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "work"))  # data, runs and outputs live here
import os, sys, glob, numpy as np, wfdb
from scipy.signal import resample
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rev_data import window_label, BEAT_SYMBOLS
def build(root, out, chans, target_fs=125, window_sec=10.0):
    recs = sorted({os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(root, "*.dat"))})
    X, Y, G, NB = [], [], [], []
    for r in recs:
        rec = wfdb.rdrecord(os.path.join(root, r)); ann = wfdb.rdann(os.path.join(root, r), "atr")
        names = rec.sig_name; idx = [names.index(c) if isinstance(c, str) else c for c in chans]
        sig = rec.p_signal[:, idx]; fs = rec.fs
        samp = np.asarray(ann.sample); sym = np.asarray(ann.symbol)
        keep = np.array([s in BEAT_SYMBOLS for s in sym], bool); bs, by = samp[keep], sym[keep]
        spw = int(round(window_sec * fs)); tlen = int(round(window_sec * target_fs))
        for w in range(sig.shape[0] // spw):
            a, b = w * spw, (w + 1) * spw; m = (bs >= a) & (bs < b)
            lab = window_label(by[m])
            if lab < 0: continue
            ch = []
            for c in range(sig.shape[1]):
                x = np.nan_to_num(sig[a:b, c].astype(np.float64)); x = resample(x, tlen)
                sd = x.std(); x = (x - x.mean()) / sd if sd > 1e-6 else x - x.mean(); ch.append(x.astype(np.float32))
            X.append(np.stack(ch, 0)); Y.append(lab); G.append(r)
    X = np.stack(X); Y = np.asarray(Y); G = np.asarray(G)
    np.savez_compressed(out, X=X, y=Y, groups=G)
    print(out, X.shape, dict(zip(*[a.tolist() for a in np.unique(Y, return_counts=True)])), len(set(G)), "records")
if __name__ == "__main__":
    build((WORK + "/data/svdb/1.0.0"), "caches/svdb.npz", [0, 1])
    build((WORK + "/data/incartdb/1.0.0"), "caches/incart.npz", ["II", "V1"])
