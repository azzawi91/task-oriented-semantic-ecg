"""WESAD multimodal loader (Gap 2 multimodal test).

Builds synchronized ECG (chest, 700 Hz) + PPG/BVP (wrist, 64 Hz) windows for the
standard binary stress-detection task (stress vs non-stress). Both modalities are
resampled to a common 64 Hz and segmented into 10 s windows -> X[N,2,640].
Per-subject and resumable (large pickles): saves results/wesad_S<n>.npz.

Usage: python data_wesad.py <WESAD_root> [budget_s]
"""
import sys, os, glob, time, pickle
import numpy as np
from scipy.signal import resample_poly

ROOT = sys.argv[1]
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 38.0
FS, WIN = 64, 10
T = FS * WIN
t0 = time.time()
subs = sorted([os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "S*")) if os.path.isdir(p)],
              key=lambda s: int(s[1:]))
os.makedirs("results", exist_ok=True)

def windowize(ecg, ppg, lab):
    n = min(len(ecg), len(ppg), len(lab)) // T
    X, y = [], []
    for w in range(n):
        a, b = w * T, (w + 1) * T
        lw = lab[a:b]
        vals, cnts = np.unique(lw, return_counts=True)
        maj = vals[cnts.argmax()]
        if maj not in (1, 2, 3, 4):   # 1 base,2 stress,3 amuse,4 medi ; drop 0/5/6/7
            continue
        if (lw == maj).mean() < 0.9:   # window must be mostly one condition
            continue
        seg = np.stack([ecg[a:b], ppg[a:b]], 0).astype(np.float32)
        for c in range(2):
            sd = seg[c].std(); seg[c] = (seg[c] - seg[c].mean()) / sd if sd > 1e-6 else seg[c] - seg[c].mean()
        X.append(seg); y.append(1 if maj == 2 else 0)   # binary stress
    return np.asarray(X, np.float32), np.asarray(y, np.int64)

done = 0
for s in subs:
    out = "results/wesad_%s.npz" % s
    if os.path.exists(out):
        continue
    if time.time() - t0 > BUDGET:
        break
    d = pickle.load(open(os.path.join(ROOT, s, "%s.pkl" % s), "rb"), encoding="latin1")
    ecg = np.asarray(d["signal"]["chest"]["ECG"]).reshape(-1)
    bvp = np.asarray(d["signal"]["wrist"]["BVP"]).reshape(-1)
    lab = np.asarray(d["label"]).reshape(-1)
    ecg64 = resample_poly(ecg, 16, 175)          # 700 -> 64 Hz
    lab64 = lab[np.minimum((np.arange(len(ecg64)) * 700 // 64), len(lab) - 1)]
    X, y = windowize(ecg64, bvp, lab64)
    np.savez_compressed(out, X=X, y=y, subj=np.array([int(s[1:])] * len(y)))
    print("%s -> %d windows (stress %d / non %d)  [%.1fs]" % (s, len(y), int((y==1).sum()), int((y==0).sum()), time.time()-t0))
    done += 1
remaining = [s for s in subs if not os.path.exists("results/wesad_%s.npz" % s)]
print("done this call:", done, "| remaining:", remaining)
