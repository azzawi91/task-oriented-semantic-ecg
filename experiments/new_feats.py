"""Per-window physiological and nuisance descriptors for the probing analysis (MIT-BIH, 125 Hz cache order)."""
import os as _os
WORK = _os.environ.get("ECG_WORK", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "work"))  # data, runs and outputs live here
import os, sys, numpy as np, wfdb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rev_data import BEAT_SYMBOLS, VEB, SVEB
d = np.load("caches/mitbih_fs125_both.npz", allow_pickle=True); X, g, w = d["X"], d["groups"], d["win_idx"]
root = (WORK + "/data/mitdb/1.0.0"); F = np.zeros((len(g), 8), np.float32); ann = {}
for i in range(len(g)):
    r = str(g[i])
    if r not in ann:
        a = wfdb.rdann(os.path.join(root, r), "atr"); s = np.asarray(a.sample); y = np.asarray(a.symbol)
        k = np.array([c in BEAT_SYMBOLS for c in y]); ann[r] = (s[k], y[k])
    s, y = ann[r]; a0, b0 = w[i] * 3600, (w[i] + 1) * 3600; m = (s >= a0) & (s < b0)
    bs = s[m]; rr = np.diff(bs) / 360.0 * 1000.0
    F[i, 0] = m.sum() * 6.0                                            # heart rate (beats/min)
    F[i, 1] = rr.std() if len(rr) > 1 else 0.0                         # SDRR (ms)
    F[i, 2] = np.sqrt(np.mean(np.diff(rr) ** 2)) if len(rr) > 2 else 0 # RMSSD (ms)
    F[i, 3] = sum(c in VEB for c in y[m])                              # ventricular ectopic beats in window
    F[i, 4] = sum(c in SVEB for c in y[m])                             # supraventricular ectopic beats
    P = np.abs(np.fft.rfft(X[i, 0])) ** 2; f = np.fft.rfftfreq(X.shape[2], 1 / 125.0); tot = P[1:].sum() + 1e-12
    F[i, 5] = P[(f > 0) & (f < 0.7)].sum() / tot                        # baseline-wander power fraction
    F[i, 6] = P[f > 40].sum() / tot                                     # high-frequency (noise/EMG) power fraction
    F[i, 7] = np.abs(X[i, 0]).max()                                     # peak amplitude in SD units (crest factor)
np.savez("caches/mitbih_feats.npz", F=F, names=np.array(["HR", "SDRR", "RMSSD", "nVEB", "nSVEB", "LFfrac", "HFfrac", "crest"]))
print(F.mean(0).round(3), F.std(0).round(3))
