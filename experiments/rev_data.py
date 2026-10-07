"""MIT-BIH window cache builder for the revision experiments (numpy only).

Reproduces the labelling of the original pipeline (data_mitbih.py) exactly:
  * 10-s non-overlapping windows, per-window per-channel z-score,
  * AAMI EC57 grouping of beat symbols -> {0 normal, 1 SVEB, 2 VEB},
  * windows containing only unknown/paced/non-beat annotations are dropped,
and adds what the reviewers asked for:
  * configurable target sampling rate (125 / 250 / 360 Hz),
  * single-lead (MLII) or two-lead input,
  * the rhythm annotation in effect at the window centre (e.g. AFIB, N, SVTA),
  * record id and window index for every window.

Usage:  python rev_data.py <mitdb_dir> <out.npz> [--fs 125] [--leads both|MLII]
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np
from scipy.signal import resample

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wfdb_lite as W

NORMAL = set("NLRej")
SVEB = set("AaJS")
VEB = set("VE")
FUSION = set("F")
BEAT_SYMBOLS = set("NLRejAaJSVEF/fQ")
CLASS_NAMES = ("normal", "supraventricular", "ventricular")


def window_label(symbols) -> int:
    s = set(symbols)
    if s & VEB:
        return 2
    if s & SVEB:
        return 1
    if s & (NORMAL | FUSION):
        return 0
    return -1


def list_records(root):
    return sorted({os.path.splitext(os.path.basename(f))[0]
                   for f in glob.glob(os.path.join(root, "*.dat"))})


def rhythm_track(samp, sym, aux, n_samp):
    """Return (starts, names): rhythm segments from '+' annotations."""
    starts, names = [0], ["N"]
    for s, y, a in zip(samp, sym, aux):
        if y == "+" and a:
            name = a.strip().lstrip("(").rstrip("\x00").strip()
            starts.append(int(s)); names.append(name)
    return np.asarray(starts), names


def rhythm_at(starts, names, t):
    i = int(np.searchsorted(starts, t, side="right") - 1)
    return names[max(i, 0)]


def load_record(root, rec, target_fs=125, window_sec=10.0, leads="both", standardize=True):
    base = os.path.join(root, rec)
    hdr = W.read_header(base)
    sig = W.read_signal_212(base, hdr)                      # [L, C] mV
    samp, sym, aux = W.read_annotation(base)
    fs = hdr["fs"]
    if leads == "MLII":
        ch = hdr["names"].index("MLII") if "MLII" in hdr["names"] else 0
        sig = sig[:, ch:ch + 1]
    starts, rnames = rhythm_track(samp, sym, aux, sig.shape[0])
    keep = np.array([x in BEAT_SYMBOLS for x in sym], bool)
    bsamp, bsym = samp[keep], sym[keep]
    spw = int(round(window_sec * fs)); tlen = int(round(window_sec * target_fs))
    n_win = sig.shape[0] // spw
    X, y, widx, rhy = [], [], [], []
    for w in range(n_win):
        a, b = w * spw, (w + 1) * spw
        m = (bsamp >= a) & (bsamp < b)
        lab = window_label(bsym[m])
        if lab < 0:
            continue
        seg = sig[a:b, :]
        chans = []
        for c in range(seg.shape[1]):
            x = np.nan_to_num(seg[:, c].astype(np.float64))
            if len(x) != tlen:
                x = resample(x, tlen)
            if standardize:
                sd = x.std()
                x = (x - x.mean()) / sd if sd > 1e-6 else x - x.mean()
            chans.append(x.astype(np.float32))
        X.append(np.stack(chans, 0)); y.append(lab); widx.append(w)
        rhy.append(rhythm_at(starts, rnames, (a + b) // 2))
    if not X:
        return None
    return (np.stack(X, 0), np.asarray(y, np.int64), np.asarray([rec] * len(y)),
            np.asarray(widx, np.int64), np.asarray(rhy))


def build_cache(root, out, target_fs=125, leads="both", window_sec=10.0, verbose=True):
    recs = list_records(root)
    if not recs:
        raise FileNotFoundError("no .dat files under %s" % root)
    parts = []
    for r in recs:
        try:
            p = load_record(root, r, target_fs, window_sec, leads)
        except Exception as e:                       # noqa
            print("[skip]", r, e); continue
        if p is not None:
            parts.append(p)
    X = np.concatenate([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
    groups = np.concatenate([p[2] for p in parts]); widx = np.concatenate([p[3] for p in parts])
    rhy = np.concatenate([p[4] for p in parts])
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, X=X, y=y, groups=groups, win_idx=widx, rhythm=rhy,
                        fs=target_fs, leads=leads, window_sec=window_sec)
    if verbose:
        u, c = np.unique(y, return_counts=True)
        print("[cache] %s: %d records -> %s windows; classes %s; rhythms %s"
              % (out, len(parts), X.shape, dict(zip([CLASS_NAMES[k] for k in u], c.tolist())),
                 dict(zip(*[a.tolist() for a in np.unique(rhy, return_counts=True)]))))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("out")
    ap.add_argument("--fs", type=int, default=125); ap.add_argument("--leads", default="both")
    a = ap.parse_args()
    build_cache(a.root, a.out, a.fs, a.leads)
