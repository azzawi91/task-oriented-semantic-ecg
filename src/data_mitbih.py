"""MIT-BIH Arrhythmia loader (window-level AAMI labels).

Fully OPEN data (no credentialing): MIT-BIH Arrhythmia Database, 48 records,
2-lead ambulatory ECG @ 360 Hz, with expert beat annotations.
    https://physionet.org/content/mitdb/1.0.0/

Each fixed-length window is given an AAMI-style label based on the beats it
contains, so the downstream task is *window-level arrhythmia detection* -- the
three-class task with a binary "abnormal vs normal" operating-point view:

    0 = normal           (window contains only normal / fusion / bundle-branch beats)
    1 = supraventricular  (window contains an SVEB beat, no ventricular ectopy)
    2 = ventricular       (window contains a ventricular-ectopic beat)

Binary abnormal view used by metrics.py = (1 or 2) = arrhythmia present.
This maps the semantic-communication question directly: does the compressed /
channel-degraded signal still let the receiver detect the arrhythmia?

Beat-symbol -> AAMI grouping follows the standard ANSI/AAMI EC57 convention
(de Chazal et al.). Paced records (/, f) contribute only when a clear N/S/V
context exists; paced-only windows are dropped rather than mislabelled.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import wfdb
from scipy.signal import resample

# ---------------------------------------------------------------------------
# AAMI EC57 grouping of MIT-BIH beat-annotation symbols
# ---------------------------------------------------------------------------
NORMAL = set("NLRej")    # normal, LBBB, RBBB, atrial escape, nodal (junctional) escape
SVEB = set("AaJS")       # atrial premature, aberrated atrial premature, nodal, supravent.
VEB = set("VE")          # premature ventricular contraction, ventricular escape
FUSION = set("F")        # fusion of ventricular and normal
# Beat symbols that are genuine *beats* (everything else, e.g. '+', '~', '|', is rhythm/noise)
BEAT_SYMBOLS = set("NLRejAaJSVEF/fQ")


@dataclass
class MITBIHConfig:
    target_fs: int = 125          # Hz after resampling; 250/360 Hz are studied in revision/ (Table VII of the manuscript)
    window_sec: float = 10.0      # one decision per 10 s window
    standardize: bool = True      # per-window, per-channel z-score
    drop_unknown: bool = True     # drop windows whose only beats are unknown/paced


CLASS_NAMES = ("normal", "supraventricular", "ventricular")


def list_records(root: str):
    return sorted({os.path.splitext(os.path.basename(f))[0]
                   for f in glob.glob(os.path.join(root, "*.dat"))})


def _window_label(symbols) -> int:
    s = set(symbols)
    if s & VEB:
        return 2
    if s & SVEB:
        return 1
    if s & (NORMAL | FUSION):
        return 0
    return -1   # only unknown/paced/non-beat -> caller drops


def load_record_windows(root: str, rec: str, cfg: MITBIHConfig):
    """One MIT-BIH record -> (X [N,C,T] float32, y [N] int64, groups [N] str)."""
    base = os.path.join(root, rec)
    record = wfdb.rdrecord(base)
    ann = wfdb.rdann(base, "atr")
    sig = np.asarray(record.p_signal, dtype=np.float32)      # [L, C]
    fs = float(record.fs)
    C = sig.shape[1]

    samp = np.asarray(ann.sample)
    sym = np.asarray(ann.symbol)
    keep = np.array([x in BEAT_SYMBOLS for x in sym], dtype=bool)
    samp, sym = samp[keep], sym[keep]

    spw = int(round(cfg.window_sec * fs))                    # samples per window (raw fs)
    tlen = int(round(cfg.window_sec * cfg.target_fs))        # samples per window (target fs)
    if spw <= 0:
        return (np.empty((0, C, tlen), np.float32),
                np.empty((0,), np.int64), [])
    n_windows = sig.shape[0] // spw

    X, y = [], []
    for w in range(n_windows):
        a, b = w * spw, (w + 1) * spw
        m = (samp >= a) & (samp < b)
        lab = _window_label(sym[m])
        if lab < 0:
            if cfg.drop_unknown:
                continue
            lab = 0
        seg = sig[a:b, :]
        chans = []
        for c in range(C):
            x = np.nan_to_num(seg[:, c], nan=0.0)
            if len(x) != tlen:
                x = resample(x, tlen)
            if cfg.standardize:
                sd = x.std()
                x = (x - x.mean()) / sd if sd > 1e-6 else x - x.mean()
            chans.append(x.astype(np.float32))
        X.append(np.stack(chans, 0))
        y.append(lab)

    if not X:
        return (np.empty((0, C, tlen), np.float32),
                np.empty((0,), np.int64), [])
    return np.stack(X, 0), np.asarray(y, np.int64), [rec] * len(y)


def build_dataset(root: str, cfg: MITBIHConfig = MITBIHConfig(),
                  max_records=None, verbose=True):
    """Read all MIT-BIH records -> (X, y, groups).

    `groups` holds the record id for each window so a patient/record-disjoint
    (inter-patient) split can be done -- the clinically honest evaluation for
    arrhythmia, avoiding train/test leakage of the same patient's beats.
    """
    recs = list_records(root)
    if not recs:
        raise FileNotFoundError(f"No .dat records found under {root}")
    if max_records:
        recs = recs[:max_records]

    Xs, ys, groups = [], [], []
    for r in recs:
        try:
            X, yv, g = load_record_windows(root, r, cfg)
        except Exception as e:
            if verbose:
                print(f"[skip] {r}: {e}")
            continue
        if len(yv):
            Xs.append(X)
            ys.append(yv)
            groups += g

    if not Xs:
        raise RuntimeError("No labelled windows assembled from MIT-BIH.")
    X = np.concatenate(Xs, 0)
    y = np.concatenate(ys, 0)
    groups = np.asarray(groups)
    if verbose:
        u, c = np.unique(y, return_counts=True)
        dist = {CLASS_NAMES[k]: int(n) for k, n in zip(u, c)}
        print(f"[mitbih] {len(set(groups))} records -> {len(y)} windows; "
              f"channels={X.shape[1]}; class dist: {dist}")
    return X, y, groups
