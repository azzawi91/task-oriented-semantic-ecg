"""Shared helpers (feature summary, logistic-regression and encoder fitting, splits)
used by run_mitbih_experiment.py. The MIMIC entry point in this file is legacy and
not used by the manuscript.

Produces the threshold-independent metrics (AUROC, AUPRC) AND the calibrated
clinical operating-point metrics for Table VI (sensitivity @ 90% specificity,
F1 @ calibrated threshold), each aggregated over N seeds as mean +/- 95% CI.

Usage
-----
  # Smoke test (synthetic, torch-free) -- proves the harness end to end:
  python run_real_experiment.py --smoke --seeds 10 --model logreg

  # Real run once data is in place:
  python run_real_experiment.py \
      --wdb-root data/mimic4wdb/0.1.0 \
      --clinical-root data/mimic-iv-demo/2.2 \
      --seeds 10 --model encoder

Models
------
  --model encoder : the paper's semantic encoder (requires torch).
  --model logreg  : a light sklearn baseline on summary features; no torch,
                    used to validate the pipeline before the real model runs.
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np

import metrics as M


# ---------------------------------------------------------------------------
# Feature summary (used by the torch-free logreg path)
# ---------------------------------------------------------------------------
def summarize(X):
    """X [N,C,T] -> features [N, C*11] (mean,std,rms + 8 FFT magnitude bins)."""
    N, C, T = X.shape
    feats = []
    for c in range(C):
        xc = X[:, c, :]
        mag = np.abs(np.fft.rfft(xc, axis=1))[:, 1:9]
        feats.append(np.stack([xc.mean(1), xc.std(1),
                               np.sqrt((xc**2).mean(1))], axis=1))
        feats.append(mag)
    return np.concatenate(feats, axis=1)


# ---------------------------------------------------------------------------
# Synthetic smoke dataset (torch-free, class-separable)
# ---------------------------------------------------------------------------
def make_smoke(n=1500, seed=0, fs=125, win_s=10.0):
    rng = np.random.default_rng(seed)
    T = int(fs * win_s)
    t = np.arange(T) / fs
    y = rng.choice(3, size=n, p=[0.85, 0.10, 0.05])
    X = np.zeros((n, 3, T), np.float32)
    for i in range(n):
        hr = {0: 1.2, 1: 1.8, 2: 2.5}[y[i]] + rng.normal(0, 0.05)
        amp = {0: 1.0, 1: 0.7, 2: 0.4}[y[i]]
        X[i, 0] = amp * np.sin(2 * np.pi * hr * t) + 0.05 * rng.standard_normal(T)
        X[i, 1] = 0.8 * np.sin(2 * np.pi * hr * t - 0.2) + 0.04 * rng.standard_normal(T)
        X[i, 2] = np.sin(2 * np.pi * 0.25 * t) + 0.03 * rng.standard_normal(T)
        for c in range(3):
            sd = X[i, c].std()
            X[i, c] = (X[i, c] - X[i, c].mean()) / (sd if sd > 1e-6 else 1.0)
    return X, y


# ---------------------------------------------------------------------------
# Per-seed train/eval
# ---------------------------------------------------------------------------
def stratified_split(y, seed, fracs=(0.6, 0.2, 0.2)):
    rng = np.random.default_rng(seed)
    idx_tr, idx_va, idx_te = [], [], []
    for k in np.unique(y):
        ids = np.where(y == k)[0]
        rng.shuffle(ids)
        n = len(ids)
        a, b = int(fracs[0] * n), int((fracs[0] + fracs[1]) * n)
        idx_tr += ids[:a].tolist(); idx_va += ids[a:b].tolist(); idx_te += ids[b:].tolist()
    return np.array(idx_tr), np.array(idx_va), np.array(idx_te)


def fit_predict_logreg(Xtr, ytr, Xva, Xte, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(summarize(Xtr))
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(sc.transform(summarize(Xtr)), ytr)
    pv = clf.predict_proba(sc.transform(summarize(Xva)))
    pt = clf.predict_proba(sc.transform(summarize(Xte)))
    return _pad_probs(pv, clf.classes_), _pad_probs(pt, clf.classes_)


def _pad_probs(p, classes):
    """Ensure a 3-column probability matrix even if a class is absent in train."""
    full = np.zeros((p.shape[0], 3))
    for j, c in enumerate(classes):
        full[:, int(c)] = p[:, j]
    return full


def fit_predict_encoder(Xtr, ytr, Xva, Xte, seed):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from config import EncoderConfig
    from semantic_encoder import SemanticEncoder, SemanticDecoder, sempq_loss
    torch.manual_seed(seed); np.random.seed(seed)
    cfg = EncoderConfig(in_channels=Xtr.shape[1])
    enc, dec = SemanticEncoder(cfg), SemanticDecoder(cfg)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=3e-3)
    loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                        batch_size=32, shuffle=True)
    for _ in range(6):
        enc.train(); dec.train()
        for xb, yb in loader:
            z, ee, rate = enc(xb); logits = dec(z)
            loss, _ = sempq_loss(logits, ee, z, rate, yb, cfg)
            opt.zero_grad(); loss.backward(); opt.step()
    enc.eval(); dec.eval()
    with torch.no_grad():
        pv = dec(enc(torch.tensor(Xva))[0]).softmax(-1).numpy()
        pt = dec(enc(torch.tensor(Xte))[0]).softmax(-1).numpy()
    return pv, pt


def run(args):
    if args.smoke:
        X, y = make_smoke(n=args.n, seed=123)
        print(f"[smoke] synthetic dataset X={X.shape}")
    else:
        import data_mimic as dm
        ds = dm.build_dataset(args.wdb_root, args.clinical_root,
                              dm.WaveformConfig(), dm.LabelConfig(),
                              max_records=args.max_records)
        X, y = ds.X_np, ds.y_np

    fit = fit_predict_logreg if args.model == "logreg" else fit_predict_encoder
    keys = ["auroc", "auprc", "macro_f1", "sens_at_spec", "f1_at_op", "alarm_rate"]
    per_seed = {k: [] for k in keys}

    for s in range(args.seeds):
        tr, va, te = stratified_split(y, seed=s)
        pv, pt = fit(X[tr], y[tr], X[va], X[te], seed=s)
        per_seed["auroc"].append(M.auroc_multiclass(y[te], pt))
        per_seed["auprc"].append(M.auprc_multiclass(y[te], pt))
        per_seed["macro_f1"].append(M.macro_f1(y[te], pt.argmax(1)))
        yv, sv = M.to_deterioration_binary(y[va], pv)
        yt, st = M.to_deterioration_binary(y[te], pt)
        op = M.calibrated_operating_point(yv, sv, yt, st, args.target_spec)
        per_seed["sens_at_spec"].append(op["sensitivity"])
        per_seed["f1_at_op"].append(op["f1"])
        per_seed["alarm_rate"].append(op["alarm_rate"])

    summary = {k: dict(zip(("mean", "ci_lo", "ci_hi"), M.bootstrap_ci(v)))
               for k, v in per_seed.items()}
    summary["_meta"] = {"seeds": args.seeds, "model": args.model,
                        "target_specificity": args.target_spec,
                        "n_windows": int(len(y)), "smoke": bool(args.smoke)}

    print("\n=== Aggregated over", args.seeds, "seeds (mean [95% CI]) ===")
    for k in keys:
        m = summary[k]
        print(f"  {k:14s}: {M.fmt_ci(m['mean'], m['ci_lo'], m['ci_hi'])}")
    print("\n--- Table VI (calibrated clinical operating-point metrics) ---")
    for label, k in [("AUPRC", "auprc"), ("Sens@90%Spec", "sens_at_spec"),
                     ("F1@op", "f1_at_op")]:
        m = summary[k]
        print(f"  {label:14s}: {M.fmt_ci(m['mean'], m['ci_lo'], m['ci_hi'])}")

    os.makedirs("results", exist_ok=True)
    with open("results/real_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[written] results/real_metrics.json")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wdb-root", default="data/mimic4wdb/0.1.0")
    ap.add_argument("--clinical-root", default="data/mimic-iv-demo/2.2")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--model", choices=["encoder", "logreg"], default="encoder")
    ap.add_argument("--target-spec", type=float, default=0.90)
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--n", type=int, default=1500, help="smoke dataset size")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
