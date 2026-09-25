"""N-seed harness for the OPEN MIT-BIH arrhythmia track (Idea A / Path 2).

Reuses the model, fit, and metric code from `run_real_experiment.py` unchanged;
only the *dataset source* is swapped to MIT-BIH (`data_mitbih.py`). This keeps a
single documented model/metric path so numbers are comparable across datasets.

Usage
-----
  # torch-free logistic-regression baseline (proves the real data path):
  python run_mitbih_experiment.py --mitbih-root \
      "../../New article - Me only/physionet.org/files/mitdb/1.0.0" \
      --seeds 10 --model logreg --split patient

  # the paper's semantic encoder (requires torch):
  python run_mitbih_experiment.py --mitbih-root <path> --seeds 10 --model encoder

Splits
------
  --split patient : record/patient-disjoint (inter-patient, AAMI-honest) -- default
  --split random  : stratified random over pooled windows (intra-patient; optimistic)
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np

import metrics as M
import run_real_experiment as R     # reuse summarize / fit_predict_* / stratified_split
import data_mitbih as dmb


def patient_split(y, groups, seed, fracs=(0.6, 0.2, 0.2), min_per_class=2, tries=200):
    """Record-disjoint split. Re-draw record partitions until every split
    contains at least `min_per_class` examples of every class (so metrics and
    the calibrated operating point are well defined). Falls back to the last
    draw if no perfect partition is found."""
    rng = np.random.default_rng(seed)
    recs = np.array(sorted(set(groups)))
    classes = np.unique(y)
    last = None
    for _ in range(tries):
        rng.shuffle(recs)
        n = len(recs)
        a, b = int(fracs[0] * n), int((fracs[0] + fracs[1]) * n)
        sets = (set(recs[:a]), set(recs[a:b]), set(recs[b:]))
        idx = [np.array([i for i, g in enumerate(groups) if g in S]) for S in sets]
        last = idx
        ok = all(len(ix) > 0 and all((y[ix] == k).sum() >= min_per_class for k in classes)
                 for ix in idx)
        if ok:
            return idx[0], idx[1], idx[2]
    return last[0], last[1], last[2]


def run(args):
    X, y, groups = dmb.build_dataset(
        args.mitbih_root, dmb.MITBIHConfig(target_fs=args.target_fs),
        max_records=args.max_records)

    fit = R.fit_predict_logreg if args.model == "logreg" else R.fit_predict_encoder
    keys = ["auroc", "auprc", "macro_f1", "sens_at_spec", "f1_at_op", "alarm_rate"]
    per = {k: [] for k in keys}

    for s in range(args.seeds):
        if args.split == "patient":
            tr, va, te = patient_split(y, groups, seed=s)
        else:
            tr, va, te = R.stratified_split(y, seed=s)
        pv, pt = fit(X[tr], y[tr], X[va], X[te], seed=s)
        per["auroc"].append(M.auroc_multiclass(y[te], pt))
        per["auprc"].append(M.auprc_multiclass(y[te], pt))
        per["macro_f1"].append(M.macro_f1(y[te], pt.argmax(1)))
        yv, sv = M.to_deterioration_binary(y[va], pv)
        yt, st = M.to_deterioration_binary(y[te], pt)
        op = M.calibrated_operating_point(yv, sv, yt, st, args.target_spec)
        per["sens_at_spec"].append(op["sensitivity"])
        per["f1_at_op"].append(op["f1"])
        per["alarm_rate"].append(op["alarm_rate"])

    summary = {k: dict(zip(("mean", "ci_lo", "ci_hi"), M.bootstrap_ci(v)))
               for k, v in per.items()}
    summary["_meta"] = {"dataset": "mitbih", "seeds": args.seeds, "model": args.model,
                        "split": args.split, "target_specificity": args.target_spec,
                        "n_windows": int(len(y)), "n_records": int(len(set(groups))),
                        "channels": int(X.shape[1])}

    print(f"\n=== MIT-BIH | {args.model} | {args.split}-split | "
          f"{args.seeds} seeds (mean [95% CI]) ===")
    for k in keys:
        m = summary[k]
        print(f"  {k:14s}: {M.fmt_ci(m['mean'], m['ci_lo'], m['ci_hi'])}")

    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", f"mitbih_metrics_{args.model}_{args.split}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[written] {out}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mitbih-root", required=True,
                    help="path to mitdb 1.0.0 directory (contains 100.dat ...)")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--model", choices=["encoder", "logreg"], default="logreg")
    ap.add_argument("--split", choices=["patient", "random"], default="patient")
    ap.add_argument("--target-spec", type=float, default=0.90)
    ap.add_argument("--target-fs", type=int, default=125)
    ap.add_argument("--max-records", type=int, default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
