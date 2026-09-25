"""One-command runner for every training experiment of the revision.

    python run_all.py --mitdb "<path to mitdb/1.0.0>" [--seeds 10] [--skip-ptbxl]

Everything is resumable: finished (config, seed) pairs are skipped, so the
script can be interrupted and re-run. Progress is written to run_log.txt and
status.json next to this file; results go to results_revision/<config>/.

Stages (in order):
  1. window caches      125 Hz two-lead / 125 Hz MLII-only / 250 Hz / 360 Hz
  2. main               published configuration, 10 seeds, de Chazal DS1/DS2
  3. recon baselines    conv-AE, deep-JSCC, same-encoder(+positional enc.) reconstruction
                        (3 seeds) and the matching positional-encoding task model (10 seeds)
  3b. latent width      D = 4 and D = 8 coefficient codecs (4- and 8-byte codes)
  4. ablations          CNN-only, 1 Transformer layer, no latent L2, no rate term,
                        no early exit, task-loss only (10 seeds each)
  5. qat / channel      4-bit and 3-bit QAT, erasure-aware training (10 seeds)
  6. single lead        MLII only (10 seeds)
  7. sampling rate      250 Hz and 360 Hz (10 seeds each)
  8. random split       record-disjoint 60/20/20 (3 seeds)
  9. PTB-XL full        21.8k records, official folds, 5 seeds (downloads ~1.7 GB, or --ptbxl-root <existing copy>)
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback, platform

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
LOG = os.path.join(HERE, "run_log.txt"); STATUS = os.path.join(HERE, "status.json")


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + str(msg)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def status(update):
    st = json.load(open(STATUS)) if os.path.exists(STATUS) else dict(done=[], failed=[], current=None, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    st.update(update); st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(st, open(STATUS, "w"), indent=1); return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mitdb", required=True, help="folder containing 100.dat ... 234.dat")
    ap.add_argument("--out", default=os.path.join(HERE, "results_revision"))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--skip-ptbxl", action="store_true")
    ap.add_argument("--ptbxl-data", default=os.path.join(HERE, "data"))
    ap.add_argument("--ptbxl-root", default=None,
                    help="folder that already contains ptbxl_database.csv, scp_statements.csv and records100/ "
                         "(e.g. an existing PhysioNet mirror); skips the download")
    ap.add_argument("--only", nargs="*", default=None, help="run only these config names")
    a = ap.parse_args()
    try:
        import torch; import numpy, scipy, sklearn  # noqa
    except ImportError as e:
        print("Missing package:", e, "\nActivate your torch environment, e.g.  source ~/torch-env/bin/activate\n"
              "and install what is missing:  pip install torch numpy scipy scikit-learn"); sys.exit(1)
    import torch
    torch.set_num_threads(os.cpu_count())
    log("=== run_all start | python %s | torch %s | %s | %d threads" % (platform.python_version(), torch.__version__, platform.platform(), os.cpu_count()))
    import rev_data, rev_train, rev_recon

    caches = {}
    cdir = os.path.join(HERE, "caches"); os.makedirs(cdir, exist_ok=True)
    for key, fs, leads in [("fs125_both", 125, "both"), ("fs125_mlii", 125, "MLII"), ("fs250_both", 250, "both"), ("fs360_both", 360, "both")]:
        p = os.path.join(cdir, "mitbih_%s.npz" % key)
        if not os.path.exists(p):
            log("building cache %s" % key); rev_data.build_cache(a.mitdb, p, fs, leads, verbose=True)
        caches[key] = p

    S = a.seeds
    plan = [
        ("main",            "fs125_both", dict(), S),
        ("recon_convae",    "fs125_both", dict(_recon="convae"), 3),
        ("recon_jscc",      "fs125_both", dict(_recon="jscc"), 3),
        ("recon_sameenc_pe", "fs125_both", dict(_recon="sameenc_pe"), 3),
        ("main_pe",         "fs125_both", dict(pos_enc=True), 10),
        ("latent4",         "fs125_both", dict(latent=4), 10),
        ("latent8",         "fs125_both", dict(latent=8), 5),
        ("abl_cnn_only",    "fs125_both", dict(use_transformer=False), S),
        ("abl_tf1",         "fs125_both", dict(n_layers=1), S),
        ("abl_no_reg",      "fs125_both", dict(lam=0.0), S),
        ("abl_no_rate",     "fs125_both", dict(gamma=0.0), S),
        ("abl_no_ee",       "fs125_both", dict(early_exit=False), S),
        ("abl_task_only",   "fs125_both", dict(lam=0.0, gamma=0.0, early_exit=False), S),
        ("qat4",            "fs125_both", dict(bits=4), S),
        ("qat3",            "fs125_both", dict(bits=3), S),
        ("chan_erase20",    "fs125_both", dict(train_erasure=0.2), S),
        ("lead_mlii",       "fs125_mlii", dict(), S),
        ("fs250",           "fs250_both", dict(), S),
        ("fs360",           "fs360_both", dict(), S),
        ("random_split",    "fs125_both", dict(split="random"), 3),
    ]
    if a.only:
        plan = [p for p in plan if p[0] in a.only]
    for name, ckey, ov, n_seeds in plan:
        for seed in range(n_seeds):
            status(dict(current="%s/seed%d" % (name, seed)))
            try:
                if "_recon" in ov:
                    rev_recon.train_codec(caches[ckey], ov["_recon"], seed, a.out, log=log)
                else:
                    rev_train.train_one(caches[ckey], name, seed, a.out, ov, log=log)
                st = status({}); st["done"] = sorted(set(st["done"] + ["%s/seed%d" % (name, seed)])); status(st)
            except Exception:
                log("FAILED %s seed %d\n%s" % (name, seed, traceback.format_exc()))
                st = status({}); st["failed"] = sorted(set(st["failed"] + ["%s/seed%d" % (name, seed)])); status(st)

    if not a.skip_ptbxl and (a.only is None or "ptbxl_full" in a.only):
        try:
            import rev_ptbxl
            p = os.path.join(cdir, "ptbxl_full.npz")
            if not os.path.exists(p):
                if a.ptbxl_root:
                    root = a.ptbxl_root
                    for need in ("ptbxl_database.csv", "scp_statements.csv", "records100"):
                        if not os.path.exists(os.path.join(root, need)):
                            raise FileNotFoundError("--ptbxl-root %s does not contain %s" % (root, need))
                    log("using existing PTB-XL copy at %s" % root)
                else:
                    root = rev_ptbxl.ensure_data(a.ptbxl_data, log=log)
                rev_ptbxl.build_cache(root, p, log=log)
            for seed in range(5):
                status(dict(current="ptbxl_full/seed%d" % seed))
                try:
                    rev_train.train_one(p, "ptbxl_full", seed, a.out, dict(split="fold", n_classes=2), log=log)
                    st = status({}); st["done"] = sorted(set(st["done"] + ["ptbxl_full/seed%d" % seed])); status(st)
                except Exception:
                    log("FAILED ptbxl_full seed %d\n%s" % (seed, traceback.format_exc()))
        except Exception:
            log("PTB-XL stage failed:\n" + traceback.format_exc())

    # summary
    summ = {}
    for name in sorted(os.listdir(a.out)):
        d = os.path.join(a.out, name)
        if not os.path.isdir(d):
            continue
        rows = [json.load(open(os.path.join(d, f))) for f in sorted(os.listdir(d)) if f.endswith(".json") and not f.startswith("._")]
        if not rows:
            continue
        if "metrics" in rows[0]:
            import numpy as np
            keys = ["auroc", "auprc", "macro_f1", "sens_at_spec", "f1_at_op"]
            summ[name] = {k: [float(np.mean([r["metrics"][k] for r in rows if k in r["metrics"]])),
                              float(np.std([r["metrics"][k] for r in rows if k in r["metrics"]]))]
                          for k in keys if k in rows[0]["metrics"]}
            summ[name]["n_seeds"] = len(rows); summ[name]["macs_enc"] = rows[0]["macs_enc"]; summ[name]["params_enc"] = rows[0]["params_enc"]
        else:
            summ[name] = dict(n_seeds=len(rows), test_mse=[r["test_mse"] for r in rows], test_prd_pct=[r["test_prd_pct"] for r in rows])
    json.dump(summ, open(os.path.join(a.out, "SUMMARY.json"), "w"), indent=1)
    status(dict(current="finished"))
    log("=== ALL DONE. Summary:\n" + json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
