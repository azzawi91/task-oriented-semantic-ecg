"""Further multi-database jobs with the light encoder: width sweep, loss ablations, single lead,
matched classifiers on wavelet-decoded windows, PTB-XL."""
import os as _os
WORK = _os.environ.get("ECG_WORK", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "work"))  # data, runs and outputs live here
import os, sys, time, traceback
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
OUT = (WORK + "/res"); CD = os.path.join(HERE, "caches"); LOCK = (WORK + "/locks5"); os.makedirs(LOCK, exist_ok=True)
import torch; torch.set_num_threads(1)
import rev_train
def log(m):
    with open((WORK + "/run_log.txt"), "a") as f: f.write(time.strftime("%H:%M:%S ") + "[s%s] " % sys.argv[1] + str(m) + "\n")
P = dict(split="pool", pool_dbs=["svdb", "incart"], use_transformer=False)
plan = [("pool_all_cnn_D%d" % D, "pool_all", dict(P, latent=D), 10) for D in (1, 2, 8, 16)]
plan += [("pool_dwt%d_cnn" % B, "pool_dwt_%d" % B, dict(P, latent=32), 5) for B in (4, 8, 16, 32, 64, 128, 256)]
plan += [("pool_all_cnn_D4_1lead", "pool_all_1lead", dict(P, latent=4), 10)]
plan += [("pool_all_cnn_D4_noreg", "pool_all", dict(P, latent=4, lam=0.0), 5), ("pool_all_cnn_D4_norate", "pool_all", dict(P, latent=4, gamma=0.0), 5),
         ("pool_all_cnn_D4_noee", "pool_all", dict(P, latent=4, early_exit=False), 5), ("pool_all_cnn_D4_taskonly", "pool_all", dict(P, latent=4, lam=0.0, gamma=0.0, early_exit=False), 5)]
plan += [("ptbxl_cnn_D4", "ptbxl_full", dict(split="fold", n_classes=2, use_transformer=False, latent=4), 5), ("ptbxl_cnn_D32", "ptbxl_full", dict(split="fold", n_classes=2, use_transformer=False, latent=32), 5)]
for name, ck, ov, n in plan:
    for seed in range(n):
        tag = "%s_seed%d" % (name, seed)
        if os.path.exists(os.path.join(OUT, name, "seed%d.npz" % seed)): continue
        cache = os.path.join(CD, ck + ".npz")
        if not os.path.exists(cache): continue
        try: os.mkdir(os.path.join(LOCK, tag))
        except FileExistsError: continue
        try: rev_train.train_one(cache, name, seed, OUT, ov, log=log)
        except Exception: log("FAILED %s\n%s" % (tag, traceback.format_exc()))
log("PHASE5 WORKER FINISHED")
