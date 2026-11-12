"""Further multi-database jobs with the light encoder: width sweep, loss ablations, single lead,
matched classifiers on wavelet-decoded windows, PTB-XL."""
import os as _os
WORK = _os.environ.get("ECG_WORK", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "work"))  # data, runs and outputs live here
import os, sys, time, traceback
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
OUT = (WORK + "/res"); CD = os.path.join(HERE, "caches"); LOCK = (WORK + "/locks6"); os.makedirs(LOCK, exist_ok=True)
import torch; torch.set_num_threads(1)
import rev_train
def log(m):
    with open((WORK + "/run_log.txt"), "a") as f: f.write(time.strftime("%H:%M:%S ") + "[t%s] " % sys.argv[1] + str(m) + "\n")
P = dict(split="pool", pool_dbs=["svdb", "incart"], use_transformer=False)
plan = [("pool_dwt%d_cnn" % B, "pool_dwt_%d" % B, dict(P, latent=32), 10) for B in (4, 16, 64, 256, 8)]
plan += [("pool_all_cnn_D%d" % D, "pool_all", dict(P, latent=D), 10) for D in (2, 1)]
plan += [("pool_all_cnn_D4_1lead", "pool_all_1lead", dict(P, latent=4), 10)]
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
log("PHASE6 WORKER FINISHED")
