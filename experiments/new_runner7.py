"""Confirmatory runs, Nov 2026: Transformer seeds 3-9 (matched to the CNN seeds) and the record-mixed control."""
import os as _os
WORK = _os.environ.get("ECG_WORK", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "work"))  # data, runs and outputs live here
import os, sys, time, traceback
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
OUT = (WORK + "/res"); CD = os.path.join(HERE, "caches"); LOCK = (WORK + "/locks7"); os.makedirs(LOCK, exist_ok=True)
import torch; torch.set_num_threads(1)
import rev_train
def log(m):
    with open((WORK + "/run_log7.txt"), "a") as f: f.write(time.strftime("%H:%M:%S ") + "[t%s] " % sys.argv[1] + str(m) + "\n")
P = dict(split="pool", pool_dbs=["svdb", "incart"])
tf = [("pool_all_full_D4", "pool_all", dict(P, latent=4), s) for s in range(3, 10)]
mx = [("pool_mixed_cnn_D4", "pool_all", dict(P, split="pool_mixed", use_transformer=False, latent=4), s) for s in range(10)]
plan = tf + mx if sys.argv[1] == "1" else mx + tf
for name, ck, ov, seed in plan:
    tag = "%s_seed%d" % (name, seed)
    if os.path.exists(os.path.join(OUT, name, "seed%d.npz" % seed)): continue
    try: os.mkdir(os.path.join(LOCK, tag))
    except FileExistsError: continue
    try: rev_train.train_one(os.path.join(CD, ck + ".npz"), name, seed, OUT, ov, log=log)
    except Exception: log("FAILED %s\n%s" % (tag, traceback.format_exc()))
log("PHASE7 WORKER FINISHED")
