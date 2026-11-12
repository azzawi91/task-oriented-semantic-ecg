"""Multi-database training (leave-one-database-out and all-pooled), light (CNN-only) and full encoders."""
import os as _os
WORK = _os.environ.get("ECG_WORK", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "work"))  # data, runs and outputs live here
import os, sys, time, traceback
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
OUT = (WORK + "/res"); CD = os.path.join(HERE, "caches"); LOCK = (WORK + "/locks4"); os.makedirs(LOCK, exist_ok=True)
import torch; torch.set_num_threads(1)
import rev_train
def log(m):
    with open((WORK + "/run_log.txt"), "a") as f: f.write(time.strftime("%H:%M:%S ") + "[r%s] " % sys.argv[1] + str(m) + "\n")
POOLS = {"all": ["svdb", "incart"], "noSV": ["incart"], "noIN": ["svdb"]}
plan = []
for pk, dbs in POOLS.items():
    plan += [("pool_%s_cnn_D4" % pk, dict(split="pool", pool_dbs=dbs, use_transformer=False, latent=4), 10),
             ("pool_%s_cnn_D32" % pk, dict(split="pool", pool_dbs=dbs, use_transformer=False, latent=32), 10)]
plan += [("ds1_cnn_D4", dict(use_transformer=False, latent=4), 10)]
plan += [("pool_all_full_D4", dict(split="pool", pool_dbs=["svdb", "incart"], latent=4), 3)]
for name, ov, n in plan:
    for seed in range(n):
        tag = "%s_seed%d" % (name, seed)
        if os.path.exists(os.path.join(OUT, name, "seed%d.npz" % seed)): continue
        try: os.mkdir(os.path.join(LOCK, tag))
        except FileExistsError: continue
        try: rev_train.train_one(os.path.join(CD, "pool_all.npz"), name, seed, OUT, ov, log=log)
        except Exception: log("FAILED %s\n%s" % (tag, traceback.format_exc()))
log("PHASE4 WORKER FINISHED")
