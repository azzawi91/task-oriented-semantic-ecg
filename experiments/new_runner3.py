import os as _os
WORK = _os.environ.get("ECG_WORK", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "work"))  # data, runs and outputs live here
import os, sys, time, traceback
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
OUT = (WORK + "/res"); CD = os.path.join(HERE, "caches"); LOCK = (WORK + "/locks3"); os.makedirs(LOCK, exist_ok=True)
import torch; torch.set_num_threads(1)
import rev_train
def log(m):
    with open((WORK + "/run_log.txt"), "a") as f: f.write(time.strftime("%H:%M:%S ") + "[q%s] " % sys.argv[1] + str(m) + "\n")
for name, n in (("cls_dwt_4", 10), ("cls_dwt_8", 10), ("cls_dwt_16", 10)):
    for seed in range(n):
        tag = "%s_seed%d" % (name, seed)
        if os.path.exists(os.path.join(OUT, name, "seed%d.npz" % seed)): continue
        try: os.mkdir(os.path.join(LOCK, tag))
        except FileExistsError: continue
        try: rev_train.train_one(os.path.join(CD, name + ".npz"), name, seed, OUT, {}, log=log)
        except Exception: log("FAILED %s\n%s" % (tag, traceback.format_exc()))
log("PHASE3 WORKER FINISHED")
