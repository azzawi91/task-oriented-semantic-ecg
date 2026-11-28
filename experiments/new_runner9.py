"""Nov 2026, third batch: repeated patient-level 5-fold cross-validation."""
import os as _os
WORK = _os.environ.get("ECG_WORK", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "work"))  # data, runs and outputs live here
import os, sys, time, traceback
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
OUT = (WORK + "/res"); CD = os.path.join(HERE, "caches"); LOCK = (WORK + "/locks9"); os.makedirs(LOCK, exist_ok=True)
import torch; torch.set_num_threads(1)
import rev_train
def log(m):
    with open((WORK + "/run_log9.txt"), "a") as f: f.write(time.strftime("%H:%M:%S ") + "[t%s] " % sys.argv[1] + str(m) + "\n")
B = dict(use_transformer=False, latent=4, pool_dbs=["svdb", "incart"]); MW = 6 * 30769
plan = []
# repeated patient-level five-fold cross-validation: repetition r uses its own patient partition and seed r (repetition 0 = the first batch, seed 0)
for D in (4, 32, 1, "mit"):
    for rep in range(1, 10):
        for k in range(5):
            if D == "mit": plan.append(("cv_mit_D4_r%d_f%d" % (rep, k), dict(B, split="cv", fold=k, cv_rep=rep, cv_dbs=["mit"], match_windows=MW), rep))
            else: plan.append(("cv_all_D%d_r%d_f%d" % (D, rep, k), dict(B, latent=D, split="cv", fold=k, cv_rep=rep), rep))
if sys.argv[1] == "2": plan = plan[1::2]
else: plan = plan[0::2]
for name, ov, seed in plan:
    tag = "%s_seed%d" % (name, seed)
    if os.path.exists(os.path.join(OUT, name, "seed%d.npz" % seed)): continue
    try: os.mkdir(os.path.join(LOCK, tag))
    except FileExistsError: continue
    try: rev_train.train_one(os.path.join(CD, "pool_all.npz"), name, seed, OUT, ov, log=log)
    except Exception: log("FAILED %s\n%s" % (tag, traceback.format_exc()))
log("PHASE9 WORKER FINISHED")
