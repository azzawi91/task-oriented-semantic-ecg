"""Nov 2026, second batch: matched-size training sets (learning curve) and patient-level 5-fold cross-validation."""
import os as _os
WORK = _os.environ.get("ECG_WORK", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "work"))  # data, runs and outputs live here
import os, sys, time, traceback
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
OUT = (WORK + "/res"); CD = os.path.join(HERE, "caches"); LOCK = (WORK + "/locks8"); os.makedirs(LOCK, exist_ok=True)
import torch; torch.set_num_threads(1)
import rev_train
def log(m):
    with open((WORK + "/run_log8.txt"), "a") as f: f.write(time.strftime("%H:%M:%S ") + "[t%s] " % sys.argv[1] + str(m) + "\n")
B = dict(use_transformer=False, latent=4, pool_dbs=["svdb", "incart"]); MW = 6 * 30769
plan = []
# learning curve on DS2, equal number of training windows processed in every condition
plan += [("lc_ds1_18_matched", dict(B, split="dechazal", match_windows=MW), s) for s in range(10)]
for nm, sub in (("lc_mix18", dict(mit=6, svdb=6, incart=6)), ("lc_mix36", dict(mit=12, svdb=12, incart=12)), ("lc_mix72", dict(mit=18, svdb=27, incart=27))):
    plan += [(nm, dict(B, split="pool_sub", sub=sub, match_windows=MW), s) for s in range(10)]
# cross-validation: code size and training-population comparisons
for k in range(5):
    for s in range(3):
        for D in (4, 1, 32): plan.append(("cv_all_D%d_f%d" % (D, k), dict(B, latent=D, split="cv", fold=k), s))
        plan.append(("cv_mit_D4_f%d" % k, dict(B, split="cv", fold=k, cv_dbs=["mit"], match_windows=MW), s))
if sys.argv[1] == "2": plan = plan[::-1]
for name, ov, seed in plan:
    tag = "%s_seed%d" % (name, seed)
    if os.path.exists(os.path.join(OUT, name, "seed%d.npz" % seed)): continue
    try: os.mkdir(os.path.join(LOCK, tag))
    except FileExistsError: continue
    try: rev_train.train_one(os.path.join(CD, "pool_all.npz"), name, seed, OUT, ov, log=log)
    except Exception: log("FAILED %s\n%s" % (tag, traceback.format_exc()))
log("PHASE8 WORKER FINISHED")
