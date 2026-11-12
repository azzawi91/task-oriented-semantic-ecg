"""Resumable two-worker job runner for the 2026-11 revision (cloud box)."""
import os as _os
WORK = _os.environ.get("ECG_WORK", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "work"))  # data, runs and outputs live here
import os, sys, json, time, traceback
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
OUT = (WORK + "/res"); CD = os.path.join(HERE, "caches"); LOCK = (WORK + "/locks")
os.makedirs(LOCK, exist_ok=True); os.makedirs(OUT, exist_ok=True)
import torch; torch.set_num_threads(1)
import rev_train, rev_recon
def log(m):
    with open((WORK + "/run_log.txt"), "a") as f: f.write(time.strftime("%H:%M:%S ") + "[w%s] " % sys.argv[1] + str(m) + "\n")
S = 10
plan = [("main","fs125_both",{},S)]
plan += [("latent%d"%D,"fs125_both",dict(latent=D),n) for D,n in [(4,10),(8,5),(1,5),(2,5),(16,5)]]
plan += [("abl_cnn_only","fs125_both",dict(use_transformer=False),S),("abl_tf1","fs125_both",dict(n_layers=1),S),
 ("abl_task_only","fs125_both",dict(lam=0.0,gamma=0.0,early_exit=False),S),("lead_mlii","fs125_mlii",{},S),
 ("abl_no_reg","fs125_both",dict(lam=0.0),S),("abl_no_rate","fs125_both",dict(gamma=0.0),S),("abl_no_ee","fs125_both",dict(early_exit=False),S),
 ("qat4","fs125_both",dict(bits=4),S),("qat3","fs125_both",dict(bits=3),S),("chan_erase20","fs125_both",dict(train_erasure=0.2),S),
 ("main_pe","fs125_both",dict(pos_enc=True),10),
 ("recon_convae","fs125_both",dict(_recon="convae"),3),("recon_jscc","fs125_both",dict(_recon="jscc"),3),("recon_sameenc_pe","fs125_both",dict(_recon="sameenc_pe"),3),
 ("random_split","fs125_both",dict(split="random"),3)]
extra = json.load(open(os.path.join(HERE,"extra_jobs.json"))) if os.path.exists(os.path.join(HERE,"extra_jobs.json")) else []
plan += [tuple(e) for e in extra]
plan += [("fs250","fs250_both",{},S),("fs360","fs360_both",{},S)]
for name, ck, ov, n in plan:
    for seed in range(n):
        tag = "%s_seed%d" % (name, seed)
        if os.path.exists(os.path.join(OUT, name, "seed%d.npz" % seed)): continue
        try: os.mkdir(os.path.join(LOCK, tag))
        except FileExistsError: continue
        try:
            cache = ck if ck.endswith(".npz") else os.path.join(CD, "mitbih_%s.npz" % ck)
            if "_recon" in ov: rev_recon.train_codec(cache, ov["_recon"], seed, OUT, log=log)
            else: rev_train.train_one(cache, name, seed, OUT, ov, log=log)
        except Exception:
            log("FAILED %s\n%s" % (tag, traceback.format_exc()))
log("WORKER FINISHED")
