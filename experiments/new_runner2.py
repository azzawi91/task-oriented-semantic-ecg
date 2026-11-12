"""Phase-2 jobs: matched classifiers on wavelet-decoded windows, reconstruction codecs (exporting decoded
windows), PTB-XL, and the 250/360 Hz inputs."""
import os as _os
WORK = _os.environ.get("ECG_WORK", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "work"))  # data, runs and outputs live here
import os, sys, time, traceback
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
OUT = (WORK + "/res"); CD = os.path.join(HERE, "caches"); LOCK = (WORK + "/locks2"); os.makedirs(LOCK, exist_ok=True)
import torch; torch.set_num_threads(1)
import rev_train, rev_recon
def log(m):
    with open((WORK + "/run_log.txt"), "a") as f: f.write(time.strftime("%H:%M:%S ") + "[p%s] " % sys.argv[1] + str(m) + "\n")
plan = [("cls_dwt_%d" % B, os.path.join(CD, "cls_dwt_%d.npz" % B), {}, 3) for B in (16, 64, 256, 32, 128)]
plan += [("recon_convae", "fs125_both", dict(_recon="convae"), 3), ("recon_sameenc_pe", "fs125_both", dict(_recon="sameenc_pe"), 3), ("recon_jscc", "fs125_both", dict(_recon="jscc"), 3)]
plan += [("ptbxl_full", os.path.join(CD, "ptbxl_full.npz"), dict(split="fold", n_classes=2), 3)]
plan += [("fs250", "fs250_both", {}, 3), ("fs360", "fs360_both", {}, 3)]
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
log("PHASE2 WORKER FINISHED")
