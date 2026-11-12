"""Step 1: build every window cache used by the experiments.

Expects the PhysioNet folders under  $ECG_WORK/data  (default: experiments/work/data):
    mitdb/1.0.0   svdb/1.0.0   incartdb/1.0.0
Run from the experiments/ folder:   python build_caches.py
"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
WORK = os.environ.get("ECG_WORK", os.path.join(HERE, "work")); D = os.path.join(WORK, "data")
import rev_data, new_xdb_cache
os.makedirs("caches", exist_ok=True)
for key, fs, leads in (("fs125_both", 125, "both"), ("fs125_mlii", 125, "MLII")):
    p = "caches/mitbih_%s.npz" % key
    if not os.path.exists(p): rev_data.build_cache(os.path.join(D, "mitdb/1.0.0"), p, fs, leads)
if not os.path.exists("caches/svdb.npz"): new_xdb_cache.build(os.path.join(D, "svdb/1.0.0"), "caches/svdb.npz", [0, 1])
if not os.path.exists("caches/incart.npz"): new_xdb_cache.build(os.path.join(D, "incartdb/1.0.0"), "caches/incart.npz", ["II", "V1"])
m = np.load("caches/mitbih_fs125_both.npz", allow_pickle=True); s = np.load("caches/svdb.npz", allow_pickle=True); i = np.load("caches/incart.npz", allow_pickle=True)
X = np.concatenate([m["X"], s["X"], i["X"]]); y = np.concatenate([m["y"], s["y"], i["y"]])
g = np.concatenate([m["groups"].astype(str), np.char.add("svdb_", s["groups"].astype(str)), np.char.add("incart_", i["groups"].astype(str))])
np.savez("caches/pool_all.npz", X=X, y=y, groups=g); np.savez("caches/pool_all_1lead.npz", X=X[:, :1], y=y, groups=g)
print("pooled cache:", X.shape, "windows from", len(set(g)), "records")
