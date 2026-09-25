"""Resumable full-schedule trainer on the de Chazal DS1/DS2 partition.

Usage: python train_dechazal.py <seed> <total_epochs> [budget_s] [batch]
Saves: results/model_dechazal_<seed>.pt, results/eval_dechazal_<seed>.npz,
and appends metrics to results/full_dechazal_perseed.json.
"""
import sys, os, json, time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import metrics as M
import dechazal as DC
from config import EncoderConfig
from semantic_encoder import SemanticEncoder, SemanticDecoder, sempq_loss

seed = int(sys.argv[1]); TOTAL = int(sys.argv[2])
BUDGET = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0
BATCH = int(sys.argv[4]) if len(sys.argv) > 4 else 64
LR = 3e-3
torch.set_num_threads(os.cpu_count()); t0 = time.time()

d = np.load("results/mitbih_cache.npz", allow_pickle=True)
X, y, g = d["X"], d["y"], d["groups"]
tr, va, te = DC.split(y, g, seed=seed)
cfg = EncoderConfig(in_channels=X.shape[1])
enc, dec = SemanticEncoder(cfg), SemanticDecoder(cfg)
opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)
ck = "results/ckdc_%d.pt" % seed
start = 0
if os.path.exists(ck):
    st = torch.load(ck); enc.load_state_dict(st["enc"]); dec.load_state_dict(st["dec"])
    opt.load_state_dict(st["opt"]); start = st["epoch"]; print("resumed epoch", start)
torch.manual_seed(2000 + seed)
loader = DataLoader(TensorDataset(torch.tensor(X[tr]), torch.tensor(y[tr])), batch_size=BATCH, shuffle=True)
ep = start
while ep < TOTAL:
    if time.time() - t0 > BUDGET: break
    enc.train(); dec.train()
    for xb, yb in loader:
        z, ee, rate = enc(xb); lo, _ = sempq_loss(dec(z), ee, z, rate, yb, cfg)
        opt.zero_grad(); lo.backward(); opt.step()
    ep += 1
    torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(), "opt": opt.state_dict(), "epoch": ep}, ck)
    print("epoch %d/%d (%.1fs)" % (ep, TOTAL, time.time() - t0))
    if time.time() - t0 + (time.time() - t0) / (ep - start) > BUDGET: break
if ep < TOTAL:
    print("PROGRESS %d/%d" % (ep, TOTAL)); sys.exit(0)
enc.eval(); dec.eval()
def infer(idx):
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), 256):
            out.append(dec(enc(torch.tensor(X[idx[i:i+256]]))[0]).softmax(-1).numpy())
    return np.concatenate(out, 0)
pv, pt = infer(va), infer(te)
yv, sv = M.to_deterioration_binary(y[va], pv); yt, st = M.to_deterioration_binary(y[te], pt)
op = M.calibrated_operating_point(yv, sv, yt, st, 0.90)
res = {"auroc": M.auroc_multiclass(y[te], pt), "auprc": M.auprc_multiclass(y[te], pt),
       "macro_f1": M.macro_f1(y[te], pt.argmax(1)), "sens_at_spec": op["sensitivity"],
       "f1_at_op": op["f1"], "alarm_rate": op["alarm_rate"]}
p = "results/full_dechazal_perseed.json"
acc = json.load(open(p)) if os.path.exists(p) else {}
acc[str(seed)] = res
acc["_meta"] = {"epochs": TOTAL, "batch": BATCH, "lr": LR, "protocol": "de Chazal DS1/DS2",
                "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te))}
json.dump(acc, open(p, "w"), indent=2)
torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(), "cfg_in": int(X.shape[1])},
           "results/model_dechazal_%d.pt" % seed)
np.savez_compressed("results/eval_dechazal_%d.npz" % seed,
                    tr=tr, va=va, te=te, y_te=y[te], pt=pt)
try: os.remove(ck)
except OSError: pass
print("FINAL seed %d:" % seed, {k: round(v, 3) for k, v in res.items()})
