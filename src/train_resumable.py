"""Resumable, time-boxed FULL-schedule trainer for the semantic encoder.

Trains enc+dec for `total_epochs` over the FULL MIT-BIH training split, check-
pointing after every epoch so the 6-epoch schedule completes across several
short (CPU-time-limited) invocations. On reaching total_epochs it evaluates on
val+test, writes per-seed metrics to results/full_<split>_perseed.json, and
saves the trained model (results/model_<split>_<seed>.pt) + the test split
(results/eval_<split>_<seed>.npz) for the rate/packet-loss sweeps.

Usage: python train_resumable.py <patient|random> <seed> <total_epochs> [budget_s] [batch]
"""
import sys, os, json, time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import metrics as M
import run_real_experiment as R
from run_mitbih_experiment import patient_split
from config import EncoderConfig
from semantic_encoder import SemanticEncoder, SemanticDecoder, sempq_loss

split = sys.argv[1]
seed = int(sys.argv[2])
TOTAL_EPOCHS = int(sys.argv[3])
BUDGET = float(sys.argv[4]) if len(sys.argv) > 4 else 34.0
BATCH = int(sys.argv[5]) if len(sys.argv) > 5 else 64
LR = 3e-3
torch.set_num_threads(os.cpu_count())
t_start = time.time()

d = np.load("results/mitbih_cache.npz", allow_pickle=True)
X, y, g = d["X"], d["y"], d["groups"]
if split == "patient":
    tr, va, te = patient_split(y, g, seed=seed)
else:
    tr, va, te = R.stratified_split(y, seed=seed)

cfg = EncoderConfig(in_channels=X.shape[1])
enc, dec = SemanticEncoder(cfg), SemanticDecoder(cfg)
opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)

ck = "results/ck_%s_%d.pt" % (split, seed)
start_epoch = 0
if os.path.exists(ck):
    st = torch.load(ck)
    enc.load_state_dict(st["enc"]); dec.load_state_dict(st["dec"])
    opt.load_state_dict(st["opt"]); start_epoch = st["epoch"]
    print("resumed from epoch", start_epoch)

torch.manual_seed(1000 + seed)
loader = DataLoader(TensorDataset(torch.tensor(X[tr]), torch.tensor(y[tr])),
                    batch_size=BATCH, shuffle=True)

epoch = start_epoch
while epoch < TOTAL_EPOCHS:
    # stop if not enough time for another epoch (est from elapsed/!epochs done this call)
    if time.time() - t_start > BUDGET:
        break
    enc.train(); dec.train()
    for xb, yb in loader:
        z, ee, rate = enc(xb); logits = dec(z)
        loss, _ = sempq_loss(logits, ee, z, rate, yb, cfg)
        opt.zero_grad(); loss.backward(); opt.step()
    epoch += 1
    torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(),
                "opt": opt.state_dict(), "epoch": epoch}, ck)
    print("epoch %d/%d done (%.1fs elapsed)" % (epoch, TOTAL_EPOCHS, time.time() - t_start))
    # if one more epoch would likely exceed budget, stop now (clean checkpoint saved)
    per = (time.time() - t_start) / (epoch - start_epoch)
    if time.time() - t_start + per > BUDGET:
        break

if epoch < TOTAL_EPOCHS:
    print("PROGRESS %d/%d (re-run to continue)" % (epoch, TOTAL_EPOCHS))
    sys.exit(0)

# ---- finalize: evaluate + save ----
enc.eval(); dec.eval()
def infer(idx):
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), 256):
            xb = torch.tensor(X[idx[i:i+256]])
            out.append(dec(enc(xb)[0]).softmax(-1).numpy())
    return np.concatenate(out, 0)
pv, pt = infer(va), infer(te)
yv, sv = M.to_deterioration_binary(y[va], pv)
yt, st = M.to_deterioration_binary(y[te], pt)
op = M.calibrated_operating_point(yv, sv, yt, st, 0.90)
res = {"auroc": M.auroc_multiclass(y[te], pt), "auprc": M.auprc_multiclass(y[te], pt),
       "macro_f1": M.macro_f1(y[te], pt.argmax(1)), "sens_at_spec": op["sensitivity"],
       "f1_at_op": op["f1"], "alarm_rate": op["alarm_rate"]}
pth = "results/full_%s_perseed.json" % split
acc = json.load(open(pth)) if os.path.exists(pth) else {}
acc[str(seed)] = res
acc["_meta"] = {"epochs": TOTAL_EPOCHS, "batch": BATCH, "lr": LR, "split": split,
                "full_train": True, "n_train": int(len(tr))}
json.dump(acc, open(pth, "w"), indent=2)
torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(), "cfg_in": int(X.shape[1])},
           "results/model_%s_%d.pt" % (split, seed))
np.savez_compressed("results/eval_%s_%d.npz" % (split, seed),
                    X_te=X[te], y_te=y[te], pt=pt)
if os.path.exists(ck):
    try:
        os.remove(ck)
    except OSError:
        pass
print("FINAL seed %d:" % seed, {k: round(v, 3) for k, v in res.items()})
