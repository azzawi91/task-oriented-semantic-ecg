"""Train one (configuration, seed) of the task-oriented codec and export
everything the post-hoc analysis needs (continuous latents, decoder weights,
probabilities, metrics). PyTorch, CPU.

Called by run_all.py; can also be used directly:
  python rev_train.py --cache caches/mitbih_fs125_both.npz --name main --seed 0
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rev_metrics as M
import rev_splits as S
from rev_model import Encoder, Decoder, task_loss, count_params, count_macs, quantize_ste

DEFAULT = dict(use_transformer=True, n_layers=2, bits=8, early_exit=True, gamma=0.05, lam=0.105,
               train_erasure=0.0, epochs=6, lr=3e-3, batch=64, split="dechazal", n_classes=3, pos_enc=False, latent=32)


def train_one(cache, name, seed, out_dir, overrides=None, log=print):
    cfg = dict(DEFAULT); cfg.update(overrides or {})
    out_path = os.path.join(out_dir, name, "seed%d.npz" % seed)
    if os.path.exists(out_path):
        log("  [skip] %s seed %d exists" % (name, seed)); return json.load(open(out_path[:-4] + ".json"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    d = np.load(cache, allow_pickle=True)
    X, y, g = d["X"], d["y"], d["groups"]
    if cfg["split"] == "dechazal":
        tr, va, te = S.dechazal_split(y, g, seed)
    elif cfg["split"] == "random":
        tr, va, te = S.record_disjoint_split(y, g, seed)
    elif cfg["split"] == "fold":            # PTB-XL official folds stored in `groups`
        tr, va, te = S.fold_split(g)
    else:
        raise ValueError(cfg["split"])
    C, T = X.shape[1], X.shape[2]
    torch.manual_seed(2000 + seed); np.random.seed(seed)
    Dl = int(cfg.get("latent", 32))
    enc = Encoder(C, use_transformer=cfg["use_transformer"], n_layers=cfg["n_layers"], bits=cfg["bits"],
                  train_erasure=cfg["train_erasure"], n_classes=cfg["n_classes"], pos_enc=cfg.get("pos_enc", False), latent=Dl)
    dec = Decoder(latent=Dl, n_classes=cfg["n_classes"])
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=cfg["lr"])
    loader = DataLoader(TensorDataset(torch.tensor(X[tr]), torch.tensor(y[tr])), batch_size=cfg["batch"], shuffle=True)
    t0 = time.time(); hist = []
    for ep in range(cfg["epochs"]):
        enc.train(); dec.train(); tot = 0.0; n = 0
        for xb, yb in loader:
            zq, ee, rate, _ = enc(xb)
            loss, parts = task_loss(dec(zq), ee, zq, rate, yb, cfg["gamma"], cfg["lam"], cfg["early_exit"])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(yb); n += len(yb)
        hist.append(tot / n)
        log("  %s seed %d epoch %d/%d loss %.4f (%.0fs)" % (name, seed, ep + 1, cfg["epochs"], tot / n, time.time() - t0))
    train_time = time.time() - t0
    enc.eval(); dec.eval()

    def latents(idx):
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), 256):
                out.append(enc(torch.tensor(X[idx[i:i + 256]]))[3].numpy())
        return np.concatenate(out, 0).astype(np.float32)

    def probs(Z):
        with torch.no_grad():
            zq = quantize_ste(torch.tensor(Z), cfg["bits"], False)
            return dec(zq).softmax(-1).numpy().astype(np.float32)

    Ztr, Zva, Zte = latents(tr), latents(va), latents(te)
    Pva, Pte = probs(Zva), probs(Zte)
    res = M.full_metrics(y[va], Pva, y[te], Pte, n_classes=cfg["n_classes"])
    macs_enc, macs_dec, L = count_macs(enc, C, T, latent=Dl, n_classes=cfg["n_classes"])
    meta = dict(name=name, seed=seed, cfg=cfg, cache=os.path.basename(cache), in_ch=C, T=T, seq_len=L,
                params_enc=count_params(enc), params_dec=count_params(dec), macs_enc=macs_enc, macs_dec=macs_dec,
                train_time_s=train_time, loss_hist=hist, n_train=int(len(tr)), n_val=int(len(va)), n_test=int(len(te)),
                metrics=res)
    np.savez_compressed(out_path, Z_tr=Ztr, Z_va=Zva, Z_te=Zte, y_tr=y[tr], y_va=y[va], y_te=y[te],
                        idx_tr=tr, idx_va=va, idx_te=te, P_va=Pva, P_te=Pte, **dec.export_numpy())
    json.dump(meta, open(out_path[:-4] + ".json", "w"), indent=1)
    torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(), "cfg": cfg, "in_ch": C}, out_path[:-4] + ".pt")
    log("  DONE %s seed %d: AUROC %.3f  F1 %.3f  (%d MACs, %d params, %.0fs)"
        % (name, seed, res["auroc"], res["macro_f1"], macs_enc, meta["params_enc"], train_time))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results_revision")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides, e.g. bits=4 use_transformer=0")
    a = ap.parse_args()
    ov = {}
    for kv in a.set:
        k, v = kv.split("="); ov[k] = json.loads(v) if v not in ("True", "False") else (v == "True")
    train_one(a.cache, a.name, a.seed, a.out, ov)
