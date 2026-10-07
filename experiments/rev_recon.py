"""Reconstruction-oriented learned codecs, trained properly (Reviewer 1 #4).

Three codecs, all with a 32-dimensional bottleneck (same transmitted dimension
as the task-oriented latent), trained to minimise waveform MSE:

  convae   : dedicated convolutional autoencoder (the architecture used in the
             original submission), MSE objective.
  jscc     : same autoencoder trained with an AWGN channel layer on the code
             (deep joint source-channel coding, Bourtsoulatze et al. 2019),
             SNR = 10 dB during training.
  sameenc  : the *task encoder architecture itself* (depthwise-separable CNN +
             2-layer Transformer) followed by a transposed-convolution decoder,
             MSE objective. NOTE: the published encoder has no positional
             encoding and mean-pools its tokens, so its latent is invariant to
             token order and cannot reconstruct a waveform (MSE stays at 1.0);
             this mode is kept for the record.
  sameenc_pe : same as above with fixed sinusoidal positional encodings added to
             the tokens, so that reconstruction is possible. The matched task
             model is the run_all.py configuration `main_pe`. This pair isolates
             the training objective: same encoder, different target.

Training: Adam, lr 2e-3 with cosine decay, batch 64, `epochs` epochs (60 for
convae/jscc, 30 for sameenc), model selected on validation MSE. Exports the
continuous latents of train/val/test windows plus test-set MSE and PRD so that
the matched-payload comparison can be computed post hoc with the shared
logistic-regression readout.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rev_splits as S
from rev_model import Encoder, count_params


class ConvAEEncoder(nn.Module):
    def __init__(self, C, T, D=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(C, 16, 7, stride=4, padding=3), nn.GELU(),
            nn.Conv1d(16, 32, 5, stride=4, padding=2), nn.GELU(),
            nn.Conv1d(32, 48, 5, stride=4, padding=2), nn.GELU())
        with torch.no_grad():
            self.L = self.conv(torch.zeros(1, C, T)).shape[-1]
        self.fc = nn.Linear(48 * self.L, D)

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


class ConvAEDecoder(nn.Module):
    def __init__(self, C, T, L, D=32):
        super().__init__(); self.T, self.L = T, L
        self.fc = nn.Linear(D, 48 * L)
        self.net = nn.Sequential(
            nn.ConvTranspose1d(48, 32, 5, stride=4, padding=2, output_padding=3), nn.GELU(),
            nn.ConvTranspose1d(32, 16, 5, stride=4, padding=2, output_padding=3), nn.GELU(),
            nn.ConvTranspose1d(16, C, 7, stride=4, padding=3, output_padding=3))

    def forward(self, z):
        out = self.net(self.fc(z).view(-1, 48, self.L))
        if out.shape[-1] < self.T:
            out = nn.functional.pad(out, (0, self.T - out.shape[-1]))
        return out[..., :self.T]


class SameEncDecoder(nn.Module):
    """Transposed-conv decoder from the task encoder's 32-d latent back to the waveform."""
    def __init__(self, C, T, D=32, d=48):
        super().__init__(); self.T = T
        self.L0 = math.ceil(T / 8)
        self.fc = nn.Linear(D, d * self.L0); self.d = d
        self.net = nn.Sequential(
            nn.ConvTranspose1d(d, 32, 5, stride=2, padding=2, output_padding=1), nn.GELU(),
            nn.ConvTranspose1d(32, 16, 5, stride=2, padding=2, output_padding=1), nn.GELU(),
            nn.ConvTranspose1d(16, C, 7, stride=2, padding=3, output_padding=1))

    def forward(self, z):
        out = self.net(self.fc(z).view(-1, self.d, self.L0))
        if out.shape[-1] < self.T:
            out = nn.functional.pad(out, (0, self.T - out.shape[-1]))
        return out[..., :self.T]


class SameEncWrapper(nn.Module):
    def __init__(self, C, pos_enc=False):
        super().__init__(); self.enc = Encoder(C, pos_enc=pos_enc)
    def forward(self, x):
        return self.enc.proj_latent(self.enc.features(x))


def add_awgn(z, snr_db):
    p = z.pow(2).mean(); n = p / (10 ** (snr_db / 10))
    return z + torch.randn_like(z) * n.sqrt()


def train_codec(cache, mode, seed, out_dir, epochs=None, lr=2e-3, batch=64, snr_db=10.0, log=print):
    out_path = os.path.join(out_dir, "recon_" + mode, "seed%d.npz" % seed)
    if os.path.exists(out_path):
        log("  [skip] recon_%s seed %d exists" % (mode, seed)); return json.load(open(out_path[:-4] + ".json"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    d = np.load(cache, allow_pickle=True); X, y, g = d["X"], d["y"], d["groups"]
    tr, va, te = S.dechazal_split(y, g, seed)
    C, T = X.shape[1], X.shape[2]
    epochs = epochs or (30 if mode.startswith("sameenc") else 60)
    torch.manual_seed(3000 + seed); np.random.seed(seed)
    if mode.startswith("sameenc"):
        enc = SameEncWrapper(C, pos_enc=(mode == "sameenc_pe")); dec = SameEncDecoder(C, T)
    else:
        enc = ConvAEEncoder(C, T); dec = ConvAEDecoder(C, T, enc.L)
    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(TensorDataset(torch.tensor(X[tr])), batch_size=batch, shuffle=True)
    Xva = torch.tensor(X[va]); mse = nn.MSELoss(); best = (1e9, None, None); t0 = time.time(); hist = []
    for ep in range(epochs):
        enc.train(); dec.train(); tot = 0.0; n = 0
        for (xb,) in loader:
            z = enc(xb)
            if mode == "jscc":
                z = add_awgn(z, snr_db)
            loss = mse(dec(z), xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(xb); n += len(xb)
        sched.step()
        enc.eval(); dec.eval()
        with torch.no_grad():
            vm = float(np.mean([mse(dec(enc(Xva[i:i + 256])), Xva[i:i + 256]).item() for i in range(0, len(Xva), 256)]))
        hist.append((tot / n, vm))
        if vm < best[0]:
            best = (vm, {k: v.clone() for k, v in enc.state_dict().items()}, {k: v.clone() for k, v in dec.state_dict().items()})
        log("  recon_%s seed %d epoch %d/%d train %.4f val %.4f (%.0fs)" % (mode, seed, ep + 1, epochs, tot / n, vm, time.time() - t0))
    enc.load_state_dict(best[1]); dec.load_state_dict(best[2]); enc.eval(); dec.eval()

    def lat(idx):
        o = []
        with torch.no_grad():
            for i in range(0, len(idx), 256):
                o.append(enc(torch.tensor(X[idx[i:i + 256]])).numpy())
        return np.concatenate(o, 0).astype(np.float32)

    Ztr, Zva, Zte = lat(tr), lat(va), lat(te)
    with torch.no_grad():
        Xt = torch.tensor(X[te]); num = 0.0; den = 0.0; se = 0.0; cnt = 0
        for i in range(0, len(Xt), 256):
            xb = Xt[i:i + 256]; xr = dec(enc(xb))
            se += ((xr - xb) ** 2).sum().item(); cnt += xb.numel()
            num += ((xr - xb) ** 2).sum().item(); den += (xb ** 2).sum().item()
    res = dict(test_mse=se / cnt, test_prd_pct=100.0 * math.sqrt(num / den), best_val_mse=best[0],
               epochs=epochs, params_enc=count_params(enc), params_dec=count_params(dec),
               train_time_s=time.time() - t0, hist=hist, snr_db=snr_db if mode == "jscc" else None)
    np.savez_compressed(out_path, Z_tr=Ztr, Z_va=Zva, Z_te=Zte, y_tr=y[tr], y_va=y[va], y_te=y[te],
                        idx_tr=tr, idx_va=va, idx_te=te)
    json.dump(dict(name="recon_" + mode, seed=seed, **res), open(out_path[:-4] + ".json", "w"), indent=1)
    log("  DONE recon_%s seed %d: test MSE %.4f  PRD %.1f%%  (%.0fs)" % (mode, seed, res["test_mse"], res["test_prd_pct"], res["train_time_s"]))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--mode", default="convae")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results_revision")
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args(); train_codec(a.cache, a.mode, a.seed, a.out, a.epochs)
