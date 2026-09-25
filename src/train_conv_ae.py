"""Resumable learned reconstruction codecs (Gap-4 fair baselines), de Chazal DS1.

A dedicated convolutional autoencoder with a NON-pooled 32-dimensional bottleneck
(same transmitted payload as the task latent) is the architecture suited to
reconstruction; only the training objective differs from the task model.
  mode=recon : MSE reconstruction.
  mode=jscc  : deep joint source-channel coding -- AWGN injected on the 32-dim
               code during training (after Bourtsoulatze & Gunduz).

Usage: python train_conv_ae.py <recon|jscc> <total_epochs> [budget_s] [batch]
Saves results/model_<mode>_dechazal.pt (encoder weights + arch tag).
"""
import sys, os, time
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import dechazal as DC

MODE = sys.argv[1] if len(sys.argv) > 1 else "recon"
TOTAL = int(sys.argv[2]) if len(sys.argv) > 2 else 6
BUDGET = float(sys.argv[3]) if len(sys.argv) > 3 else 34.0
BATCH = int(sys.argv[4]) if len(sys.argv) > 4 else 64
JSCC_SNR_DB, D = 10.0, 32
torch.set_num_threads(os.cpu_count()); t0 = time.time()

d = np.load("results/mitbih_cache.npz", allow_pickle=True)
X, y, g = d["X"], d["y"], d["groups"]
tr, va, te = DC.split(y, g, seed=0)
C, T = X.shape[1], X.shape[2]

class ConvAEEncoder(nn.Module):
    def __init__(self, C, D):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(C, 16, 7, stride=4, padding=3), nn.GELU(),   # 1250->313
            nn.Conv1d(16, 32, 5, stride=4, padding=2), nn.GELU(),  # 313->79
            nn.Conv1d(32, 48, 5, stride=4, padding=2), nn.GELU())  # 79->20
        self.fc = nn.Linear(48 * 20, D)
    def forward(self, x):
        h = self.conv(x); return self.fc(h.flatten(1))

class ConvAEDecoder(nn.Module):
    def __init__(self, C, D, T):
        super().__init__(); self.T = T
        self.fc = nn.Linear(D, 48 * 20)
        self.net = nn.Sequential(
            nn.ConvTranspose1d(48, 32, 5, stride=4, padding=2, output_padding=3), nn.GELU(),
            nn.ConvTranspose1d(32, 16, 5, stride=4, padding=2, output_padding=3), nn.GELU(),
            nn.ConvTranspose1d(16, C, 7, stride=4, padding=3, output_padding=3))
    def forward(self, z):
        h = self.fc(z).view(-1, 48, 20)
        return self.net(h)[..., :self.T]

enc = ConvAEEncoder(C, D); dec = ConvAEDecoder(C, D, T)
opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=2e-3)
ck = "results/ckcae2_%s.pt" % MODE; start = 0
if os.path.exists(ck):
    st = torch.load(ck); enc.load_state_dict(st["enc"]); dec.load_state_dict(st["dec"])
    opt.load_state_dict(st["opt"]); start = st["epoch"]; print("resumed", start)
torch.manual_seed(11)
loader = DataLoader(TensorDataset(torch.tensor(X[tr])), batch_size=BATCH, shuffle=True)
mse = nn.MSELoss()
def add_awgn(z, snr_db):
    p = z.pow(2).mean(); n = p / (10 ** (snr_db / 10))
    return z + torch.randn_like(z) * n.sqrt()
ep = start
while ep < TOTAL:
    if time.time() - t0 > BUDGET: break
    enc.train(); dec.train()
    for (xb,) in loader:
        z = enc(xb)
        if MODE == "jscc": z = add_awgn(z, JSCC_SNR_DB)
        loss = mse(dec(z), xb)
        opt.zero_grad(); loss.backward(); opt.step()
    ep += 1
    torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(), "opt": opt.state_dict(), "epoch": ep}, ck)
    print("epoch %d/%d (%.1fs) mse=%.4f" % (ep, TOTAL, time.time() - t0, loss.item()))
    if time.time() - t0 + (time.time() - t0) / (ep - start) > BUDGET: break
if ep < TOTAL:
    print("PROGRESS %d/%d" % (ep, TOTAL)); sys.exit(0)
torch.save({"enc": enc.state_dict(), "cfg_in": C, "mode": MODE, "arch": "convae"}, "results/model_%s_dechazal.pt" % MODE)
try: os.remove(ck)
except OSError: pass
print("DONE %s mse=%.4f" % (MODE, loss.item()))
