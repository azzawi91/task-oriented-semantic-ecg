"""Rate-fidelity and packet-loss sweeps for the trained semantic encoder.

Loads the full-schedule model (results/model_patient_0.pt) and the held-out
inter-patient test set, then MEASURES (not models) diagnostic fidelity (AUROC,
macro-F1) as a function of:
  (a) latent quantization bit-width  -> payload bytes per decision (rate-fidelity)
  (b) random latent-coefficient erasure rate  -> packet-loss robustness
Also reports a bootstrap 95% CI over test windows for the operating point.
Writes results/sweeps.json.
"""
import json, numpy as np, torch
import metrics as M
from config import EncoderConfig
from semantic_encoder import SemanticEncoder, SemanticDecoder

AEAD_OVERHEAD = 28           # 16-byte GCM tag + 12-byte nonce
RAW_FS, RAW_CH, BYTES_PER_SAMPLE = 125, 2, 2   # operating sampling rate / channels / 16-bit
WINDOW_S = 10.0
RAW_BYTES = int(RAW_FS * WINDOW_S) * RAW_CH * BYTES_PER_SAMPLE

st = torch.load("results/model_patient_0.pt")
cfg = EncoderConfig(in_channels=st["cfg_in"])
enc, dec = SemanticEncoder(cfg), SemanticDecoder(cfg)
enc.load_state_dict(st["enc"]); dec.load_state_dict(st["dec"])
enc.eval(); dec.eval()
D = cfg.latent_dim

d = np.load("results/eval_patient_0.npz", allow_pickle=True)
Xte, yte = d["X_te"], d["y_te"]

def continuous_latent(X):
    """Encoder forward up to the continuous (pre-quantization) latent."""
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            x = torch.tensor(X[i:i+256])
            h = enc.stem(x); h = enc.convs(h); h = h.transpose(1, 2)
            h = enc.transformer(h); pooled = h.mean(dim=1)
            outs.append(enc.proj_latent(pooled).numpy())
    return np.concatenate(outs, 0)

Z = continuous_latent(Xte)   # [N, D] continuous

def decode(zq):
    with torch.no_grad():
        return dec(torch.tensor(zq, dtype=torch.float32)).softmax(-1).numpy()

def quantize(z, bits):
    qmax = max(1, 2 ** (bits - 1) - 1)
    return np.clip(np.round(z * qmax), -qmax - 1, qmax) / qmax

# ---- (a) rate-fidelity: quantization bit-width ----
rate_curve = []
for b in [8, 6, 4, 3, 2]:
    p = decode(quantize(Z, b))
    payload = int(np.ceil(D * b / 8)) + AEAD_OVERHEAD
    rate_curve.append({"bits": b, "payload_bytes": payload,
                       "auroc": M.auroc_multiclass(yte, p),
                       "macro_f1": M.macro_f1(yte, p.argmax(1))})

# ---- (b) packet-loss / latent-erasure robustness ----
loss_curve = []
for pr in [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]:
    aucs = []
    for s in range(10):
        rng = np.random.default_rng(s)
        zq = quantize(Z, 8).copy()
        mask = rng.random(zq.shape) < pr
        zq[mask] = 0.0
        aucs.append(M.auroc_multiclass(yte, decode(zq)))
    loss_curve.append({"loss_rate": pr, "auroc_mean": float(np.mean(aucs)),
                       "auroc_std": float(np.std(aucs))})

# ---- bootstrap CI over test windows at 8-bit ----
p8 = decode(quantize(Z, 8))
auc_pt, auc_lo, auc_hi = M.bootstrap_metric(yte, p8, M.auroc_multiclass, n_boot=1000, seed=0)
f1_pt, f1_lo, f1_hi = M.bootstrap_metric(yte, p8, lambda yt, sc: M.macro_f1(yt, sc.argmax(1)),
                                         n_boot=1000, seed=0)

out = {
    "raw_bytes_per_decision": RAW_BYTES,
    "semantic_bytes_8bit": int(np.ceil(D * 8 / 8)) + AEAD_OVERHEAD,
    "latent_dim": D,
    "byte_reduction_pct_8bit": round(100 * (1 - (D + AEAD_OVERHEAD) / RAW_BYTES), 2),
    "rate_fidelity": rate_curve,
    "packet_loss": loss_curve,
    "bootstrap_8bit": {"auroc": [auc_pt, auc_lo, auc_hi], "macro_f1": [f1_pt, f1_lo, f1_hi]},
    "n_test_windows": int(len(yte)),
}
json.dump(out, open("results/sweeps.json", "w"), indent=2)
print("RAW bytes/decision:", RAW_BYTES, "| semantic 8-bit:", out["semantic_bytes_8bit"],
      "| reduction %%:", out["byte_reduction_pct_8bit"])
print("rate-fidelity:")
for r in rate_curve:
    print("  %d-bit  %4dB  AUROC %.3f  F1 %.3f" % (r["bits"], r["payload_bytes"], r["auroc"], r["macro_f1"]))
print("packet-loss:")
for r in loss_curve:
    print("  loss %.2f  AUROC %.3f +/- %.3f" % (r["loss_rate"], r["auroc_mean"], r["auroc_std"]))
print("bootstrap AUROC %.3f [%.3f, %.3f]" % (auc_pt, auc_lo, auc_hi))
