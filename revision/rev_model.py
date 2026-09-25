"""Task-oriented semantic encoder / decoder for the revision experiments (PyTorch).

Identical to the published model (semantic_encoder.py in the public repo) when
built with the default arguments; the extra switches implement the ablations
requested by Reviewer 1 (#2) and the QAT / channel-aware variants:

  use_transformer : False -> CNN-only backbone (mean-pooled conv features)
  n_layers        : Transformer depth (2 in the paper)
  bits            : straight-through quantiser bit-width (8 in the paper; 4/3 = QAT)
  early_exit      : include the early-exit head in the task loss
  train_erasure   : probability of zeroing a latent coefficient during training
                    (channel-aware training, cf. Reviewer 1 #5)
  pos_enc         : add fixed sinusoidal positional encodings to the tokens before
                    the Transformer. The published encoder has none: with mean
                    pooling its latent is invariant to token order (a "bag of local
                    morphology" code), which is why the same encoder cannot be
                    trained for waveform reconstruction; the reconstruction control
                    therefore uses pos_enc=True, and a task model with pos_enc=True
                    is trained as well so that the comparison is symmetric.

Loss (exactly as implemented in the published code, written without the two
redundant L2 terms):  L = L_task + gamma * L_rate + lam * mean(z^2)
with L_task = 0.5*(CE(decoder) + CE(early-exit)), gamma = 0.05, lam = 0.105
(= 0.5*0.01 [KL surrogate] + 0.10 [magnitude penalty] in the original code).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv1d(nn.Module):
    def __init__(self, c_in, c_out, kernel=5, stride=2):
        super().__init__()
        self.dw = nn.Conv1d(c_in, c_in, kernel, stride, kernel // 2, groups=c_in)
        self.pw = nn.Conv1d(c_in, c_out, 1)
        self.norm = nn.GroupNorm(4, c_out)

    def forward(self, x):
        return F.gelu(self.norm(self.pw(self.dw(x))))


class EntropyBottleneck(nn.Module):
    """Factorised Gaussian prior with learned per-dimension scale -> bits/dim."""
    def __init__(self, dim):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(dim))

    def rate_bits(self, z):
        sigma = torch.exp(self.log_sigma).clamp_min(1e-4)
        log_p = -0.5 * ((z / sigma) ** 2) - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
        return (-log_p / math.log(2.0)).mean(dim=1)


def sinusoidal_pe(L, d):
    pos = torch.arange(L, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(L, d); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
    return pe


def quantize_ste(z, bits, training):
    qmax = float(max(1, 2 ** (bits - 1) - 1))
    zq = (z * qmax).round().clamp(-qmax - 1, qmax) / qmax
    if training:
        zq = z + (zq - z).detach()
    return zq


class Encoder(nn.Module):
    def __init__(self, in_ch, widths=(16, 32, 48), d=48, heads=4, n_layers=2, ff=96,
                 latent=32, n_classes=3, dropout=0.1, use_transformer=True, bits=8,
                 train_erasure=0.0, pos_enc=False):
        super().__init__()
        self.bits, self.train_erasure, self.use_transformer, self.pos_enc = bits, train_erasure, use_transformer, pos_enc
        self.d = d
        self.stem = nn.Conv1d(in_ch, widths[0], 7, 2, 3)
        self.convs = nn.Sequential(*[SeparableConv1d(widths[i - 1], widths[i]) for i in range(1, len(widths))])
        if use_transformer and n_layers > 0:
            layer = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=ff, dropout=dropout,
                                               batch_first=True, activation="gelu")
            self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        else:
            self.transformer = None
        self.proj_latent = nn.Linear(d, latent)
        self.ee_head = nn.Linear(d, n_classes)
        self.entropy = EntropyBottleneck(latent)

    def features(self, x):
        h = self.convs(self.stem(x)).transpose(1, 2)          # [B, L, d]
        if self.pos_enc:
            h = h + sinusoidal_pe(h.shape[1], self.d).to(h.device)[None]
        if self.transformer is not None:
            h = self.transformer(h)
        return h.mean(dim=1)                                  # [B, d]

    def forward(self, x):
        pooled = self.features(x)
        z = self.proj_latent(pooled)                          # continuous latent
        zq = quantize_ste(z, self.bits, self.training)
        if self.training and self.train_erasure > 0:
            keep = (torch.rand_like(zq) >= self.train_erasure).float()
            zq = zq * keep
        return zq, self.ee_head(pooled), self.entropy.rate_bits(zq), z


class Decoder(nn.Module):
    def __init__(self, latent=32, d=48, n_classes=3, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(latent, 2 * d), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(2 * d, d), nn.GELU())
        self.head = nn.Linear(d, n_classes)

    def forward(self, z):
        return self.head(self.mlp(z))

    def export_numpy(self):
        """Weights for a dependency-free NumPy re-implementation (see rev_np.py)."""
        l1, l2, l3 = self.mlp[0], self.mlp[3], self.head
        return {k: v.detach().cpu().numpy().astype("float32") for k, v in
                dict(W1=l1.weight, b1=l1.bias, W2=l2.weight, b2=l2.bias, W3=l3.weight, b3=l3.bias).items()}


def task_loss(logits_full, logits_ee, zq, rate_bits, y, gamma=0.05, lam=0.105, early_exit=True):
    ce_full = F.cross_entropy(logits_full, y)
    l_task = 0.5 * (ce_full + F.cross_entropy(logits_ee, y)) if early_exit else ce_full
    l_rate = rate_bits.mean()
    l_reg = (zq ** 2).mean()
    return l_task + gamma * l_rate + lam * l_reg, dict(task=l_task.item(), rate=l_rate.item(), reg=l_reg.item())


def count_params(m):
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad))


def count_macs(enc: Encoder, in_ch, T, latent=32, d=48, ff=96, n_classes=3):
    """Analytic multiply-accumulate count per window for one encoder forward pass,
    INCLUDING the attention projections and score/context matmuls that hook-based
    profilers (thop) miss because nn.MultiheadAttention is functional."""
    def out_len(L, k, s, p):
        return (L + 2 * p - k) // s + 1
    macs = 0
    w0 = enc.stem.out_channels
    L = out_len(T, 7, 2, 3); macs += in_ch * w0 * 7 * L
    c_in = w0
    for m in enc.convs:
        L = out_len(L, 5, 2, 2)
        macs += c_in * 5 * L                     # depthwise
        macs += c_in * m.pw.out_channels * L     # pointwise
        c_in = m.pw.out_channels
    if enc.transformer is not None:
        for _ in enc.transformer.layers:
            macs += 3 * d * d * L                # Q,K,V projections
            macs += 2 * L * L * d                # scores + context
            macs += d * d * L                    # output projection
            macs += 2 * d * ff * L               # feed-forward
    macs += d * latent + d * n_classes           # latent projection + early-exit head
    dec = latent * 2 * d + 2 * d * d + d * n_classes
    return int(macs), int(dec), int(L)
