"""Task-aware semantic encoder/decoder (§4.2).

A compact 1D-CNN backbone + small transformer head, jointly trained with a
four-term loss: L = alpha*L_task + beta*L_KL + gamma*L_rate + delta*L_priv.

Output: (logits, latent, rate_bits) — the latent is what gets transmitted,
quantized to INT8 and entropy-coded. Rate is approximated via a learned
factorized entropy model (Balle et al. 2018).

Designed to fit ~48k parameters, INT8 quantizable to ~64 kB flash.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import EncoderConfig


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class SeparableConv1d(nn.Module):
    """Depthwise-separable 1D convolution for parameter efficiency."""
    def __init__(self, c_in: int, c_out: int, kernel: int = 5, stride: int = 2):
        super().__init__()
        self.dw = nn.Conv1d(c_in, c_in, kernel, stride, kernel // 2, groups=c_in)
        self.pw = nn.Conv1d(c_in, c_out, 1)
        self.norm = nn.GroupNorm(4, c_out)

    def forward(self, x):
        return F.gelu(self.norm(self.pw(self.dw(x))))


class _EntropyBottleneck(nn.Module):
    """Learned factorized prior for rate estimation (simplified)."""
    def __init__(self, dim: int):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(dim))

    def rate_bits(self, z: torch.Tensor) -> torch.Tensor:
        # Approximate bits under a Gaussian factorized prior.
        sigma = torch.exp(self.log_sigma).clamp_min(1e-4)
        log_p = -0.5 * ((z / sigma) ** 2) - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
        bits = -log_p / math.log(2.0)
        return bits.mean(dim=(1,)) if bits.dim() == 2 else bits.mean(dim=(1, 2))


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------
class SemanticEncoder(nn.Module):
    """Node-side encoder E_phi.

    Input:  x  ∈ R^{B, C, T}  (multimodal time-series window)
    Output: z  ∈ R^{B, D}     (quantized latent),
            logits ∈ R^{B, K}  (early-exit prediction head),
            rate_bits ∈ R^{B}  (rate estimate)
    """
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.stem = nn.Conv1d(cfg.in_channels, cfg.cnn_widths[0], 7, 2, 3)
        convs = []
        widths = cfg.cnn_widths
        for i in range(1, len(widths)):
            convs.append(SeparableConv1d(widths[i - 1], widths[i]))
        self.convs = nn.Sequential(*convs)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.transformer_dim,
            nhead=cfg.transformer_heads,
            dim_feedforward=2 * cfg.transformer_dim,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.transformer_layers)

        self.proj_latent = nn.Linear(cfg.transformer_dim, cfg.latent_dim)
        self.ee_head = nn.Linear(cfg.transformer_dim, cfg.num_classes)  # early-exit head
        self.entropy = _EntropyBottleneck(cfg.latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.stem(x)
        h = self.convs(h)                         # [B, C, T']
        h = h.transpose(1, 2)                     # [B, T', C]
        h = self.transformer(h)                   # [B, T', D]
        pooled = h.mean(dim=1)                    # [B, D]
        z = self.proj_latent(pooled)              # [B, D_latent]
        # Straight-through INT8 quantization
        z_q = (z * 127.0).round().clamp(-128, 127) / 127.0
        if self.training:
            z_q = z + (z_q - z).detach()
        logits_ee = self.ee_head(pooled)
        rate = self.entropy.rate_bits(z_q)
        return z_q, logits_ee, rate


class SemanticDecoder(nn.Module):
    """Gateway-side decoder D_psi + classifier.

    Receives the quantized latent and produces the full task distribution.
    """
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.mlp = nn.Sequential(
            nn.Linear(cfg.latent_dim, 2 * cfg.transformer_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(2 * cfg.transformer_dim, cfg.transformer_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(cfg.transformer_dim, cfg.num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(self.mlp(z))


# ---------------------------------------------------------------------------
# Composite loss (§4.2.2)
# ---------------------------------------------------------------------------
def sempq_loss(logits_full, logits_ee, z, rate_bits, y, cfg: EncoderConfig):
    """L = alpha*L_task + beta*L_KL + gamma*L_rate + delta*L_priv.

    Privacy term (L_priv) is implemented here as an L2 penalty on the latent;
    the differentially-private noise is injected via gradient noise during
    training (see `train.py`, not included in scaffold)."""
    L_task_full = F.cross_entropy(logits_full, y)
    L_task_ee   = F.cross_entropy(logits_ee, y)
    L_task = 0.5 * (L_task_full + L_task_ee)

    # KL to unit-variance Gaussian prior (information-bottleneck surrogate)
    L_kl = 0.5 * (z ** 2).mean()

    # Rate (bits per sample) — differentiable
    L_rate = rate_bits.mean()

    # Privacy surrogate
    L_priv = (z.pow(2).mean(dim=1)).mean()

    total = (cfg.loss_alpha_task * L_task
             + cfg.loss_beta_kl  * L_kl
             + cfg.loss_gamma_rate * L_rate
             + cfg.loss_delta_priv * L_priv)
    return total, dict(task=L_task.item(), kl=L_kl.item(),
                       rate=L_rate.item(), priv=L_priv.item())


# ---------------------------------------------------------------------------
# Utility: count parameters
# ---------------------------------------------------------------------------
def param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


if __name__ == "__main__":
    cfg = EncoderConfig()
    enc = SemanticEncoder(cfg)
    dec = SemanticDecoder(cfg)
    x = torch.randn(4, cfg.in_channels, int(cfg.window_sec * cfg.sampling_rate))  # [4, 2, 1250]
    z, ee, rate = enc(x)
    logits = dec(z)
    print(f"latent shape: {z.shape}")
    print(f"encoder params: {param_count(enc):,}")
    print(f"decoder params: {param_count(dec):,}")
    print(f"rate (bits/sample, mean): {rate.mean().item():.2f}")
