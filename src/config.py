"""Shared configuration for the task-oriented semantic ECG codec.

All hyperparameters live here so that every reported number comes from a single,
documented configuration. Defaults correspond to the manuscript: two-lead MIT-BIH
input resampled to 125 Hz, 10-s windows, three AAMI classes, a 32-dimensional
latent quantised to 8 bits with a straight-through estimator.

Loss (semantic_encoder.sempq_loss):
    L = alpha * L_task + beta * L_KL + gamma * L_rate + delta * L_priv
where L_KL = 0.5 * mean(z^2) and L_priv = mean(z^2) are both l2 penalties on the
quantised latent (jointly equivalent to lambda * mean(z^2) with
lambda = 0.5 * beta + delta = 0.105, as written in the manuscript), L_rate is the
factorised-Gaussian code-length estimate, and L_task averages the cross-entropy
of the gateway decoder and of the early-exit head.
"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"


@dataclass
class EncoderConfig:
    window_sec: float = 10.0                 # seconds of signal per decision
    sampling_rate: int = 125                 # Hz after resampling (250/360 Hz studied in the revision)
    in_channels: int = 2                     # ECG leads (1 for MLII-only, 12 for PTB-XL)
    cnn_widths: tuple = (16, 32, 48)         # stem + two depthwise-separable blocks
    transformer_dim: int = 48
    transformer_heads: int = 4
    transformer_layers: int = 2              # 0 -> CNN-only variant (see revision/rev_model.py)
    latent_dim: int = 32
    num_classes: int = 3                     # AAMI EC57: normal / SVEB / VEB (2 for PTB-XL, WESAD)
    dropout: float = 0.1
    loss_alpha_task: float = 1.0
    loss_beta_kl: float = 0.01
    loss_gamma_rate: float = 0.05
    loss_delta_priv: float = 0.10
    latent_bits: int = 8                     # straight-through quantiser width during training
