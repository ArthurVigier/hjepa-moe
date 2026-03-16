"""
universal_jepa.py
==================
Build a JEPA world model from scratch on ANY dataset,
regardless of modality or shape.

Supported input forms (auto-detected or declared):
  - Tabular:      (N, D)              — sensor streams, finance, EHR
  - Sequence:     (N, T, D)           — time series, text embeddings, audio
  - Image:        (N, C, H, W)        — frames, spectrograms, heatmaps
  - Video:        (N, T, C, H, W)     — video, multi-frame sensors
  - Graph nodes:  (N, V, D)           — graph node features (V nodes)
  - Point cloud:  (N, P, 3)           — LiDAR, 3D scans
  - Multi-modal:  dict of the above   — any combination

The script:
  1. Auto-detects input shape and selects appropriate encoder backbone
  2. Builds a H-JEPA-MoE stack on top
  3. Trains with SIGReg (best for unknown distributions)
  4. Exports analysis: representation geometry, routing profiles,
     nearest-neighbor structure, linear separability score

Usage:
  # From numpy array (any shape)
  python universal_jepa.py --data path/to/data.npy --modality auto

  # From CSV (tabular)
  python universal_jepa.py --data path/to/data.csv --modality tabular

  # From folder of images
  python universal_jepa.py --data path/to/images/ --modality image

  # From folder of videos
  python universal_jepa.py --data path/to/videos/ --modality video

  # Multi-modal (JSON manifest)
  python universal_jepa.py --data path/to/manifest.json --modality multimodal

  # From HuggingFace dataset
  python universal_jepa.py --hf_dataset speech_commands --modality audio

  # Probe only (no training — just geometry analysis)
  python universal_jepa.py --data path/to/data.npy --probe_only
"""

import os
import sys
import json
import math
import time
import argparse
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

sys.path.insert(0, str(Path(__file__).parent))

from hjepa_moe.model import HJEPAMoE, HJEPAMoEConfig, LevelConfig
from hjepa_moe.losses.vicreg import VICRegLoss, SIGRegLoss
from hjepa_moe.utils import AverageMeter, cosine_schedule, set_lr, routing_entropy

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S", level=logging.INFO,
)
log = logging.getLogger("universal_jepa")

# ─────────────────────────────────────────────────────────────────────────────
# Modality detection
# ─────────────────────────────────────────────────────────────────────────────

MODALITIES = [
    "auto", "tabular", "sequence", "image",
    "video", "audio", "graph", "pointcloud", "multimodal",
]


def detect_modality(shape: Tuple[int, ...]) -> str:
    """Heuristic modality detection from tensor shape."""
    ndim = len(shape)
    if ndim == 1:
        return "tabular"         # (D,) — single sample
    if ndim == 2:
        # (N, D) or (T, D) — tabular batch or single sequence
        if shape[-1] <= 4:
            return "pointcloud"  # (N, 3) or (N, 4) — xyz or xyzr
        return "tabular"
    if ndim == 3:
        # (N, T, D) — sequences
        if shape[-1] in (1, 2, 3):
            return "pointcloud"
        return "sequence"
    if ndim == 4:
        # (N, C, H, W) or (N, T, C, D) — images or batched sequences
        if shape[1] in (1, 2, 3, 4):
            return "image"
        return "sequence"
    if ndim == 5:
        return "video"           # (N, T, C, H, W)
    return "sequence"


def shape_summary(shape: Tuple[int, ...], modality: str) -> str:
    labels = {
        "tabular":    "batch × features",
        "sequence":   "batch × time × features",
        "image":      "batch × channels × H × W",
        "video":      "batch × time × channels × H × W",
        "audio":      "batch × time × features",
        "graph":      "batch × nodes × features",
        "pointcloud": "batch × points × coords",
        "multimodal": "dict of tensors",
    }
    return f"{shape}  [{labels.get(modality, modality)}]"

# ─────────────────────────────────────────────────────────────────────────────
# Data loading — universal
# ─────────────────────────────────────────────────────────────────────────────

class UniversalDataset(Dataset):
    """
    Wraps ANY tensor-like data into a JEPA-compatible dataset.

    JEPA needs sequences to predict across — so we ensure every sample
    has a temporal dimension T. Strategy per modality:

    tabular    (N, D)         → treat D features as T=D sequence of scalars
                               → or chunk into T windows if D is large
    sequence   (N, T, D)      → direct, already has T
    image      (N, C, H, W)   → patch into T=num_patches spatial tokens
    video      (N, T, C, H, W)→ direct, already has T
    audio      (N, T, F)      → direct (T = time frames, F = freq bins)
    graph      (N, V, D)      → treat V nodes as T = graph walk sequence
    pointcloud (N, P, 3)      → sort by distance or random, T = P
    """

    def __init__(
        self,
        data:        Union[torch.Tensor, np.ndarray, List],
        modality:    str = "auto",
        seq_len:     int = 64,
        target_dim:  int = 128,
        labels:      Optional[torch.Tensor] = None,
        patch_size:  int = 8,         # for image patching
        augment:     bool = True,
    ):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        elif isinstance(data, list):
            data = torch.stack([torch.tensor(d).float() for d in data])
        
        self.raw_data   = data
        self.labels     = labels
        self.seq_len    = seq_len
        self.patch_size = patch_size
        self.augment    = augment

        # Auto-detect modality
        if modality == "auto":
            modality = detect_modality(tuple(data.shape))
        self.modality = modality

        # Build preprocessor
        self.prep, self.input_dim = self._build_preprocessor(data, target_dim)
        log.info(f"Dataset: modality={modality}  raw_shape={tuple(data.shape)}  "
                 f"input_dim={self.input_dim}  seq_len={seq_len}")

    def _build_preprocessor(self, data, target_dim):
        """Returns (preprocessor_fn, output_dim_per_token)."""
        shape = tuple(data.shape)
        mod   = self.modality

        if mod == "tabular":
            # (N, D) → (N, T, 1) via sliding windows
            # If D < seq_len, repeat/pad; if D > seq_len, chunk
            D = shape[-1] if len(shape) > 1 else 1
            def prep(x):
                # x: (D,) or (B, D)
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                B, D_ = x.shape
                # Expand to seq_len via interpolation
                x = F.interpolate(
                    x.unsqueeze(1),                # (B, 1, D)
                    size=self.seq_len,
                    mode="linear", align_corners=False
                ).squeeze(1)                       # (B, seq_len)
                return x.unsqueeze(-1)             # (B, seq_len, 1)
            return prep, 1

        elif mod == "sequence":
            # (N, T, D) → already good, subsample/pad T to seq_len
            D = shape[-1]
            def prep(x):
                # x: (B, T, D) or (T, D)
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                B, T, D_ = x.shape
                if T != self.seq_len:
                    x = F.interpolate(
                        x.transpose(1, 2),           # (B, D, T)
                        size=self.seq_len,
                        mode="linear", align_corners=False
                    ).transpose(1, 2)                # (B, seq_len, D)
                return x
            return prep, D

        elif mod == "image":
            # (N, C, H, W) → patch into seq of spatial tokens
            C = shape[1] if len(shape) == 4 else 3
            P = self.patch_size
            def prep(x):
                # x: (B, C, H, W) or (C, H, W)
                if x.dim() == 3:
                    x = x.unsqueeze(0)
                B, C_, H, W = x.shape
                # Resize to ensure divisibility
                H_ = (H // P) * P
                W_ = (W // P) * P
                x = F.interpolate(x, (H_, W_), mode="bilinear", align_corners=False)
                # Patch: (B, C, H//P, P, W//P, P) → (B, N_patches, C*P*P)
                x = x.unfold(2, P, P).unfold(3, P, P)   # (B, C, nh, nw, P, P)
                B_, C_, nh, nw, _, _ = x.shape
                x = x.permute(0, 2, 3, 1, 4, 5).reshape(B_, nh*nw, C_*P*P)
                # Subsample to seq_len
                if x.shape[1] != self.seq_len:
                    x = F.interpolate(
                        x.transpose(1, 2), size=self.seq_len, mode="linear",
                        align_corners=False
                    ).transpose(1, 2)
                return x
            patch_dim = C * P * P
            return prep, patch_dim

        elif mod == "video":
            # (N, T, C, H, W) → flatten spatial, keep T
            C = shape[2] if len(shape) == 5 else 3
            H = shape[3] if len(shape) == 5 else 64
            W = shape[4] if len(shape) == 5 else 64
            def prep(x):
                if x.dim() == 4:
                    x = x.unsqueeze(0)               # (1, T, C, H, W)
                B, T, C_, H_, W_ = x.shape
                # Pool spatial → single feature per frame
                x = x.reshape(B*T, C_, H_, W_)
                x = F.adaptive_avg_pool2d(x, 8)      # (B*T, C, 8, 8)
                x = x.flatten(1)                     # (B*T, C*64)
                x = x.reshape(B, T, -1)              # (B, T, C*64)
                # Resample T → seq_len
                if T != self.seq_len:
                    x = F.interpolate(
                        x.transpose(1, 2), size=self.seq_len, mode="linear",
                        align_corners=False
                    ).transpose(1, 2)
                return x
            spatial_dim = C * 8 * 8
            return prep, spatial_dim

        elif mod in ("audio",):
            # (N, T, F) — same as sequence
            F_dim = shape[-1]
            def prep(x):
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                B, T, F_ = x.shape
                if T != self.seq_len:
                    x = F.interpolate(
                        x.transpose(1, 2), size=self.seq_len, mode="linear",
                        align_corners=False
                    ).transpose(1, 2)
                return x
            return prep, F_dim

        elif mod == "graph":
            # (N, V, D) → treat nodes as sequence tokens
            D = shape[-1]
            def prep(x):
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                B, V, D_ = x.shape
                if V != self.seq_len:
                    x = F.interpolate(
                        x.transpose(1, 2), size=self.seq_len, mode="linear",
                        align_corners=False
                    ).transpose(1, 2)
                return x
            return prep, D

        elif mod == "pointcloud":
            # (N, P, 3) → sort by norm, treat as sequence
            def prep(x):
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                B, P, _ = x.shape
                # Sort points by L2 norm from origin
                norms = x.norm(dim=-1)               # (B, P)
                idx   = norms.argsort(dim=1)
                x = x.gather(1, idx.unsqueeze(-1).expand_as(x))
                # Subsample to seq_len
                if P != self.seq_len:
                    x = F.interpolate(
                        x.transpose(1, 2), size=self.seq_len, mode="linear",
                        align_corners=False
                    ).transpose(1, 2)
                return x
            return prep, 3

        else:
            # Fallback: flatten and treat as sequence
            flat_dim = int(np.prod(shape[1:])) if len(shape) > 1 else shape[0]
            def prep(x):
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                x = x.reshape(x.shape[0], -1)
                x = F.interpolate(
                    x.unsqueeze(1), size=self.seq_len * flat_dim,
                    mode="linear", align_corners=False
                ).squeeze(1)
                return x.reshape(x.shape[0], self.seq_len, -1)
            return prep, flat_dim

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        x = self.raw_data[idx]
        x = self.prep(x.unsqueeze(0) if x.dim() == len(self.raw_data.shape) - 1
                       else x).squeeze(0)   # (seq_len, input_dim)

        # Light augmentation for SSL robustness
        if self.augment and self.training_mode:
            # Gaussian noise
            x = x + torch.randn_like(x) * 0.01
            # Random temporal jitter (shift by 1-2 steps)
            shift = torch.randint(0, 3, (1,)).item()
            if shift > 0:
                x = torch.cat([x[shift:], x[:shift]], dim=0)

        label = self.labels[idx] if self.labels is not None else torch.tensor(0)
        return x, label

    # Flag for augmentation — set True during training, False during eval
    training_mode: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Input encoder — maps raw tokens to d_model
# ─────────────────────────────────────────────────────────────────────────────

class UniversalInputEncoder(nn.Module):
    """
    Maps raw tokens (seq_len, input_dim) → (seq_len, d_model).

    Three strategies based on input_dim:
      small  (≤32):   linear projection + positional encoding
      medium (33-512): 2-layer MLP + positional encoding
      large  (>512):  1D conv + positional encoding (efficient for large patches)

    For all modalities this is the ONLY part that touches raw data.
    Everything above is modality-agnostic.
    """

    def __init__(self, input_dim: int, d_model: int, seq_len: int,
                 dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.d_model   = d_model
        self.seq_len   = seq_len

        if input_dim <= 32:
            # Small: linear
            self.proj = nn.Sequential(
                nn.Linear(input_dim, d_model, bias=False),
                nn.RMSNorm(d_model),
            )
        elif input_dim <= 512:
            # Medium: 2-layer MLP
            self.proj = nn.Sequential(
                nn.Linear(input_dim, d_model * 2, bias=False),
                nn.SiLU(),
                nn.Linear(d_model * 2, d_model, bias=False),
                nn.RMSNorm(d_model),
            )
        else:
            # Large: 1D conv (efficient for large patches)
            self.proj = nn.Sequential(
                nn.Conv1d(input_dim, d_model * 2, kernel_size=3,
                           padding=1, bias=False),
                nn.SiLU(),
                nn.Conv1d(d_model * 2, d_model, kernel_size=1, bias=False),
            )
            self._use_conv = True

        self._use_conv = input_dim > 512
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)
        if self._use_conv:
            x = self.proj(x.transpose(1, 2)).transpose(1, 2)  # (B, T, d_model)
        else:
            x = self.proj(x)           # (B, T, d_model)
        x = x + self.pos_embed[:, :x.shape[1]]
        return self.drop(x)


# ─────────────────────────────────────────────────────────────────────────────
# Universal JEPA model
# ─────────────────────────────────────────────────────────────────────────────

class UniversalJEPA(nn.Module):
    """
    Complete JEPA world model for any input modality.

    Architecture:
        raw input  (B, T, input_dim)
              ↓
        UniversalInputEncoder   → (B, T, d_model)    [modality-specific]
              ↓
        TemporalEncoder_1       → (B, T/k, d_model)  [temporal compression]
              ↓
        MoEPredictor_1          → predicts next state at level 1
              ↓
        TemporalEncoder_2       → (B, T/k², d_model)
              ↓
        MoEPredictor_2          → predicts next state at level 2
              ↓
        ...

    The JEPA objective: predict the target encoder's representation
    of a future/masked state from a context, in embedding space only.
    No reconstruction. No token prediction.
    """

    def __init__(
        self,
        input_dim:    int,
        d_model:      int = 256,
        seq_len:      int = 64,
        n_levels:     int = 3,
        pool_factor:  int = 4,
        n_experts:    int = 4,
        top_k:        int = 2,
        loss_type:    str = "sigreg",
        ema_decay:    float = 0.996,
        dropout:      float = 0.0,
    ):
        super().__init__()
        self.d_model   = d_model
        self.seq_len   = seq_len
        self.n_levels  = n_levels

        # Input encoder (modality bridge)
        self.input_encoder = UniversalInputEncoder(
            input_dim, d_model, seq_len, dropout
        )

        # Build JEPA stack (reuse HJEPAMoE internals)
        import copy
        from hjepa_moe.encoders.temporal import TemporalEncoder
        from hjepa_moe.predictors.moe_predictor import MoEPredictor

        self.temporal_encoders = nn.ModuleList()
        self.moe_predictors    = nn.ModuleList()
        self.target_encoders   = []

        for ℓ in range(n_levels):
            expert_type = "ffn" if ℓ == 0 else "transformer"
            enc  = TemporalEncoder(d_model, d_model, pool_factor=pool_factor,
                                    pooling="attention")
            pred = MoEPredictor(d_model, d_model, d_model=d_model,
                                 n_experts=n_experts, top_k=top_k,
                                 expert_type=expert_type)
            self.temporal_encoders.append(enc)
            self.moe_predictors.append(pred)

            t_enc = copy.deepcopy(enc)
            for p in t_enc.parameters():
                p.requires_grad_(False)
            self.target_encoders.append(t_enc)

        # Loss
        if loss_type == "sigreg":
            self.loss_fn = SIGRegLoss(d_model=d_model)
        elif loss_type == "vicreg":
            self.loss_fn = VICRegLoss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        self.ema_decay  = ema_decay
        self.level_weights = [2.0 ** ℓ for ℓ in range(n_levels)]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        x: (B, T, input_dim) — raw tokens, already preprocessed by UniversalDataset
        """
        # Encode to d_model
        states = self.input_encoder(x)    # (B, T, d_model)

        total_loss = torch.tensor(0.0, device=x.device)
        stats = {}

        for ℓ, (enc, t_enc, pred) in enumerate(zip(
            self.temporal_encoders, self.target_encoders, self.moe_predictors
        )):
            states_ℓ = enc(states)    # (B, T/k, d_model)
            with torch.no_grad():
                target_ℓ = t_enc(states)

            B, Tl, d = states_ℓ[:, :-1].shape
            ctx    = states_ℓ[:, :-1].reshape(B * Tl, d)
            target = target_ℓ[:, 1: ].reshape(B * Tl, d)

            pred_s, aux = pred(ctx)
            level_loss, level_stats = self.loss_fn(pred_s, target.detach())

            if aux is not None:
                level_loss = level_loss + aux

            total_loss = total_loss + self.level_weights[ℓ] * level_loss
            stats[f"level_{ℓ}"] = level_stats
            states = states_ℓ.detach()

        stats["loss_total"] = total_loss.item()
        return total_loss, stats

    @torch.no_grad()
    def update_ema(self):
        for enc, t_enc in zip(self.temporal_encoders, self.target_encoders):
            for po, pt in zip(enc.parameters(), t_enc.parameters()):
                pt.data = self.ema_decay * pt.data + (1 - self.ema_decay) * po.data

    @torch.no_grad()
    def encode(self, x: torch.Tensor, level: int = -1) -> torch.Tensor:
        """Get level-ℓ representation. level=-1 → top level."""
        states = self.input_encoder(x)
        for ℓ, enc in enumerate(self.temporal_encoders):
            states = enc(states)
            if ℓ == level or (level < 0 and ℓ == self.n_levels - 1):
                break
        return states.mean(dim=1)    # (B, d_model) — pool time


# ─────────────────────────────────────────────────────────────────────────────
# Analysis suite
# ─────────────────────────────────────────────────────────────────────────────

class JEPAAnalyzer:
    """
    In-depth analysis of a trained (or randomly initialized) UniversalJEPA.

    Runs a full battery of geometric and functional tests:
      1. Representation geometry    (isotropy, rank, norm distribution)
      2. Per-level routing entropy  (expert specialization)
      3. Temporal predictability    (how well each level predicts next state)
      4. Linear separability        (kNN accuracy at each level)
      5. CKA similarity matrix      (between levels, and vs random baseline)
      6. Nearest-neighbor structure (retrieval R@1, R@5, R@10)
      7. Dimensional collapse score (effective rank metric)
    """

    def __init__(self, model: UniversalJEPA, device: str = "cpu"):
        self.model  = model.to(device)
        self.device = device

    @torch.no_grad()
    def extract_representations(
        self, loader: DataLoader, n_batches: int = 20
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Extract representations at all levels + labels."""
        self.model.eval()
        all_reps    = [[] for _ in range(self.model.n_levels + 1)]
        all_labels  = []

        for i, (x, y) in enumerate(loader):
            if i >= n_batches:
                break
            x = x.to(self.device)
            states_enc = self.model.input_encoder(x)   # level 0
            all_reps[0].append(states_enc.mean(1).cpu())
            states = states_enc

            for ℓ, enc in enumerate(self.model.temporal_encoders):
                states = enc(states)
                all_reps[ℓ+1].append(states.mean(1).cpu())

            all_labels.append(y.cpu())

        reps   = [torch.cat(r) for r in all_reps]
        labels = torch.cat(all_labels)
        self.model.train()
        return reps, labels

    # ── 1. Representation geometry ────────────────────────────

    def geometry_report(self, reps: List[torch.Tensor]) -> dict:
        """
        For each level, compute:
          - effective_rank: how many dimensions carry signal
          - isotropy:       variance uniformity across dimensions
          - mean_norm:      average L2 norm
          - std_norm:       norm variance (collapsed = very low)
        """
        report = {}
        for ℓ, r in enumerate(reps):
            r_n = r - r.mean(0)
            cov = (r_n.T @ r_n) / (r_n.shape[0] - 1)
            eigvals = torch.linalg.eigvalsh(cov).abs()
            eigvals = eigvals / eigvals.sum()

            # Effective rank: exp(H) where H = -sum(p log p)
            p = eigvals.clamp(min=1e-8)
            H = -(p * p.log()).sum().item()
            eff_rank = math.exp(H)

            # Isotropy: ratio of min to max eigenvalue
            isotropy = (eigvals.min() / eigvals.max()).item()

            norms = r.norm(dim=-1)
            report[f"level_{ℓ}"] = {
                "effective_rank":  round(eff_rank, 2),
                "isotropy":        round(isotropy, 4),
                "mean_norm":       round(norms.mean().item(), 4),
                "std_norm":        round(norms.std().item(), 4),
                "dim":             r.shape[-1],
            }
        return report

    # ── 2. Routing entropy ────────────────────────────────────

    def routing_report(self, loader: DataLoader, n_batches: int = 5) -> dict:
        self.model.eval()
        report = {}
        for ℓ, pred in enumerate(self.model.moe_predictors):
            all_usage = []
            for i, (x, _) in enumerate(loader):
                if i >= n_batches:
                    break
                x = x.to(self.device)
                states = self.model.input_encoder(x)
                for enc in self.model.temporal_encoders[:ℓ+1]:
                    states = enc(states)
                s_flat = states[:, :-1].reshape(-1, states.shape[-1])
                with torch.no_grad():
                    stats = pred.get_routing_stats(s_flat)
                all_usage.append(stats["expert_usage"])
            usage = np.mean(all_usage, axis=0)
            H     = routing_entropy(usage)
            H_max = math.log(pred.router.n_experts)
            report[f"level_{ℓ+1}"] = {
                "entropy":          round(H, 4),
                "max_entropy":      round(H_max, 4),
                "normalized_H":     round(H / H_max, 4),
                "per_expert_usage": usage.tolist(),
                "collapsed":        H < 0.5 * H_max,
            }
        self.model.train()
        return report

    # ── 3. Temporal predictability ────────────────────────────

    @torch.no_grad()
    def predictability_report(self, loader: DataLoader, n_batches: int = 5) -> dict:
        """
        Measure 1-step prediction error at each level.
        Lower = better (model predicts next state more accurately).
        """
        self.model.eval()
        report = {}
        for ℓ, (enc, pred) in enumerate(
            zip(self.model.temporal_encoders, self.model.moe_predictors)
        ):
            errors = []
            for i, (x, _) in enumerate(loader):
                if i >= n_batches:
                    break
                x = x.to(self.device)
                states = self.model.input_encoder(x)
                for enc_prev in self.model.temporal_encoders[:ℓ]:
                    states = enc_prev(states)
                states_ℓ = enc(states)
                ctx    = states_ℓ[:, :-1].reshape(-1, states_ℓ.shape[-1])
                target = states_ℓ[:, 1: ].reshape(-1, states_ℓ.shape[-1])
                pred_s, _ = pred(ctx)
                err = F.mse_loss(pred_s, target).item()
                errors.append(err)
            report[f"level_{ℓ+1}"] = {
                "mean_prediction_error": round(float(np.mean(errors)), 6),
                "std_prediction_error":  round(float(np.std(errors)),  6),
            }
        self.model.train()
        return report

    # ── 4. Linear separability (kNN) ─────────────────────────

    def separability_report(
        self, reps: List[torch.Tensor], labels: torch.Tensor, k: int = 5
    ) -> dict:
        """
        kNN accuracy at each level.
        High accuracy = representations are semantically organized.
        Compare to level 0 (raw encoder) as baseline.
        """
        report = {}
        n = len(labels)
        if n < k + 1:
            return {"error": "Not enough samples for kNN"}

        for ℓ, r in enumerate(reps):
            r_n = F.normalize(r, dim=-1)   # (N, d)
            sims = r_n @ r_n.T             # (N, N)
            sims.fill_diagonal_(-1e9)      # exclude self

            topk_idx  = sims.topk(k, dim=-1).indices  # (N, k)
            topk_lbls = labels[topk_idx]               # (N, k)
            vote      = topk_lbls.mode(dim=-1).values  # (N,)
            acc       = (vote == labels).float().mean().item()

            report[f"level_{ℓ}"] = {
                "knn_accuracy": round(acc, 4),
                "k": k,
            }
        return report

    # ── 5. CKA similarity matrix ──────────────────────────────

    def cka_matrix(self, reps: List[torch.Tensor]) -> np.ndarray:
        """
        Compute CKA (Centered Kernel Alignment) between all level pairs.
        Values close to 1 = similar representations.
        Values close to 0 = orthogonal / independent representations.
        """
        L = len(reps)
        matrix = np.zeros((L, L))

        def _hsic(X: torch.Tensor, Y: torch.Tensor) -> float:
            n = X.shape[0]
            K = X @ X.T
            L_ = Y @ Y.T
            H = torch.eye(n) - torch.ones(n, n) / n
            Kc = H @ K @ H
            Lc = H @ L_ @ H
            return (Kc * Lc).sum().item() / (n - 1) ** 2

        for i in range(L):
            for j in range(L):
                Xi = F.normalize(reps[i], dim=-1)
                Xj = F.normalize(reps[j], dim=-1)
                hsic_ij = _hsic(Xi, Xj)
                hsic_ii = _hsic(Xi, Xi)
                hsic_jj = _hsic(Xj, Xj)
                denom = math.sqrt(hsic_ii * hsic_jj)
                matrix[i, j] = hsic_ij / denom if denom > 1e-8 else 0.0

        return np.round(matrix, 4)

    # ── 6. Nearest-neighbor retrieval ────────────────────────

    def retrieval_report(
        self,
        reps:   List[torch.Tensor],
        labels: torch.Tensor,
    ) -> dict:
        """R@1, R@5, R@10 at each level."""
        report = {}
        for ℓ, r in enumerate(reps):
            r_n = F.normalize(r, dim=-1)
            sims = r_n @ r_n.T
            sims.fill_diagonal_(-1e9)
            n = len(labels)

            for k in [1, 5, 10]:
                if k >= n:
                    continue
                topk = sims.topk(k, dim=-1).indices    # (N, k)
                hits = (labels[topk] == labels.unsqueeze(1)).any(dim=1)
                recall = hits.float().mean().item()
                report.setdefault(f"level_{ℓ}", {})[f"R@{k}"] = round(recall, 4)

        return report

    # ── Full report ───────────────────────────────────────────

    def full_report(
        self, loader: DataLoader, n_batches: int = 20, save_path: str = None
    ) -> dict:
        log.info("Extracting representations...")
        reps, labels = self.extract_representations(loader, n_batches)

        log.info("Computing geometry report...")
        geometry = self.geometry_report(reps)

        log.info("Computing routing report...")
        routing = self.routing_report(loader)

        log.info("Computing predictability report...")
        predictability = self.predictability_report(loader)

        log.info("Computing separability (kNN)...")
        separability = self.separability_report(reps, labels)

        log.info("Computing CKA matrix...")
        cka = self.cka_matrix(reps)

        log.info("Computing retrieval metrics...")
        retrieval = self.retrieval_report(reps, labels)

        report = {
            "geometry":       geometry,
            "routing":        routing,
            "predictability": predictability,
            "separability":   separability,
            "cka_matrix":     cka.tolist(),
            "retrieval":      retrieval,
        }

        self._print_report(report, cka)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(report, f, indent=2)
            log.info(f"Report saved to {save_path}")

        return report

    def _print_report(self, report: dict, cka: np.ndarray):
        log.info("\n" + "=" * 65)
        log.info("H-JEPA-MoE ANALYSIS REPORT")
        log.info("=" * 65)

        log.info("\n── Representation Geometry ──────────────────────────────")
        log.info(f"{'Level':<10} {'eff_rank':>10} {'isotropy':>10} {'mean_norm':>12}")
        for k, v in report["geometry"].items():
            log.info(f"{k:<10} {v['effective_rank']:>10.2f} "
                     f"{v['isotropy']:>10.4f} {v['mean_norm']:>12.4f}")

        log.info("\n── MoE Routing Entropy ──────────────────────────────────")
        log.info(f"{'Level':<10} {'entropy':>10} {'max_H':>8} {'norm_H':>8} {'collapsed':>12}")
        for k, v in report["routing"].items():
            log.info(f"{k:<10} {v['entropy']:>10.4f} {v['max_entropy']:>8.4f} "
                     f"{v['normalized_H']:>8.4f} {str(v['collapsed']):>12}")

        log.info("\n── Temporal Predictability ──────────────────────────────")
        for k, v in report["predictability"].items():
            log.info(f"  {k}: pred_error={v['mean_prediction_error']:.6f} "
                     f"± {v['std_prediction_error']:.6f}")

        log.info("\n── kNN Separability ─────────────────────────────────────")
        for k, v in report["separability"].items():
            log.info(f"  {k}: kNN-{v['k']} acc = {v['knn_accuracy']:.4f}")

        log.info("\n── Retrieval (R@k) ──────────────────────────────────────")
        for k, v in report["retrieval"].items():
            metrics = "  ".join(f"{mk}={mv}" for mk, mv in v.items())
            log.info(f"  {k}: {metrics}")

        log.info("\n── CKA Similarity Matrix ────────────────────────────────")
        log.info("  (rows/cols = levels 0..L, closer to 1 = more similar)")
        header = "        " + "".join(f"  L{i}" for i in range(cka.shape[0]))
        log.info(header)
        for i, row in enumerate(cka):
            cells = "".join(f"  {v:.2f}" for v in row)
            log.info(f"  L{i}    {cells}")
        log.info("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model:      UniversalJEPA,
    loader:     DataLoader,
    val_loader: DataLoader,
    cfg:        dict,
    device:     str,
) -> None:
    tc     = cfg.get("training", {})
    steps  = tc.get("max_steps", 20000)
    warmup = tc.get("warmup_steps", 500)
    lr     = tc.get("lr", 3e-4)
    min_lr = tc.get("min_lr", 1e-6)
    wd     = tc.get("weight_decay", 0.05)
    clip   = tc.get("grad_clip", 1.0)
    log_ev = tc.get("log_every", 100)
    eval_ev= tc.get("eval_every", 2000)
    save_ev= tc.get("save_every", 5000)
    run    = tc.get("run_name", "universal_jepa")
    ckpt   = Path(tc.get("checkpoint_dir", "checkpoints"))
    ckpt.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
                                   betas=(0.9, 0.95))
    use_amp   = tc.get("mixed_precision", False) and device.startswith("cuda")
    scaler    = torch.cuda.amp.GradScaler() if use_amp else None
    meter     = AverageMeter()
    analyzer  = JEPAAnalyzer(model, device)
    step      = 0

    model.train()
    log.info(f"Training: {steps} steps  lr={lr:.2e}  device={device}")

    while step < steps:
        for x, _ in loader:
            if step >= steps:
                break
            x = x.to(device)
            current_lr = cosine_schedule(step, steps, warmup, lr, min_lr)
            set_lr(optimizer, current_lr)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16,
                                          enabled=use_amp):
                loss, stats = model(x)

            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

            model.update_ema()
            meter.update({"loss": loss.item()})
            for k, v in stats.items():
                if isinstance(v, dict):
                    meter.update({f"{k}/{k2}": v2 for k2, v2 in v.items()
                                   if isinstance(v2, float)})

            if step % log_ev == 0:
                avg = meter.avg()
                log.info(f"step={step:6d}  lr={current_lr:.2e}  "
                          f"loss={avg.get('loss', 0):.4f}")
                meter.reset()

            if step % eval_ev == 0 and step > 0:
                log.info(f"\nEvaluation at step {step}")
                report = analyzer.full_report(val_loader, n_batches=10,
                                              save_path=str(ckpt / f"{run}_report_step{step}.json"))

            if step % save_ev == 0 and step > 0:
                torch.save({"model": model.state_dict(), "step": step, "cfg": cfg},
                            ckpt / f"{run}_step{step}.pt")
                log.info(f"Saved checkpoint: step {step}")

            step += 1

    # Final report
    log.info("\nFinal analysis:")
    analyzer.full_report(val_loader, save_path=str(ckpt / f"{run}_final_report.json"))
    torch.save({"model": model.state_dict(), "step": step, "cfg": cfg},
                ckpt / f"{run}_final.pt")
    log.info("Done.")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_data(
    path: str, modality: str = "auto"
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Load data from various formats. Returns (data_tensor, labels_or_None).

    Supported:
      .npy / .npz       numpy arrays
      .csv              tabular data (last column = label if dtype int)
      .pt / .pth        PyTorch tensors or dicts
      folder/           image or video folder
      .json             manifest for multi-modal data
    """
    path = Path(path)
    data, labels = None, None

    if path.suffix == ".npy":
        arr  = np.load(str(path))
        data = torch.from_numpy(arr).float()

    elif path.suffix == ".npz":
        npz = np.load(str(path))
        key = "data" if "data" in npz else list(npz.keys())[0]
        data = torch.from_numpy(npz[key]).float()
        if "labels" in npz:
            labels = torch.from_numpy(npz["labels"]).long()

    elif path.suffix == ".csv":
        import csv as _csv
        rows = []
        with open(str(path)) as f:
            reader = _csv.reader(f)
            header = next(reader, None)
            for row in reader:
                rows.append([float(v) for v in row])
        arr  = np.array(rows, dtype=np.float32)
        # Heuristic: if last column is integer labels
        if np.all(arr[:, -1] == arr[:, -1].astype(int)):
            labels = torch.from_numpy(arr[:, -1].astype(np.int64))
            data   = torch.from_numpy(arr[:, :-1])
        else:
            data   = torch.from_numpy(arr)

    elif path.suffix in (".pt", ".pth"):
        obj = torch.load(str(path), map_location="cpu")
        if isinstance(obj, dict):
            data   = obj.get("data", obj.get("x", obj.get("X")))
            labels = obj.get("labels", obj.get("y", obj.get("Y")))
        else:
            data = obj

    elif path.is_dir():
        # Image or video folder — load as paths, convert to random tensors for now
        exts   = {".png", ".jpg", ".jpeg", ".mp4", ".avi", ".webm"}
        files  = [f for f in path.rglob("*") if f.suffix.lower() in exts]
        n = min(len(files), 1000)
        if modality == "image" or any(f.suffix in {".png", ".jpg", ".jpeg"}
                                       for f in files[:5]):
            data = torch.rand(n, 3, 64, 64)
            log.warning(f"Image folder: loaded {n} placeholder tensors. "
                        f"Replace with real image loading for production.")
        else:
            data = torch.rand(n, 16, 3, 64, 64)
            log.warning(f"Video folder: loaded {n} placeholder tensors. "
                        f"Replace with real video loading for production.")

    elif path.suffix == ".json":
        with open(str(path)) as f:
            manifest = json.load(f)
        # Expect {"data": [...], "labels": [...]}
        data   = torch.tensor(manifest["data"]).float()
        labels = torch.tensor(manifest.get("labels", [0]*len(manifest["data"]))).long()

    else:
        raise ValueError(f"Unsupported format: {path.suffix}. "
                          f"Supported: .npy .npz .csv .pt .pth folder .json")

    if data is None:
        raise RuntimeError(f"Could not load data from {path}")

    log.info(f"Loaded: {tuple(data.shape)}  dtype={data.dtype}  "
              f"labels={'yes' if labels is not None else 'no'}")
    return data, labels


def load_hf_dataset(name: str, split: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a HuggingFace dataset and convert to tensor."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets transformers")

    ds = load_dataset(name, split=split)
    log.info(f"HF dataset '{name}': {len(ds)} samples, features: {ds.features}")

    # Convert to tensor (assumes numeric or audio features)
    if "input_values" in ds.features:     # audio
        data   = torch.stack([torch.tensor(s["input_values"]).float() for s in ds])
        labels = torch.tensor([s.get("label", 0) for s in ds])
    elif "pixel_values" in ds.features:   # vision
        data   = torch.stack([torch.tensor(s["pixel_values"]).float() for s in ds])
        labels = torch.tensor([s.get("label", 0) for s in ds])
    else:
        # Generic: use first numeric feature
        key    = [k for k, v in ds.features.items() if "Sequence" in str(type(v))][0]
        data   = torch.stack([torch.tensor(s[key]).float() for s in ds])
        labels = torch.tensor([s.get("label", 0) for s in ds])

    return data, labels


# ─────────────────────────────────────────────────────────────────────────────
# Auto model sizing
# ─────────────────────────────────────────────────────────────────────────────

def auto_config(
    input_dim: int,
    n_samples: int,
    modality:  str,
    target_gpu_gb: float = 8.0,
) -> dict:
    """
    Automatically size model based on data and available GPU memory.
    Returns a training config dict.

    Heuristics based on empirical JEPA scaling:
      - d_model scales with sqrt(input_dim)
      - n_experts scales with log2(n_samples)
      - batch_size scales with available GPU memory
    """
    d_model    = max(64, min(512, int(math.sqrt(input_dim) * 8)))
    d_model    = 2 ** round(math.log2(d_model))   # round to power of 2
    n_experts  = max(2, min(8, int(math.log2(max(n_samples, 100)) - 5)))
    n_levels   = 2 if n_samples < 5000 else 3
    batch_size = max(16, min(256, int(target_gpu_gb * 1000 / (d_model * 2))))
    seq_len    = 64
    max_steps  = max(5000, min(100000, n_samples * 20 // batch_size))

    log.info(f"Auto-config: d_model={d_model}  n_experts={n_experts}  "
              f"n_levels={n_levels}  batch={batch_size}  steps={max_steps}")

    return {
        "model": {
            "d_model":     d_model,
            "seq_len":     seq_len,
            "n_levels":    n_levels,
            "pool_factor": 4,
            "n_experts":   n_experts,
            "top_k":       max(1, n_experts // 2),
            "loss_type":   "sigreg",
            "ema_decay":   0.996,
        },
        "training": {
            "lr":            3e-4,
            "min_lr":        1e-6,
            "weight_decay":  0.05,
            "warmup_steps":  max(200, max_steps // 20),
            "max_steps":     max_steps,
            "grad_clip":     1.0,
            "log_every":     max(10, max_steps // 100),
            "eval_every":    max(500, max_steps // 10),
            "save_every":    max(1000, max_steps // 5),
            "mixed_precision": torch.cuda.is_available(),
            "run_name":      f"ujjepa_{modality}",
            "checkpoint_dir": "checkpoints/universal",
        },
        "data": {
            "batch_size": batch_size,
            "seq_len":    seq_len,
            "n_classes":  10,    # placeholder
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Universal JEPA — train from any data shape",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From numpy array (auto-detect modality)
  python universal_jepa.py --data data.npy

  # From CSV
  python universal_jepa.py --data data.csv --modality tabular

  # From image folder
  python universal_jepa.py --data ./images --modality image

  # From video folder
  python universal_jepa.py --data ./videos --modality video

  # From HuggingFace
  python universal_jepa.py --hf_dataset speech_commands --modality audio

  # Probe only (no training, just geometry analysis)
  python universal_jepa.py --data data.npy --probe_only

  # Custom d_model and n_levels
  python universal_jepa.py --data data.npy --d_model 512 --n_levels 3 --n_experts 8

  # Save a detailed report
  python universal_jepa.py --data data.npy --report_path report.json
        """,
    )

    # Data source
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--data",       default=None, help="Path to data file or folder")
    src.add_argument("--hf_dataset", default=None, help="HuggingFace dataset name")

    # Modality
    parser.add_argument("--modality", default="auto", choices=MODALITIES)

    # Model overrides
    parser.add_argument("--d_model",    type=int,   default=None)
    parser.add_argument("--n_levels",   type=int,   default=None)
    parser.add_argument("--n_experts",  type=int,   default=None)
    parser.add_argument("--top_k",      type=int,   default=None)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--loss_type",  default="sigreg", choices=["sigreg","vicreg"])

    # Training overrides
    parser.add_argument("--max_steps",  type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--gpu_gb",     type=float, default=8.0,
                         help="Available GPU memory in GB (for auto-sizing)")

    # Modes
    parser.add_argument("--probe_only", action="store_true",
                         help="Skip training, just run analysis on random init")
    parser.add_argument("--resume",     default=None, help="Path to checkpoint to resume")
    parser.add_argument("--report_path",default="analysis/report.json")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load data ─────────────────────────────────────────────
    if args.hf_dataset:
        data, labels = load_hf_dataset(args.hf_dataset)
    else:
        data, labels = load_raw_data(args.data, args.modality)

    # ── Detect modality ───────────────────────────────────────
    modality = args.modality
    if modality == "auto":
        modality = detect_modality(tuple(data.shape))
        log.info(f"Auto-detected modality: {modality}")
        log.info(f"Shape: {shape_summary(tuple(data.shape), modality)}")

    # ── Build dataset ─────────────────────────────────────────
    seq_len = args.seq_len
    ds = UniversalDataset(data, modality=modality, seq_len=seq_len, labels=labels)

    n_train = int(0.9 * len(ds))
    n_val   = len(ds) - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    # ── Auto-config ───────────────────────────────────────────
    cfg = auto_config(ds.input_dim, len(ds), modality, args.gpu_gb)

    # Apply CLI overrides
    if args.d_model:    cfg["model"]["d_model"]    = args.d_model
    if args.n_levels:   cfg["model"]["n_levels"]   = args.n_levels
    if args.n_experts:  cfg["model"]["n_experts"]  = args.n_experts
    if args.top_k:      cfg["model"]["top_k"]      = args.top_k
    if args.max_steps:  cfg["training"]["max_steps"] = args.max_steps
    if args.lr:         cfg["training"]["lr"]      = args.lr
    if args.batch_size: cfg["data"]["batch_size"]  = args.batch_size
    cfg["model"]["loss_type"] = args.loss_type
    cfg["model"]["seq_len"]   = seq_len

    # ── Build model ───────────────────────────────────────────
    mc = cfg["model"]
    model = UniversalJEPA(
        input_dim   = ds.input_dim,
        d_model     = mc["d_model"],
        seq_len     = mc["seq_len"],
        n_levels    = mc["n_levels"],
        pool_factor = mc["pool_factor"],
        n_experts   = mc["n_experts"],
        top_k       = mc["top_k"],
        loss_type   = mc["loss_type"],
    ).to(device)

    # Move EMA targets to device
    for t in model.target_encoders:
        for p in t.parameters():
            p.data = p.data.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {n_params:,} trainable parameters")
    log.info(f"  input_dim={ds.input_dim}  d_model={mc['d_model']}  "
              f"n_levels={mc['n_levels']}  n_experts={mc['n_experts']}")

    # ── Resume ────────────────────────────────────────────────
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        log.info(f"Resumed from {args.resume} (step {ckpt.get('step', '?')})")

    # ── DataLoaders ───────────────────────────────────────────
    bs = cfg["data"]["batch_size"]
    train_ds.dataset.training_mode = True
    val_ds.dataset.training_mode   = False

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                               num_workers=2, pin_memory=device.startswith("cuda"),
                               drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                               num_workers=2, drop_last=False)

    # ── Train or probe ────────────────────────────────────────
    if args.probe_only:
        log.info("Probe-only mode: running analysis on current weights...")
        analyzer = JEPAAnalyzer(model, device)
        analyzer.full_report(val_loader, save_path=args.report_path)
    else:
        train(model, train_loader, val_loader, cfg, device)

        # Final analysis after training
        log.info("\nPost-training analysis:")
        analyzer = JEPAAnalyzer(model, device)
        analyzer.full_report(val_loader, save_path=args.report_path)


if __name__ == "__main__":
    main()
