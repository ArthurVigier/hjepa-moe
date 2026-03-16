"""Utilities: LR scheduling, attentive probe, logging."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ── LR Scheduling ──────────────────────────────────────────────

def cosine_schedule(step: int, max_steps: int, warmup: int,
                    lr_max: float, lr_min: float = 1e-6) -> float:
    """Cosine decay with linear warmup (FAIR standard)."""
    if step < warmup:
        return lr_max * max(step, 1) / max(warmup, 1)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ── Attentive Probe ────────────────────────────────────────────

class AttentiveProbe(nn.Module):
    """
    Lightweight attentive probe for downstream linear evaluation.
    Follows V-JEPA 2 / EB-JEPA probe style:
      - Single learned query attending over the sequence
      - Linear classifier on top of pooled features
    Frozen backbone, only probe parameters trained.
    """
    def __init__(self, d_in: int, n_classes: int, n_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_in) * 0.02)
        self.attn  = nn.MultiheadAttention(d_in, n_heads, batch_first=True)
        self.norm  = nn.LayerNorm(d_in)
        self.head  = nn.Linear(d_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_in) or (B, d_in)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        pooled, _ = self.attn(q, x, x)
        pooled = self.norm(pooled.squeeze(1))
        return self.head(pooled)


def train_probe(
    model,
    probe: AttentiveProbe,
    loader: torch.utils.data.DataLoader,
    level: int,
    n_epochs: int = 20,
    device: str = "cpu",
) -> float:
    """Train probe on frozen level-{level} features. Returns top-1 accuracy."""
    probe = probe.to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    model.eval()

    for epoch in range(n_epochs):
        correct = total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                states = model.get_level_states(x)
                feats  = states[level + 1]          # +1 because index 0 = raw frames
            logits = probe(feats)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            correct += (logits.argmax(1) == y).sum().item()
            total   += y.size(0)

    return correct / total if total > 0 else 0.0


# ── Routing stats helpers ──────────────────────────────────────

def routing_entropy(usage: "np.ndarray") -> float:
    """Shannon entropy of expert usage distribution. Max = log(N)."""
    import numpy as np
    p = usage / (usage.sum() + 1e-8)
    p = np.clip(p, 1e-8, 1)
    return float(-(p * np.log(p)).sum())


# ── AverageMeter ──────────────────────────────────────────────

class AverageMeter:
    """Tracks running averages of arbitrary scalar metrics."""
    def __init__(self):
        self._sums  = {}
        self._counts = {}

    def update(self, d: dict):
        for k, v in d.items():
            self._sums[k]   = self._sums.get(k, 0.0)   + float(v)
            self._counts[k] = self._counts.get(k, 0)    + 1

    def avg(self) -> dict:
        return {k: self._sums[k] / self._counts[k]
                for k in self._sums if self._counts[k] > 0}

    def reset(self):
        self._sums.clear(); self._counts.clear()
