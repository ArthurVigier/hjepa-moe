"""
Regularization losses for H-JEPA training.

Two options (both from FAIR/AMI Labs lineage):

1. VICReg (Bardes et al., 2022)
   - Variance: pushes std > threshold per feature (prevents collapse)
   - Invariance: MSE between prediction and target (alignment)
   - Covariance: decorrelates features (promotes diversity)
   Used in EB-JEPA examples.

2. SIGReg (Balestriero & LeCun, 2025) — from LeJEPA
   - Sketched Isotropic Gaussian Regularization
   - Constrains embeddings to follow isotropic Gaussian
   - Single hyperparameter, O(d) memory, no matrix decompositions
   - Preferred for large-scale runs (AMI Labs style)

Both also support InfoNCE (contrastive) as used in VL-JEPA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class VICRegLoss(nn.Module):
    """
    VICReg loss.
    
    L = λ * invariance + μ * variance + ν * covariance
    
    where:
        invariance = MSE(z1, z2)
        variance   = mean(max(0, γ - std_j(z)) for j in features)
        covariance = (1/d) * sum_{i≠j} [cov(z)_ij]^2
    
    Args:
        sim_coef:  λ — invariance (prediction) weight
        var_coef:  μ — variance weight
        cov_coef:  ν — covariance weight
        gamma:     std target (typically 1.0)
    """
    
    def __init__(
        self,
        sim_coef: float = 25.0,
        var_coef: float = 25.0,
        cov_coef: float = 1.0,
        gamma:    float = 1.0,
    ):
        super().__init__()
        self.sim_coef = sim_coef
        self.var_coef = var_coef
        self.cov_coef = cov_coef
        self.gamma    = gamma
    
    def forward(
        self,
        z_pred: torch.Tensor,
        z_tgt:  torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            z_pred: predicted embeddings (B, d)
            z_tgt:  target embeddings    (B, d)
        Returns:
            loss:  scalar total loss
            stats: dict of individual components
        """
        # Invariance: MSE between prediction and target
        inv_loss = F.mse_loss(z_pred, z_tgt)
        
        # Variance: computed on both predicted and target
        var_loss = (
            self._variance_loss(z_pred) +
            self._variance_loss(z_tgt)
        ) / 2
        
        # Covariance: computed on predicted only
        cov_loss = self._covariance_loss(z_pred)
        
        total = (
            self.sim_coef * inv_loss +
            self.var_coef * var_loss +
            self.cov_coef * cov_loss
        )
        
        return total, {
            "loss_total":    total.item(),
            "loss_inv":      inv_loss.item(),
            "loss_var":      var_loss.item(),
            "loss_cov":      cov_loss.item(),
        }
    
    def _variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        z = z - z.mean(dim=0)
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        return F.relu(self.gamma - std).mean()
    
    def _covariance_loss(self, z: torch.Tensor) -> torch.Tensor:
        B, d = z.shape
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (B - 1)   # (d, d)
        off_diag = cov.pow(2)
        off_diag.fill_diagonal_(0)
        return off_diag.sum() / d


class SIGRegLoss(nn.Module):
    """
    SIGReg — Sketched Isotropic Gaussian Regularization.
    From: Balestriero & LeCun (2025) LeJEPA.
    
    Objective: push embedding distribution toward N(0, I)
    
    Method: Random sketch (Johnson-Lindenstrauss projection)
    to estimate 1st and 2nd moments, then penalize deviations
    from mean=0 and variance=1.
    
    Advantages over VICReg:
      - O(d) memory (no d×d covariance matrix)
      - Single hyperparameter alpha
      - Works well with any architecture (ResNet, ViT, ConvNet)
    
    Args:
        d_model:     embedding dimension
        n_sketches:  number of random projections (default: min(512, d))
        alpha:       regularization strength
        sim_coef:    prediction loss weight
    """
    
    def __init__(
        self,
        d_model:    int,
        n_sketches: int = None,
        alpha:      float = 1.0,
        sim_coef:   float = 1.0,
    ):
        super().__init__()
        self.n_sketches = n_sketches or min(512, d_model)
        self.alpha      = alpha
        self.sim_coef   = sim_coef
        
        # Fixed random projection matrix (not learned)
        sketch = torch.randn(d_model, self.n_sketches) / (d_model ** 0.5)
        self.register_buffer("sketch", sketch)
    
    def forward(
        self,
        z_pred: torch.Tensor,
        z_tgt:  torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            z_pred: (B, d) predicted
            z_tgt:  (B, d) target
        Returns:
            loss, stats
        """
        # Prediction loss (cosine similarity, JEPA-style)
        z_pred_n = F.normalize(z_pred, dim=-1)
        z_tgt_n  = F.normalize(z_tgt, dim=-1)
        sim_loss = (2 - 2 * (z_pred_n * z_tgt_n).sum(dim=-1)).mean()
        
        # SIGReg: sketch and match to N(0,1)
        reg_loss = (
            self._sigreg(z_pred) +
            self._sigreg(z_tgt)
        ) / 2
        
        total = self.sim_coef * sim_loss + self.alpha * reg_loss
        
        return total, {
            "loss_total":  total.item(),
            "loss_sim":    sim_loss.item(),
            "loss_sigreg": reg_loss.item(),
        }
    
    def _sigreg(self, z: torch.Tensor) -> torch.Tensor:
        # Project to sketch space
        z_sk = z @ self.sketch    # (B, n_sketches)
        
        # Penalty on mean deviation from 0
        mean_pen = z_sk.mean(dim=0).pow(2).mean()
        
        # Penalty on std deviation from 1
        std_pen = (z_sk.std(dim=0) - 1).pow(2).mean()
        
        return mean_pen + std_pen


class InfoNCELoss(nn.Module):
    """
    InfoNCE / CLIP-style contrastive loss.
    Used in VL-JEPA and M3-JEPA.
    
    Mathematically = alignment + uniformity (Wang & Isola 2020):
        L = -E[sim(z_pred, z_tgt)] + E[log sum_j exp(sim(z_pred, z_j))]
    
    Args:
        temperature: softmax temperature (default 0.07 like CLIP)
        bidirectional: compute loss in both directions
    """
    
    def __init__(self, temperature: float = 0.07, bidirectional: bool = True):
        super().__init__()
        self.temp = temperature
        self.bidirectional = bidirectional
    
    def forward(
        self,
        z_pred: torch.Tensor,
        z_tgt:  torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        B = z_pred.shape[0]
        z_pred_n = F.normalize(z_pred, dim=-1)
        z_tgt_n  = F.normalize(z_tgt, dim=-1)
        
        logits = (z_pred_n @ z_tgt_n.T) / self.temp   # (B, B)
        labels = torch.arange(B, device=z_pred.device)
        
        loss_pt = F.cross_entropy(logits, labels)
        if self.bidirectional:
            loss_tp = F.cross_entropy(logits.T, labels)
            total = (loss_pt + loss_tp) / 2
        else:
            total = loss_pt
        
        # Accuracy (top-1)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        
        return total, {
            "loss_total": total.item(),
            "acc_top1":   acc.item(),
        }
