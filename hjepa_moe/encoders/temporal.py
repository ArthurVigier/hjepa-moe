"""
Temporal Encoder for H-JEPA.

Each level ℓ has a TemporalEncoder that:
  1. Takes a window of k_ℓ embeddings from level ℓ-1
  2. Pools them into a single representation at level ℓ
  3. Optionally increases feature dimension d_ℓ > d_{ℓ-1}

Three pooling strategies (matching FAIR/AMI style):
  - 'attention': Learnable Q from learned query, K/V from frames
                 Most expressive, used for higher levels.
  - 'mean':      Simple temporal mean pooling — fast, minimal.
  - 'conv':      1D causal convolution — good for motion patterns.

Level 0 encoder is special: maps raw pixel patches (ViT tokens)
to level-0 embeddings. Here we support plug-in with frozen
V-JEPA 2 ViT-L (FAIR-style) or a small ConvNet for experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AttentivePooling(nn.Module):
    """
    Attentive temporal pooling.
    Uses a learnable query to attend over k_ℓ input frames.
    Output: single vector summarizing the temporal window.
    
    This mirrors the "attentive probe" pattern from V-JEPA 2.
    """
    
    def __init__(self, d_in: int, d_out: int, n_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_in) * 0.02)
        self.attn  = nn.MultiheadAttention(d_in, n_heads, batch_first=True)
        self.norm  = nn.RMSNorm(d_in)
        self.proj  = nn.Linear(d_in, d_out, bias=False) if d_in != d_out else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, k, d_in) — k frames
        B = x.size(0)
        q = self.query.expand(B, -1, -1)   # (B, 1, d_in)
        pooled, _ = self.attn(q, x, x)     # (B, 1, d_in)
        pooled = self.norm(pooled.squeeze(1))  # (B, d_in)
        return self.proj(pooled)             # (B, d_out)


class ConvTemporalPool(nn.Module):
    """
    1D causal convolution over temporal dimension.
    Good for capturing short-term motion patterns.
    Kernel size = pool_factor, stride = pool_factor.
    """
    
    def __init__(self, d_in: int, d_out: int, pool_factor: int):
        super().__init__()
        self.conv = nn.Conv1d(
            d_in, d_out,
            kernel_size=pool_factor,
            stride=pool_factor,
            bias=False
        )
        self.norm = nn.RMSNorm(d_out)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, k, d_in) -> (B, d_out)
        x = x.transpose(1, 2)          # (B, d_in, k)
        x = self.conv(x)               # (B, d_out, 1) if stride=k
        x = x.squeeze(-1)             # (B, d_out)
        return self.norm(x)


class TemporalEncoder(nn.Module):
    """
    Temporal Encoder for level ℓ of H-JEPA.
    
    Args:
        d_in:          embedding dim at level ℓ-1
        d_out:         embedding dim at level ℓ  (typically d_in * 2)
        pool_factor:   how many level-(ℓ-1) steps to pool  (e.g. 4 or 8)
        pooling:       'attention', 'mean', or 'conv'
        n_heads:       attention heads (only for 'attention' pooling)
    
    Usage in hierarchy:
        Level 0 -> 1: pool_factor=4,  d_in=256, d_out=512
        Level 1 -> 2: pool_factor=8,  d_in=512, d_out=768
        Level 2 -> 3: pool_factor=8,  d_in=768, d_out=768
    """
    
    def __init__(
        self,
        d_in:        int,
        d_out:       int,
        pool_factor: int,
        pooling:     str = "attention",
        n_heads:     int = 4,
    ):
        super().__init__()
        assert pooling in ("attention", "mean", "conv")
        self.pool_factor = pool_factor
        self.pooling     = pooling
        self.d_in  = d_in
        self.d_out = d_out
        
        if pooling == "attention":
            self.pool = AttentivePooling(d_in, d_out, n_heads)
        elif pooling == "conv":
            self.pool = ConvTemporalPool(d_in, d_out, pool_factor)
        else:  # mean
            self.proj = nn.Sequential(
                nn.Linear(d_in, d_out, bias=False),
                nn.RMSNorm(d_out),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_in) — sequence of level-(ℓ-1) states
               T must be divisible by pool_factor
        Returns:
            s: (B, T // pool_factor, d_out)
        """
        B, T, d = x.shape
        assert T % self.pool_factor == 0, \
            f"T={T} must be divisible by pool_factor={self.pool_factor}"
        
        n_windows = T // self.pool_factor
        
        if self.pooling == "mean":
            # Simple mean pool then project
            x_reshaped = x.reshape(B, n_windows, self.pool_factor, d)
            pooled = x_reshaped.mean(dim=2)   # (B, n_windows, d)
            return self.proj(pooled)
        
        elif self.pooling == "conv":
            # Conv1D operates on full sequence
            out = self.pool(x)  # returns (B, d_out) — single window
            # For multi-window: apply per window
            windows = x.reshape(B * n_windows, self.pool_factor, d)
            out = self.pool(windows).reshape(B, n_windows, self.d_out)
            return out
        
        else:  # attention
            windows = x.reshape(B * n_windows, self.pool_factor, d)
            pooled = self.pool(windows)  # (B * n_windows, d_out)
            return pooled.reshape(B, n_windows, self.d_out)


class Level0Encoder(nn.Module):
    """
    Level 0 encoder: raw observations -> patch embeddings.
    
    Two modes:
      - 'vjepa2': Wraps a frozen V-JEPA 2 ViT-L (FAIR style, 304M params)
                  Best for serious experiments.
      - 'small':  Lightweight ConvNet for quick ablations on CIFAR/MovingMNIST.
                  Matches EB-JEPA's ResNet-18 / IMPALA encoders.
    
    For 'vjepa2' mode, pass in the pretrained model:
        enc0 = Level0Encoder('vjepa2', vjepa2_model=loaded_vjepa2)
    """
    
    def __init__(
        self,
        mode:         str = "small",
        d_out:        int = 256,
        img_size:     int = 64,
        patch_size:   int = 8,
        vjepa2_model: Optional[nn.Module] = None,
    ):
        super().__init__()
        assert mode in ("vjepa2", "small")
        self.mode = mode
        
        if mode == "vjepa2":
            assert vjepa2_model is not None, "Pass vjepa2_model for vjepa2 mode"
            self.encoder = vjepa2_model
            for p in self.encoder.parameters():
                p.requires_grad_(False)  # Always frozen
            # V-JEPA 2 ViT-L output dim is 1024
            self.proj = nn.Sequential(
                nn.Linear(1024, d_out, bias=False),
                nn.RMSNorm(d_out),
            )
        
        else:  # small ConvNet for ablations
            n_patches = (img_size // patch_size) ** 2
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, patch_size, stride=patch_size, bias=False),
                nn.ReLU(),
                nn.Conv2d(64, d_out, 1, bias=False),
            )
            self.pos_embed = nn.Parameter(
                torch.randn(1, n_patches, d_out) * 0.02
            )
            self.norm = nn.RMSNorm(d_out)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image or (B, T, C, H, W) video
        Returns:
            tokens: (B, N_patches, d_out)
        """
        if self.mode == "vjepa2":
            with torch.no_grad():
                tokens = self.encoder(x)    # frozen ViT-L
            return self.proj(tokens)
        
        else:
            is_video = x.dim() == 5
            if is_video:
                B, T, C, H, W = x.shape
                x = x.reshape(B * T, C, H, W)
            
            patches = self.encoder(x)              # (B, d_out, H', W')
            B_  = patches.shape[0]
            tokens = patches.flatten(2).transpose(1, 2)  # (B, N, d_out)
            tokens = self.norm(tokens + self.pos_embed)
            
            if is_video:
                tokens = tokens.reshape(B, T, -1, self.norm.normalized_shape[0])
                tokens = tokens.mean(dim=2)   # pool spatial -> (B, T, d_out)
            
            return tokens
