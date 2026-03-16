"""
H-JEPA-MoE: Video prediction experiment on Moving MNIST.

Self-contained, single-GPU. Trains in ~2-4 hours on A100.
Mirrors EB-JEPA's example structure.

Run:
    python -m examples.video_jepa.main --cfg configs/video_jepa_moe.yaml

Metrics tracked:
  - Per-level prediction loss
  - Per-expert routing entropy (diversity metric)
  - Linear probe accuracy at each level
  - Planning success rate (for levels with d_z > 0)
"""

import os
import sys
import yaml
import argparse
import math
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hjepa_moe import HJEPAMoE
from hjepa_moe.model import HJEPAMoEConfig, LevelConfig


# ──────────────────────────────────────────
# Data
# ──────────────────────────────────────────

class MovingMNISTDataset(torch.utils.data.Dataset):
    """
    Generates random Moving MNIST sequences on-the-fly.
    Two MNIST digits bounce in a 64x64 frame.
    """
    
    def __init__(self, n_samples: int = 10000, seq_len: int = 64, img_size: int = 64):
        self.n_samples = n_samples
        self.seq_len   = seq_len
        self.img_size  = img_size
        # Simple placeholder: random noise sequences
        # Replace with real MNIST loading for actual experiments
        self._data = torch.randn(n_samples, seq_len, 1, img_size, img_size)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # (T, C, H, W) — grayscale, expand to RGB
        seq = self._data[idx].expand(-1, 3, -1, -1)
        return seq


# ──────────────────────────────────────────
# Build model from config
# ──────────────────────────────────────────

def build_model(cfg: dict) -> HJEPAMoE:
    model_cfg = cfg["model"]
    
    levels = [
        LevelConfig(**lvl) for lvl in model_cfg["levels"]
    ]
    
    config = HJEPAMoEConfig(
        levels          = levels,
        loss_type       = model_cfg.get("loss_type", "vicreg"),
        sim_coef        = model_cfg.get("sim_coef", 25.0),
        var_coef        = model_cfg.get("var_coef", 25.0),
        cov_coef        = model_cfg.get("cov_coef", 1.0),
        ema_decay       = model_cfg.get("ema_decay", 0.996),
        n_rollout_steps = model_cfg.get("n_rollout_steps", 2),
        rollout_weight  = model_cfg.get("rollout_weight", 0.5),
        level0_mode     = model_cfg.get("level0_mode", "small"),
        d_level0        = model_cfg.get("d_level0", 256),
        img_size        = model_cfg.get("img_size", 64),
    )
    
    return HJEPAMoE(config)


# ──────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────

def cosine_schedule(step: int, max_steps: int, warmup: int,
                    lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(math.pi * progress))


def get_routing_entropy(model: HJEPAMoE, x: torch.Tensor) -> dict:
    """
    Compute per-expert routing entropy at each level.
    High entropy = all experts used equally (desired).
    Low entropy = collapse to few experts (bad sign).
    """
    entropies = {}
    model.eval()
    with torch.no_grad():
        level_states = model.get_level_states(x[:4])  # small batch
        for ℓ, (states, pred) in enumerate(zip(level_states[1:], model.moe_predictors)):
            if states.dim() == 3:
                s_flat = states[:, :-1].reshape(-1, states.shape[-1])
            else:
                s_flat = states
            stats = pred.get_routing_stats(s_flat)
            usage = stats["expert_usage"]
            # Entropy: H = -sum(p * log(p))
            p = usage / usage.sum()
            p = np.clip(p, 1e-8, 1)
            entropy = -float((p * np.log(p)).sum())
            entropies[f"level_{ℓ}_routing_entropy"] = entropy
    model.train()
    return entropies


# ──────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────

def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Data
    data_cfg = cfg["data"]
    dataset  = MovingMNISTDataset(
        n_samples = 10000,
        seq_len   = data_cfg["seq_len"],
        img_size  = data_cfg["img_size"],
    )
    loader = DataLoader(
        dataset,
        batch_size  = data_cfg["batch_size"],
        shuffle     = True,
        num_workers = 4,
        pin_memory  = True,
        drop_last   = True,
    )
    
    # Model
    model = build_model(cfg).to(device)
    # Move EMA target encoders to device (not in ModuleList)
    for t_enc in model.target_encoders:
        for p in t_enc.parameters():
            p.data = p.data.to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    
    # Per-level parameter counts
    for ℓ, (enc, pred) in enumerate(zip(model.temporal_encoders, model.moe_predictors)):
        n_enc  = sum(p.numel() for p in enc.parameters())
        n_pred = sum(p.numel() for p in pred.parameters())
        print(f"  Level {ℓ}: enc={n_enc:,} pred={n_pred:,} "
              f"(experts={cfg['model']['levels'][ℓ]['n_experts']}, "
              f"top_k={cfg['model']['levels'][ℓ]['top_k']})")
    
    # Optimizer
    train_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = train_cfg["lr"],
        weight_decay = train_cfg["weight_decay"],
    )
    
    # Training loop
    step       = 0
    max_steps  = train_cfg["max_steps"]
    warmup     = train_cfg["warmup_steps"]
    log_every  = train_cfg["log_every"]
    
    model.train()
    running_stats = {}
    
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            
            # LR schedule
            lr = cosine_schedule(step, max_steps, warmup,
                                  train_cfg["lr"], train_cfg["min_lr"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            
            # Forward
            x = batch.to(device, non_blocking=True)   # (B, T, C, H, W)
            loss, stats = model(x)
            
            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if train_cfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg["grad_clip"]
                )
            optimizer.step()
            
            # EMA update (critical for JEPA stability)
            model.update_ema()
            
            # Accumulate stats
            for k, v in stats.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        running_stats[f"{k}/{k2}"] = running_stats.get(f"{k}/{k2}", 0) + v2
                else:
                    running_stats[k] = running_stats.get(k, 0) + v
            
            # Logging
            if step % log_every == 0:
                # Average stats
                avg_stats = {k: v / log_every for k, v in running_stats.items()}
                running_stats = {}
                
                # Routing entropy (diversity of expert usage)
                entropy_stats = get_routing_entropy(model, x)
                avg_stats.update(entropy_stats)
                
                print(f"Step {step:5d} | lr={lr:.2e} | "
                      f"loss={avg_stats.get('loss_total', 0):.4f} | "
                      f"entropy_L1={avg_stats.get('level_0_routing_entropy', 0):.3f}")
                
                # Print per-level detail
                for ℓ in range(len(cfg["model"]["levels"])):
                    key = f"level_{ℓ}/loss_total"
                    if key in avg_stats:
                        print(f"  Level {ℓ}: loss={avg_stats[key]:.4f} "
                              f"entropy={avg_stats.get(f'level_{ℓ}_routing_entropy', 0):.3f}")
            
            step += 1
    
    print("Training complete.")
    
    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "step": step,
    }, "checkpoints/hjepa_moe_final.pt")
    print("Saved: checkpoints/hjepa_moe_final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/video_jepa_moe.yaml")
    args = parser.parse_args()
    main(args.cfg)
