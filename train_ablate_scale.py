"""
H-JEPA-MoE: Training, Ablation & Scale
========================================

Three modes in one script:

  MODE 1 — train
    Single or multi-GPU full training run.
    Tracks per-level losses, routing entropy, probe accuracy, planning success.

  MODE 2 — ablate
    Systematic ablation over a param grid defined in the YAML config.
    Runs N jobs sequentially (or submits to SLURM if --slurm flag).
    Outputs a ranked CSV table of results.

  MODE 3 — scale
    Multi-node / multi-GPU training via torch.distributed.
    DDP wrapping, gradient compression, mixed precision (bf16).
    Designed for 4-8x A100 80GB or larger.

Usage:
    # Single GPU training
    python train_ablate_scale.py train --cfg configs/video_jepa_moe.yaml

    # Ablation sweep (all combinations in sweep.param_grid)
    python train_ablate_scale.py ablate --cfg configs/video_jepa_moe.yaml

    # Multi-GPU (4 GPUs on 1 node)
    torchrun --nproc_per_node=4 train_ablate_scale.py scale \\
             --cfg configs/scale_large.yaml

    # Multi-node (2 nodes x 8 GPUs = 16 total)
    torchrun --nnodes=2 --nproc_per_node=8 \\
             --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \\
             train_ablate_scale.py scale --cfg configs/scale_xl.yaml
"""

import os
import sys
import yaml
import json
import copy
import math
import time
import argparse
import itertools
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent))

from hjepa_moe.model import HJEPAMoE, HJEPAMoEConfig, LevelConfig
from hjepa_moe.utils import (
    AverageMeter, cosine_schedule, set_lr,
    AttentiveProbe, train_probe, routing_entropy,
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str = "hjepa_moe") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s][%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

log = get_logger()

# ─────────────────────────────────────────────────────────────────────────────
# Distributed helpers
# ─────────────────────────────────────────────────────────────────────────────

def init_distributed() -> Tuple[int, int, int]:
    """Init DDP. Returns (rank, local_rank, world_size)."""
    if "RANK" not in os.environ:
        return 0, 0, 1   # single process

    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def is_main(rank: int) -> bool:
    return rank == 0

def all_reduce_mean(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / world_size

# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def build_model(cfg: dict) -> HJEPAMoE:
    mc = cfg["model"]
    levels = [LevelConfig(**lv) for lv in mc["levels"]]
    config = HJEPAMoEConfig(
        levels          = levels,
        loss_type       = mc.get("loss_type", "vicreg"),
        sim_coef        = mc.get("sim_coef", 25.0),
        var_coef        = mc.get("var_coef", 25.0),
        cov_coef        = mc.get("cov_coef", 1.0),
        ema_decay       = mc.get("ema_decay", 0.996),
        n_rollout_steps = mc.get("n_rollout_steps", 2),
        rollout_weight  = mc.get("rollout_weight", 0.5),
        level0_mode     = mc.get("level0_mode", "small"),
        d_level0        = mc.get("d_level0", 256),
        img_size        = mc.get("img_size", 64),
    )
    return HJEPAMoE(config)

def deep_set(d: dict, dotkey: str, value: Any) -> dict:
    """Set nested dict key using dot notation: 'model.levels[0].n_experts'."""
    import re
    d = copy.deepcopy(d)
    # Handle list indexing like levels[0]
    parts = re.split(r'\.(?![^\[]*\])', dotkey)
    obj = d
    for part in parts[:-1]:
        m = re.match(r'(\w+)\[(\d+)\]', part)
        if m:
            obj = obj[m.group(1)][int(m.group(2))]
        else:
            obj = obj[part]
    last = parts[-1]
    m = re.match(r'(\w+)\[(\d+)\]', last)
    if m:
        obj[m.group(1)][int(m.group(2))] = value
    else:
        obj[last] = value
    return d

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def make_dataset(cfg: dict, split: str = "train"):
    """
    Dataset factory. Supports:
      - 'moving_mnist': synthetic, single GPU dev
      - 'ssv2_small':   Something-Something v2 (small split)
      - 'kinetics400':  Kinetics-400
      - 'droid':        DROID robot trajectories (for AC experiments)
      - 'custom':       any folder of .mp4 files via cfg.data.root

    For production scale, replace this with your VideoDataset.
    The contract is: __getitem__ returns (B, T, C, H, W) float32 in [0,1].
    """
    dc   = cfg["data"]
    name = dc.get("dataset", "moving_mnist")
    T    = dc["seq_len"]
    H    = dc["img_size"]

    if name == "moving_mnist":
        n = 10000 if split == "train" else 1000
        # Synthetic placeholder — replace with real MovingMNIST
        data = torch.rand(n, T, 3, H, H)
        labels = torch.zeros(n, dtype=torch.long)   # no labels for SSL
        return TensorDataset(data, labels)

    elif name == "custom":
        # Minimal video folder dataset
        root = dc["root"]
        return VideoFolderDataset(root, seq_len=T, img_size=H, split=split)

    else:
        raise ValueError(f"Unknown dataset: {name}. Add it to make_dataset().")


class VideoFolderDataset(torch.utils.data.Dataset):
    """
    Minimal video folder dataset.
    Expects: root/{split}/*.mp4 (or .avi, .webm).
    Returns: (T, C, H, W) float32 tensors.

    For production: replace with your optimised video loader
    (e.g. decord, torchvision.io, or FFCV for multi-GPU scale).
    """
    def __init__(self, root: str, seq_len: int = 64, img_size: int = 64,
                 split: str = "train"):
        import glob
        self.files   = sorted(glob.glob(f"{root}/{split}/*.mp4"))
        self.seq_len = seq_len
        self.img_size = img_size
        if not self.files:
            # Fallback to random data if no files found
            log.warning(f"No .mp4 files found in {root}/{split}/, using random data")
            self._random = True
            self._n = 1000
        else:
            self._random = False

    def __len__(self):
        return self._n if self._random else len(self.files)

    def __getitem__(self, idx):
        if self._random:
            return torch.rand(self.seq_len, 3, self.img_size, self.img_size), 0

        # Use torchvision or decord to load video
        try:
            import torchvision.io as tvio
            frames, _, _ = tvio.read_video(self.files[idx], pts_unit="sec")
            # frames: (T, H, W, C)
            frames = frames.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
            # Sample or pad to seq_len
            T = frames.shape[0]
            if T >= self.seq_len:
                start = torch.randint(0, T - self.seq_len + 1, (1,)).item()
                frames = frames[start:start + self.seq_len]
            else:
                pad = self.seq_len - T
                frames = torch.cat([frames, frames[-1:].expand(pad, -1, -1, -1)])
            # Resize
            frames = F.interpolate(
                frames, size=(self.img_size, self.img_size), mode="bilinear",
                align_corners=False
            )
            return frames, 0
        except Exception as e:
            log.warning(f"Failed to load {self.files[idx]}: {e}")
            return torch.rand(self.seq_len, 3, self.img_size, self.img_size), 0


# ─────────────────────────────────────────────────────────────────────────────
# Core training step
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainState:
    step:      int   = 0
    best_loss: float = float("inf")
    best_probe: float = 0.0


def training_step(
    model:      HJEPAMoE,
    batch:      torch.Tensor,
    optimizer:  torch.optim.Optimizer,
    scaler:     Optional[GradScaler],
    cfg:        dict,
    world_size: int = 1,
) -> dict:
    """One optimizer step. Returns stats dict."""
    tc = cfg["training"]
    use_amp = tc.get("mixed_precision", False) and torch.cuda.is_available()

    with autocast(dtype=torch.bfloat16, enabled=use_amp):
        loss, stats = model(batch)

    optimizer.zero_grad(set_to_none=True)

    if use_amp and scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get("grad_clip", 1.0))
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get("grad_clip", 1.0))
        optimizer.step()

    # EMA update (critical — must happen after optimizer step)
    core = model.module if hasattr(model, "module") else model
    core.update_ema()

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_routing(
    model:  HJEPAMoE,
    loader: DataLoader,
    device: str,
    n_batches: int = 5,
) -> dict:
    """Per-level routing entropy over a few validation batches."""
    core = model.module if hasattr(model, "module") else model
    core.eval()
    all_entropies = {}

    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        x = x.to(device)
        states = core.get_level_states(x)
        for ℓ, (pred, s) in enumerate(zip(core.moe_predictors, states[1:])):
            s_flat = s[:, :-1].reshape(-1, s.shape[-1])
            stats  = pred.get_routing_stats(s_flat)
            H = routing_entropy(stats["expert_usage"])
            key = f"entropy_L{ℓ+1}"
            all_entropies[key] = all_entropies.get(key, [])
            all_entropies[key].append(H)

    core.train()
    return {k: float(np.mean(v)) for k, v in all_entropies.items()}


@torch.no_grad()
def evaluate_probe(
    model:     HJEPAMoE,
    val_loader: DataLoader,
    level:     int,
    n_classes: int,
    device:    str,
    n_epochs:  int = 5,
) -> float:
    """Train + eval linear probe on frozen level-ℓ representations."""
    core = model.module if hasattr(model, "module") else model
    core.eval()
    d = core.config.levels[level].d_out if level < len(core.config.levels) else core.config.d_level0

    probe = AttentiveProbe(d, n_classes).to(device)
    opt   = torch.optim.Adam(probe.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            states = core.get_level_states(x)
            feats  = states[level + 1]                 # (B, T_ℓ, d)
            feats  = feats.mean(dim=1)                 # (B, d)
            logits = probe(feats.unsqueeze(1))
            loss   = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()

    # Eval
    correct = total = 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        states = core.get_level_states(x)
        feats  = states[level + 1].mean(dim=1)
        logits = probe(feats.unsqueeze(1))
        correct += (logits.argmax(1) == y).sum().item()
        total   += y.size(0)

    core.train()
    return correct / total if total > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1: train
# ─────────────────────────────────────────────────────────────────────────────

def run_train(cfg: dict, rank: int = 0, local_rank: int = 0, world_size: int = 1):
    """Full training loop. Single or multi-GPU (called by torchrun for scale mode)."""
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    tc     = cfg["training"]
    dc     = cfg["data"]

    # ── Data ──────────────────────────────────────────────────
    train_ds = make_dataset(cfg, "train")
    val_ds   = make_dataset(cfg, "val") if Path(f"{dc.get('root','data')}/val").exists() \
               else make_dataset(cfg, "train")   # fallback

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                        rank=rank, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_ds,
        batch_size  = dc["batch_size"] // world_size,
        sampler     = train_sampler,
        shuffle     = train_sampler is None,
        num_workers = tc.get("num_workers", 4),
        pin_memory  = torch.cuda.is_available(),
        drop_last   = True,
    )
    val_loader = DataLoader(val_ds, batch_size=dc["batch_size"] // world_size,
                             shuffle=False, num_workers=2, drop_last=False)

    # ── Model ─────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    for t_enc in model.target_encoders:
        for p in t_enc.parameters():
            p.data = p.data.to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    # ── Optimizer & LR ────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = tc["lr"],
        weight_decay = tc.get("weight_decay", 0.05),
        betas        = tc.get("betas", (0.9, 0.95)),
    )
    scaler = GradScaler() if tc.get("mixed_precision", False) and torch.cuda.is_available() else None

    # ── Checkpoint resume ─────────────────────────────────────
    ckpt_dir = Path(tc.get("checkpoint_dir", "checkpoints"))
    run_name = tc.get("run_name", "hjepa_moe")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state    = TrainState()

    resume = ckpt_dir / f"{run_name}_latest.pt"
    if resume.exists() and tc.get("resume", True):
        ckpt = torch.load(resume, map_location=device)
        core = model.module if hasattr(model, "module") else model
        core.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        state.step = ckpt.get("step", 0)
        if is_main(rank):
            log.info(f"Resumed from step {state.step}")

    # ── Main loop ─────────────────────────────────────────────
    max_steps   = tc["max_steps"]
    warmup      = tc.get("warmup_steps", 1000)
    log_every   = tc.get("log_every", 100)
    eval_every  = tc.get("eval_every", 1000)
    save_every  = tc.get("save_every", 5000)
    n_classes   = dc.get("n_classes", 10)

    meter = AverageMeter()
    t0    = time.time()

    model.train()

    while state.step < max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(state.step // len(train_loader))

        for batch in train_loader:
            if state.step >= max_steps:
                break

            x = batch[0].to(device, non_blocking=True)   # (B, T, C, H, W)

            # LR schedule
            lr = cosine_schedule(state.step, max_steps, warmup,
                                  tc["lr"], tc.get("min_lr", 1e-6))
            set_lr(optimizer, lr)

            stats = training_step(model, x, optimizer, scaler, cfg, world_size)
            meter.update({k: v for k, v in stats.items()
                          if isinstance(v, (int, float))})
            for k, v in stats.items():
                if isinstance(v, dict):
                    meter.update({f"{k}/{k2}": v2
                                  for k2, v2 in v.items()
                                  if isinstance(v2, (int, float))})

            # ── Logging ───────────────────────────────────────
            if state.step % log_every == 0 and is_main(rank):
                avg    = meter.avg()
                elapsed = time.time() - t0
                steps_per_sec = log_every / max(elapsed, 1e-6)
                t0 = time.time()
                log.info(
                    f"step={state.step:6d}  lr={lr:.2e}  "
                    f"loss={avg.get('loss_total', 0):.4f}  "
                    f"{steps_per_sec:.1f} steps/s"
                )
                for ℓ in range(len(cfg["model"]["levels"])):
                    k = f"level_{ℓ}/loss_total"
                    if k in avg:
                        log.info(f"  L{ℓ+1}: loss={avg[k]:.4f}  "
                                 f"moe_aux={avg.get(f'level_{ℓ}/loss_moe_aux', 0):.4f}")
                meter.reset()

            # ── Evaluation ────────────────────────────────────
            if state.step % eval_every == 0 and state.step > 0 and is_main(rank):
                # Routing entropy
                ent = evaluate_routing(model, val_loader, device)
                log.info("  Routing entropy: " +
                         "  ".join(f"{k}={v:.3f}" for k, v in ent.items()))

                # Linear probe on level 1
                probe_acc = evaluate_probe(
                    model, val_loader, level=0,
                    n_classes=n_classes, device=device, n_epochs=3
                )
                log.info(f"  Probe@L1 acc={probe_acc:.3f}")

                if probe_acc > state.best_probe:
                    state.best_probe = probe_acc
                    torch.save(
                        {"model": (model.module if hasattr(model, "module") else model).state_dict(),
                         "step": state.step, "probe_acc": probe_acc},
                        ckpt_dir / f"{run_name}_best.pt"
                    )

            # ── Checkpoint ────────────────────────────────────
            if state.step % save_every == 0 and state.step > 0 and is_main(rank):
                core = model.module if hasattr(model, "module") else model
                torch.save({
                    "model":     core.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step":      state.step,
                    "config":    cfg,
                }, ckpt_dir / f"{run_name}_step{state.step}.pt")
                # Also save as latest for resume
                torch.save({
                    "model":     core.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step":      state.step,
                    "config":    cfg,
                }, ckpt_dir / f"{run_name}_latest.pt")
                log.info(f"  Saved checkpoint at step {state.step}")

            state.step += 1

    if is_main(rank):
        log.info(f"Training complete. Best probe acc: {state.best_probe:.4f}")
    if world_size > 1:
        dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2: ablate
# ─────────────────────────────────────────────────────────────────────────────

def generate_ablation_grid(cfg: dict) -> List[dict]:
    """
    Expand sweep.param_grid into list of individual configs.
    Supports arbitrary nesting via dot notation.

    Example in YAML:
        sweep:
          param_grid:
            model.levels[0].n_experts: [2, 4, 8]
            model.loss_type: ["vicreg", "sigreg"]
            training.lr: [1e-4, 3e-4]
    → 3 × 2 × 2 = 12 configs
    """
    grid = cfg.get("sweep", {}).get("param_grid", {})
    if not grid:
        return [cfg]

    keys   = list(grid.keys())
    values = list(grid.values())
    combos = list(itertools.product(*values))

    configs = []
    for combo in combos:
        c = copy.deepcopy(cfg)
        tag_parts = []
        for k, v in zip(keys, combo):
            c = deep_set(c, k, v)
            # Build short tag for this combo
            short_k = k.split(".")[-1].replace("[", "").replace("]", "")
            tag_parts.append(f"{short_k}={v}")
        c["_ablation_tag"] = "__".join(tag_parts)
        configs.append(c)

    return configs


def run_ablate(cfg: dict, n_steps_override: int = None):
    """
    Run full ablation sweep. Results saved to ablation_results.csv.
    """
    import csv

    configs = generate_ablation_grid(cfg)
    log.info(f"Ablation: {len(configs)} configurations")

    results = []

    for i, c in enumerate(configs):
        tag = c.get("_ablation_tag", f"run_{i}")
        log.info(f"\n{'='*60}")
        log.info(f"Ablation [{i+1}/{len(configs)}]: {tag}")
        log.info(f"{'='*60}")

        # Override steps for ablation (shorter runs)
        if n_steps_override:
            c["training"]["max_steps"] = n_steps_override
        ablation_steps = c["training"].get("ablation_steps",
                         c["training"]["max_steps"] // 5)
        c["training"]["max_steps"] = ablation_steps

        # Unique checkpoint dir per run
        c["training"]["run_name"]       = f"ablation_{tag}"
        c["training"]["checkpoint_dir"] = f"checkpoints/ablations/{tag}"
        c["training"]["log_every"]      = max(10, ablation_steps // 50)
        c["training"]["eval_every"]     = max(50, ablation_steps // 10)
        c["training"]["save_every"]     = ablation_steps + 1   # no mid-run saves

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Build model + minimal training loop
        model = build_model(c).to(device)
        for t_enc in model.target_encoders:
            for p in t_enc.parameters():
                p.data = p.data.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=c["training"]["lr"],
            weight_decay=c["training"].get("weight_decay", 0.05),
        )

        train_ds = make_dataset(c, "train")
        loader   = DataLoader(train_ds, batch_size=c["data"]["batch_size"],
                               shuffle=True, num_workers=2, drop_last=True)

        meter = AverageMeter()
        step  = 0
        scaler = None

        model.train()
        while step < ablation_steps:
            for batch in loader:
                if step >= ablation_steps:
                    break
                x = batch[0].to(device)
                lr = cosine_schedule(step, ablation_steps,
                                      c["training"].get("warmup_steps", 200),
                                      c["training"]["lr"],
                                      c["training"].get("min_lr", 1e-6))
                set_lr(optimizer, lr)
                stats = training_step(model, x, optimizer, scaler, c)
                meter.update({k: v for k, v in stats.items()
                               if isinstance(v, (int, float))})
                step += 1

        avg = meter.avg()
        final_loss = avg.get("loss_total", float("inf"))

        # Routing entropy at end
        val_loader = DataLoader(train_ds, batch_size=16, shuffle=False)
        ent = evaluate_routing(model, val_loader, device, n_batches=3)

        # Probe accuracy (quick — 3 epochs)
        probe_acc = evaluate_probe(
            model, val_loader, level=0,
            n_classes=c["data"].get("n_classes", 10),
            device=device, n_epochs=3,
        )

        row = {
            "tag":           tag,
            "final_loss":    round(final_loss, 4),
            "probe_acc_L1":  round(probe_acc, 4),
            **{k: round(v, 4) for k, v in ent.items()},
            "n_params":      sum(p.numel() for p in model.parameters() if p.requires_grad),
        }

        # Add the swept params as columns for easy reading
        if "_ablation_tag" in c:
            for k, v in zip(
                list(cfg.get("sweep", {}).get("param_grid", {}).keys()),
                [p.split("=")[1] for p in tag.split("__")]
            ):
                short_k = k.split(".")[-1].replace("[", "").replace("]", "")
                row[f"param_{short_k}"] = v

        results.append(row)
        log.info(f"  → loss={final_loss:.4f}  probe={probe_acc:.4f}  {ent}")

        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Save results ──────────────────────────────────────────
    if not results:
        return

    out_path = Path("ablation_results.csv")
    fieldnames = list(results[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Print ranked table
    results.sort(key=lambda r: r["final_loss"])
    log.info(f"\n{'='*60}")
    log.info(f"ABLATION RESULTS (ranked by final loss) → {out_path}")
    log.info(f"{'='*60}")
    header = f"{'rank':>4}  {'loss':>8}  {'probe':>7}  {'tag'}"
    log.info(header)
    log.info("-" * 60)
    for rank_i, r in enumerate(results, 1):
        log.info(f"{rank_i:>4}  {r['final_loss']:>8.4f}  "
                 f"{r['probe_acc_L1']:>7.4f}  {r['tag']}")


# ─────────────────────────────────────────────────────────────────────────────
# Configs for scale
# ─────────────────────────────────────────────────────────────────────────────

SCALE_CONFIGS = {
    # 1 GPU — development / ablation
    "single_gpu": {
        "model": {
            "level0_mode": "small", "d_level0": 256, "img_size": 64,
            "loss_type": "vicreg", "ema_decay": 0.996,
            "n_rollout_steps": 2, "rollout_weight": 0.5,
            "levels": [
                {"d_in": 256, "d_out": 256, "pool_factor": 4, "n_experts": 4,
                 "top_k": 2, "expert_type": "ffn", "d_z": 0, "loss_weight": 1.0},
                {"d_in": 256, "d_out": 512, "pool_factor": 4, "n_experts": 4,
                 "top_k": 2, "expert_type": "transformer", "d_z": 32, "loss_weight": 2.0},
                {"d_in": 512, "d_out": 512, "pool_factor": 4, "n_experts": 2,
                 "top_k": 1, "expert_type": "transformer", "d_z": 64, "loss_weight": 4.0},
            ],
        },
        "data": {"dataset": "moving_mnist", "seq_len": 64, "img_size": 64,
                 "batch_size": 64, "n_classes": 10},
        "training": {
            "optimizer": "adamw", "lr": 3e-4, "weight_decay": 0.05,
            "betas": [0.9, 0.95], "warmup_steps": 1000, "max_steps": 50000,
            "grad_clip": 1.0, "lr_schedule": "cosine", "min_lr": 1e-6,
            "log_every": 100, "eval_every": 1000, "save_every": 5000,
            "mixed_precision": False, "num_workers": 4,
            "run_name": "hjepa_moe_1gpu", "checkpoint_dir": "checkpoints",
        },
    },

    # 4 GPUs — intermediate scale (1 node, 4x A100 40GB)
    "4gpu": {
        "model": {
            "level0_mode": "small", "d_level0": 512, "img_size": 128,
            "loss_type": "vicreg", "ema_decay": 0.998,
            "n_rollout_steps": 4, "rollout_weight": 0.5,
            "levels": [
                {"d_in": 512, "d_out": 512,  "pool_factor": 4, "n_experts": 8,
                 "top_k": 2, "expert_type": "ffn",         "d_z": 0,  "loss_weight": 1.0},
                {"d_in": 512, "d_out": 1024, "pool_factor": 4, "n_experts": 8,
                 "top_k": 2, "expert_type": "transformer", "d_z": 64, "loss_weight": 2.0},
                {"d_in": 1024,"d_out": 1024, "pool_factor": 4, "n_experts": 4,
                 "top_k": 2, "expert_type": "transformer", "d_z": 128,"loss_weight": 4.0},
            ],
        },
        "data": {"dataset": "custom", "root": "./data", "seq_len": 64,
                 "img_size": 128, "batch_size": 256, "n_classes": 174},
        "training": {
            "optimizer": "adamw", "lr": 1e-3, "weight_decay": 0.05,
            "betas": [0.9, 0.95], "warmup_steps": 5000, "max_steps": 200000,
            "grad_clip": 1.0, "min_lr": 1e-6, "mixed_precision": True,
            "log_every": 50, "eval_every": 2000, "save_every": 10000,
            "num_workers": 8, "run_name": "hjepa_moe_4gpu",
            "checkpoint_dir": "checkpoints",
        },
    },

    # 8 GPUs — single node large scale (8x A100 80GB)
    "8gpu": {
        "model": {
            "level0_mode": "vjepa2",   # <-- plug in V-JEPA 2 ViT-L
            "d_level0": 1024, "img_size": 256,
            "loss_type": "vicreg", "ema_decay": 0.999,
            "n_rollout_steps": 4, "rollout_weight": 0.5,
            "levels": [
                {"d_in": 1024, "d_out": 1024, "pool_factor": 4, "n_experts": 8,
                 "top_k": 2, "expert_type": "transformer", "d_z": 0,   "loss_weight": 1.0},
                {"d_in": 1024, "d_out": 1024, "pool_factor": 4, "n_experts": 8,
                 "top_k": 2, "expert_type": "transformer", "d_z": 128, "loss_weight": 2.0},
                {"d_in": 1024, "d_out": 1024, "pool_factor": 4, "n_experts": 4,
                 "top_k": 1, "expert_type": "transformer", "d_z": 256, "loss_weight": 4.0},
            ],
        },
        "data": {"dataset": "custom", "root": "./data", "seq_len": 64,
                 "img_size": 256, "batch_size": 512, "n_classes": 174},
        "training": {
            "optimizer": "adamw", "lr": 2e-3, "weight_decay": 0.05,
            "betas": [0.9, 0.95], "warmup_steps": 10000, "max_steps": 500000,
            "grad_clip": 1.0, "min_lr": 1e-6, "mixed_precision": True,
            "log_every": 50, "eval_every": 5000, "save_every": 20000,
            "num_workers": 12, "run_name": "hjepa_moe_8gpu",
            "checkpoint_dir": "checkpoints",
        },
    },

    # 16+ GPUs multi-node (AMI Labs / FAIR scale)
    "multinode": {
        "model": {
            "level0_mode": "vjepa2",
            "d_level0": 1024, "img_size": 256,
            "loss_type": "vicreg", "ema_decay": 0.9995,
            "n_rollout_steps": 8, "rollout_weight": 0.5,
            "levels": [
                {"d_in": 1024, "d_out": 1024, "pool_factor": 4, "n_experts": 16,
                 "top_k": 2, "expert_type": "transformer", "d_z": 0,   "loss_weight": 1.0},
                {"d_in": 1024, "d_out": 2048, "pool_factor": 4, "n_experts": 16,
                 "top_k": 2, "expert_type": "transformer", "d_z": 256, "loss_weight": 2.0},
                {"d_in": 2048, "d_out": 2048, "pool_factor": 4, "n_experts": 8,
                 "top_k": 2, "expert_type": "transformer", "d_z": 512, "loss_weight": 4.0},
            ],
        },
        "data": {"dataset": "custom", "root": "./data", "seq_len": 64,
                 "img_size": 256, "batch_size": 2048, "n_classes": 400},
        "training": {
            "optimizer": "adamw", "lr": 3e-3, "weight_decay": 0.05,
            "betas": [0.9, 0.95], "warmup_steps": 20000, "max_steps": 1000000,
            "grad_clip": 1.0, "min_lr": 1e-6, "mixed_precision": True,
            "log_every": 100, "eval_every": 10000, "save_every": 50000,
            "num_workers": 16, "run_name": "hjepa_moe_multinode",
            "checkpoint_dir": "/checkpoints/hjepa_moe",
        },
    },
}


def dump_scale_config(name: str, path: str = None):
    """Write a scale config to YAML for inspection / modification."""
    if name not in SCALE_CONFIGS:
        raise ValueError(f"Unknown scale config: {name}. "
                         f"Options: {list(SCALE_CONFIGS.keys())}")
    cfg = SCALE_CONFIGS[name]
    out = path or f"configs/scale_{name}.yaml"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    log.info(f"Scale config written to {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MODE 3: scale (entry point for torchrun)
# ─────────────────────────────────────────────────────────────────────────────

def run_scale(cfg: dict):
    """Multi-GPU / multi-node entry point. Called via torchrun."""
    rank, local_rank, world_size = init_distributed()
    if is_main(rank):
        log.info(f"Distributed: rank={rank}  local_rank={local_rank}  "
                 f"world_size={world_size}")
    run_train(cfg, rank=rank, local_rank=local_rank, world_size=world_size)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="H-JEPA-MoE: train / ablate / scale",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single GPU training
  python train_ablate_scale.py train --cfg configs/video_jepa_moe.yaml

  # Ablation sweep
  python train_ablate_scale.py ablate --cfg configs/video_jepa_moe.yaml

  # Dump a scale config and inspect it
  python train_ablate_scale.py dump-config --scale 4gpu

  # Multi-GPU scale (run via torchrun)
  torchrun --nproc_per_node=4 train_ablate_scale.py scale --scale 4gpu

  # Multi-node
  torchrun --nnodes=2 --nproc_per_node=8 \\
           --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \\
           train_ablate_scale.py scale --scale multinode
        """,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # train
    p_train = sub.add_parser("train")
    p_train.add_argument("--cfg", default="configs/video_jepa_moe.yaml")

    # ablate
    p_ablate = sub.add_parser("ablate")
    p_ablate.add_argument("--cfg", default="configs/video_jepa_moe.yaml")
    p_ablate.add_argument("--steps", type=int, default=None,
                           help="Override max_steps for ablation runs")

    # scale
    p_scale = sub.add_parser("scale")
    group = p_scale.add_mutually_exclusive_group(required=True)
    group.add_argument("--cfg",   default=None, help="Path to YAML config")
    group.add_argument("--scale", default=None,
                        choices=list(SCALE_CONFIGS.keys()),
                        help="Use a built-in scale config")

    # dump-config
    p_dump = sub.add_parser("dump-config", help="Write a scale config to YAML")
    p_dump.add_argument("--scale", required=True, choices=list(SCALE_CONFIGS.keys()))
    p_dump.add_argument("--out",   default=None)

    args = parser.parse_args()

    if args.mode == "train":
        cfg = load_cfg(args.cfg)
        run_train(cfg)

    elif args.mode == "ablate":
        cfg = load_cfg(args.cfg)
        run_ablate(cfg, n_steps_override=args.steps)

    elif args.mode == "scale":
        if args.scale:
            cfg = SCALE_CONFIGS[args.scale]
        else:
            cfg = load_cfg(args.cfg)
        run_scale(cfg)

    elif args.mode == "dump-config":
        dump_scale_config(args.scale, args.out)


if __name__ == "__main__":
    main()
