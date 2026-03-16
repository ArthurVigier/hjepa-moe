"""
Evaluation utilities for H-JEPA-MoE.

Three evaluation axes:
  1. Representation quality: linear / attentive probing per level
  2. Expert specialization: routing entropy and expert activation patterns
  3. Planning quality: success rate in Two Rooms / maze environments

Mirrors EB-JEPA evaluation protocol.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


# ──────────────────────────────────────────────
# 1. Linear / Attentive Probing
# ──────────────────────────────────────────────

class LinearProbe(nn.Module):
    """
    Linear probe on frozen JEPA representations.
    Standard evaluation: train only this layer, freeze everything else.
    """
    def __init__(self, d_in: int, n_classes: int):
        super().__init__()
        self.head = nn.Linear(d_in, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class AttentiveProbe(nn.Module):
    """
    Attentive probe — allows attending over temporal sequence.
    Used in V-JEPA 2 evaluation for video classification.
    More expressive than linear but still lightweight.
    """
    def __init__(self, d_in: int, n_classes: int, n_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_in) * 0.02)
        self.attn  = nn.MultiheadAttention(d_in, n_heads, batch_first=True)
        self.norm  = nn.RMSNorm(d_in)
        self.head  = nn.Linear(d_in, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d) or (B, d)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        pooled, _ = self.attn(q, x, x)
        pooled = self.norm(pooled.squeeze(1))
        return self.head(pooled)


@torch.no_grad()
def extract_features(
    model,
    dataloader: torch.utils.data.DataLoader,
    level: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract frozen features from a specific JEPA level."""
    all_feats, all_labels = [], []
    model.eval()
    
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            x, y = batch[0].to(device), batch[1].to(device)
        else:
            x = batch.to(device)
            y = torch.zeros(x.size(0), dtype=torch.long, device=device)
        
        level_states = model.get_level_states(x)
        feats = level_states[level + 1]   # +1 because index 0 = raw enc0 output
        
        # Pool temporal dim if needed
        if feats.dim() == 3:
            feats = feats.mean(1)
        
        all_feats.append(feats.cpu())
        all_labels.append(y.cpu())
    
    return torch.cat(all_feats), torch.cat(all_labels)


def train_probe(
    train_feats:  torch.Tensor,
    train_labels: torch.Tensor,
    val_feats:    torch.Tensor,
    val_labels:   torch.Tensor,
    probe_type:   str = "linear",
    n_epochs:     int = 20,
    lr:           float = 1e-3,
    device:       torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Train a probe and return accuracy metrics."""
    n_classes = int(train_labels.max().item()) + 1
    d_in      = train_feats.shape[-1]
    
    if probe_type == "linear":
        probe = LinearProbe(d_in, n_classes)
    else:
        probe = AttentiveProbe(d_in, n_classes)
    probe = probe.to(device)
    
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    train_ds  = torch.utils.data.TensorDataset(
        train_feats.to(device), train_labels.to(device)
    )
    loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)
    
    for epoch in range(n_epochs):
        probe.train()
        for x, y in loader:
            logits = probe(x)
            loss   = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Evaluate
    probe.eval()
    with torch.no_grad():
        val_logits = probe(val_feats.to(device))
        acc = (val_logits.argmax(-1).cpu() == val_labels).float().mean().item()
    
    return {"probe_acc": acc, "n_classes": n_classes, "d_in": d_in}


# ──────────────────────────────────────────────
# 2. Expert Specialization Analysis
# ──────────────────────────────────────────────

def compute_routing_entropy(gate_probs: np.ndarray) -> float:
    """
    Routing entropy H = -Σ p_i log(p_i).
    Max entropy = log(N) = all experts used equally.
    Min entropy = 0 = single expert dominates.
    """
    p = gate_probs / gate_probs.sum()
    p = np.clip(p, 1e-8, 1.0)
    return float(-(p * np.log(p)).sum())


def analyze_expert_specialization(
    model,
    dataloader: torch.utils.data.DataLoader,
    level: int,
    device: torch.device,
    metadata_key: str = "motion_type",   # if dataset provides motion labels
) -> Dict:
    """
    Analyze which experts activate for different input types.
    Returns per-expert activation rates and routing entropy.
    
    Key metric: if experts specialize by dynamics type,
    routing entropy should be high overall but low per-class
    (each class uses a specific subset of experts).
    """
    model.eval()
    pred = model.moe_predictors[level]
    enc  = model.temporal_encoders[level]
    
    expert_activations = []
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            
            # Get level states
            level_states = model.get_level_states(x)
            states = level_states[level + 1]   # (B, T, d)
            
            if states.dim() == 3:
                B, T, d = states.shape
                s_flat = states[:, :-1].reshape(B * (T-1), d)
            else:
                s_flat = states
            
            stats = pred.get_routing_stats(s_flat)
            expert_activations.append(stats["expert_usage"])
    
    expert_activations = np.array(expert_activations)
    mean_usage = expert_activations.mean(0)
    
    return {
        "mean_expert_usage":    mean_usage,
        "routing_entropy":      compute_routing_entropy(mean_usage),
        "max_entropy":          float(np.log(len(mean_usage))),
        "entropy_ratio":        compute_routing_entropy(mean_usage) / max(np.log(len(mean_usage)), 1e-8),
        "dominant_expert":      int(mean_usage.argmax()),
        "expert_usage_std":     float(mean_usage.std()),
    }


# ──────────────────────────────────────────────
# 3. Planning Evaluation (Two Rooms env)
# ──────────────────────────────────────────────

class TwoRoomsEnv:
    """
    Two Rooms environment from EB-JEPA.
    Agent navigates a 2D grid with a wall + doorway.
    Tests long-horizon planning (must go through door).
    
    State: (x, y) position
    Goal:  reach target (x_g, y_g) in other room
    """
    
    def __init__(self, size: int = 32, door_pos: int = None):
        self.size     = size
        self.door_pos = door_pos or size // 2   # door at center
        self.wall_x   = size // 2
        # Walls: x=wall_x except at door_pos
        self._make_walls()
    
    def _make_walls(self):
        self.walls = set()
        for y in range(self.size):
            if y != self.door_pos:
                self.walls.add((self.wall_x, y))
    
    def reset(self, random_start: bool = True, random_goal: bool = True):
        import random
        # Start in left room
        while True:
            x = random.randint(0, self.wall_x - 1)
            y = random.randint(0, self.size - 1)
            if (x, y) not in self.walls:
                self.pos = (x, y)
                break
        # Goal in right room
        while True:
            x_g = random.randint(self.wall_x + 1, self.size - 1)
            y_g = random.randint(0, self.size - 1)
            if (x_g, y_g) not in self.walls:
                self.goal = (x_g, y_g)
                break
        return self._obs()
    
    def step(self, action: int):
        # 0=up, 1=down, 2=left, 3=right
        dx, dy = [(0,-1),(0,1),(-1,0),(1,0)][action]
        nx, ny = self.pos[0] + dx, self.pos[1] + dy
        nx = np.clip(nx, 0, self.size - 1)
        ny = np.clip(ny, 0, self.size - 1)
        if (nx, ny) not in self.walls:
            self.pos = (nx, ny)
        done = (self.pos == self.goal)
        return self._obs(), -1.0, done, {}
    
    def _obs(self):
        """One-hot grid observation."""
        obs = np.zeros((2, self.size, self.size), dtype=np.float32)
        obs[0, self.pos[1], self.pos[0]] = 1.0
        obs[1, self.goal[1], self.goal[0]] = 1.0
        return obs


def evaluate_planning(
    model,
    planner,
    level: int,
    n_episodes: int = 200,
    max_steps: int = 100,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Evaluate planning success rate in Two Rooms environment.
    Mirrors EB-JEPA evaluation protocol.
    """
    env     = TwoRoomsEnv()
    success = 0
    total_steps = []
    
    for ep in range(n_episodes):
        obs  = env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            # Encode current state and goal
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                level_states = model.get_level_states(obs_t.unsqueeze(1))
                s_curr = level_states[level + 1].mean(1)   # (1, d)
                
                # Goal state from goal channel
                goal_t = torch.zeros_like(obs_t)
                goal_t[0, 1] = obs_t[0, 1]
                goal_states = model.get_level_states(goal_t.unsqueeze(1))
                s_goal = goal_states[level + 1].mean(1)
            
            # Plan
            d_z = model.config.levels[level].d_z
            z_seq, cost, _ = planner.plan(
                predictor   = model.moe_predictors[level],
                s_current   = s_curr,
                s_goal      = s_goal,
                d_z         = d_z,
            )
            
            # Execute first action (MPC: replan at every step)
            # For simplicity, map planned z to discrete action via argmin distance
            action = int(cost) % 4   # placeholder: replace with learned action decoder
            obs, reward, done, _ = env.step(action)
            steps += 1
        
        if done:
            success += 1
            total_steps.append(steps)
    
    return {
        "success_rate":  success / n_episodes,
        "mean_steps":    float(np.mean(total_steps)) if total_steps else max_steps,
        "n_episodes":    n_episodes,
    }


# ──────────────────────────────────────────────
# 4. Full Evaluation Pipeline
# ──────────────────────────────────────────────

def run_full_eval(
    model,
    train_loader: torch.utils.data.DataLoader,
    val_loader:   torch.utils.data.DataLoader,
    device:       torch.device,
    n_levels:     int,
    probe_type:   str = "linear",
) -> Dict:
    """Run all evaluations and return aggregated metrics."""
    results = {}
    
    for level in range(n_levels):
        print(f"\n[Eval] Level {level}")
        
        # 1. Extract features and probe
        train_feats, train_labels = extract_features(model, train_loader, level, device)
        val_feats,   val_labels   = extract_features(model, val_loader, level, device)
        
        probe_results = train_probe(
            train_feats, train_labels,
            val_feats, val_labels,
            probe_type=probe_type,
            device=device,
        )
        results[f"level_{level}_probe"] = probe_results
        print(f"  Probe acc: {probe_results['probe_acc']:.3f}")
        
        # 2. Expert specialization
        spec = analyze_expert_specialization(model, val_loader, level, device)
        results[f"level_{level}_routing"] = spec
        print(f"  Routing entropy: {spec['routing_entropy']:.3f} "
              f"/ {spec['max_entropy']:.3f} (ratio: {spec['entropy_ratio']:.2f})")
        print(f"  Expert usage: {spec['mean_expert_usage'].round(3)}")
    
    return results
