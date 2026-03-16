"""
Latent-space planners for H-JEPA-MoE.

Two planners (both used in FAIR/AMI Labs style):
  - CEM  : Cross-Entropy Method — iterative elite selection
           Used in V-JEPA 2-AC (Assran et al. 2025)
  - MPPI : Model Predictive Path Integral
           Used in EB-JEPA (Terver et al. 2026)

Both operate in the JEPA latent space at a chosen level.
Goal: find action/latent sequence z₀..z_{H-1} that minimizes
      distance(rollout(s₀, z), s_goal) in embedding space.

Key insight from Destrade et al. (2025) "Value-guided planning":
  Standard JEPA planning has many local minima.
  Shaping the latent space so Euclidean distance ≈ value function
  (via IQL-inspired training) dramatically improves success rate.
  This is implemented as an optional value-shaping loss in HJEPAMoE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
from dataclasses import dataclass


@dataclass
class PlannerConfig:
    horizon:       int   = 10      # planning steps H
    n_samples:     int   = 512     # CEM population / MPPI trajectories
    n_elite:       int   = 64      # CEM: number of elites kept
    n_iter:        int   = 5       # CEM: optimization iterations
    temperature:   float = 1.0    # MPPI: temperature λ
    noise_sigma:   float = 0.5    # initial action noise std
    min_sigma:     float = 0.05   # minimum std (prevents collapse)


class CEMPlanner(nn.Module):
    """
    Cross-Entropy Method planner in JEPA latent space.
    
    Follows V-JEPA 2-AC methodology exactly:
        Given s_current and s_goal embeddings at level ℓ,
        find z_0..z_{H-1} minimizing ||rollout(s, z) - s_goal||₁
        (L1 distance = goal-conditioned energy function)
    
    The predictor is rolled out autoregressively with the planned z.
    At each CEM iteration, the top n_elite trajectories become the new
    distribution mean/std.
    """
    
    def __init__(self, config: PlannerConfig = None):
        super().__init__()
        self.cfg = config or PlannerConfig()
    
    def plan(
        self,
        predictor:     nn.Module,         # MoEPredictor at target level
        s_current:     torch.Tensor,      # (1, d) current state embedding
        s_goal:        torch.Tensor,      # (1, d) goal state embedding
        d_z:           int,               # latent variable dimension
        value_fn:      Optional[Callable] = None,  # optional value shaping
    ) -> Tuple[torch.Tensor, float, dict]:
        """
        Args:
            predictor:  trained MoEPredictor
            s_current:  (1, d) starting state
            s_goal:     (1, d) goal state
            d_z:        dimension of latent z (0 = no action)
            value_fn:   optional V(s, g) -> scalar for shaping MPC cost
        Returns:
            z_seq:      (H, d_z) planned latent sequence
            best_cost:  final cost
            stats:      planning diagnostics
        """
        device = s_current.device
        H = self.cfg.horizon
        
        if d_z == 0:
            # Deterministic rollout — no latent variable
            s = s_current
            for _ in range(H):
                s, _ = predictor(s)
            cost = F.l1_loss(s, s_goal.expand_as(s)).item()
            return torch.zeros(H, 0, device=device), cost, {"n_iter": 0}
        
        # Initialize CEM distribution: z ~ N(0, sigma²)
        z_mean  = torch.zeros(H, d_z, device=device)
        z_sigma = torch.ones(H, d_z, device=device) * self.cfg.noise_sigma
        
        best_cost  = float("inf")
        best_z_seq = z_mean.clone()
        cost_history = []
        
        for iteration in range(self.cfg.n_iter):
            # Sample trajectories: (n_samples, H, d_z)
            noise = torch.randn(self.cfg.n_samples, H, d_z, device=device)
            z_batch = z_mean.unsqueeze(0) + z_sigma.unsqueeze(0) * noise
            
            # Rollout all samples in parallel
            costs = self._batch_rollout(predictor, s_current, s_goal,
                                         z_batch, value_fn)
            
            # Select elites
            n_elite = min(self.cfg.n_elite, self.cfg.n_samples)
            elite_idx = costs.argsort()[:n_elite]
            elites = z_batch[elite_idx]   # (n_elite, H, d_z)
            
            # Update distribution
            z_mean  = elites.mean(0)
            z_sigma = (elites.std(0) + 1e-6).clamp(min=self.cfg.min_sigma)
            
            iter_best = costs[elite_idx[0]].item()
            cost_history.append(iter_best)
            if iter_best < best_cost:
                best_cost  = iter_best
                best_z_seq = elites[0]
        
        stats = {
            "cost_history":   cost_history,
            "final_z_std":    z_sigma.mean().item(),
            "n_iter":         self.cfg.n_iter,
        }
        return best_z_seq, best_cost, stats
    
    def _batch_rollout(
        self,
        predictor: nn.Module,
        s_init:    torch.Tensor,        # (1, d)
        s_goal:    torch.Tensor,        # (1, d)
        z_batch:   torch.Tensor,        # (N, H, d_z)
        value_fn:  Optional[Callable],
    ) -> torch.Tensor:
        """Roll out N trajectories in batch, return costs (N,)."""
        N, H, d_z = z_batch.shape
        d = s_init.shape[-1]
        
        # Expand initial state for batch
        s = s_init.expand(N, -1)   # (N, d)
        
        for t in range(H):
            z_t = z_batch[:, t]       # (N, d_z)
            s, _ = predictor(s, z_t)  # (N, d)
        
        # L1 goal-conditioned energy (V-JEPA 2-AC style)
        costs = F.l1_loss(s, s_goal.expand(N, -1), reduction="none").mean(-1)
        
        # Optional value shaping (Destrade et al. 2025)
        if value_fn is not None:
            value_cost = -value_fn(s, s_goal.expand(N, -1))  # negate: V is reward
            costs = costs + 0.5 * value_cost
        
        return costs


class MPPIPlanner(nn.Module):
    """
    Model Predictive Path Integral planner.
    
    Used in EB-JEPA examples (Terver et al. 2026).
    Gradient-free optimizer that weights sampled trajectories
    by their softmax-normalized returns (temperature λ).
    
    More sample-efficient than CEM for smooth cost landscapes.
    """
    
    def __init__(self, config: PlannerConfig = None):
        super().__init__()
        self.cfg = config or PlannerConfig()
        # Running mean for warm-starting
        self._z_mean = None
    
    def plan(
        self,
        predictor:  nn.Module,
        s_current:  torch.Tensor,
        s_goal:     torch.Tensor,
        d_z:        int,
        reset:      bool = False,
    ) -> Tuple[torch.Tensor, float, dict]:
        device = s_current.device
        H = self.cfg.horizon
        N = self.cfg.n_samples
        
        if d_z == 0:
            s = s_current
            for _ in range(H):
                s, _ = predictor(s)
            cost = F.l1_loss(s, s_goal.expand_as(s)).item()
            return torch.zeros(H, 0, device=device), cost, {}
        
        # Warm start from previous plan (receding horizon MPC)
        if self._z_mean is None or reset:
            self._z_mean = torch.zeros(H, d_z, device=device)
        
        # Sample perturbations
        eps = torch.randn(N, H, d_z, device=device) * self.cfg.noise_sigma
        z_batch = self._z_mean.unsqueeze(0) + eps   # (N, H, d_z)
        
        # Rollout
        s = s_current.expand(N, -1)
        for t in range(H):
            s, _ = predictor(s, z_batch[:, t])
        
        # Costs and MPPI weights
        costs = F.l1_loss(s, s_goal.expand(N, -1), reduction="none").mean(-1)
        
        # MPPI weighting: w_i = exp(-costs_i / λ) / Z
        beta = costs.min()
        weights = torch.exp(-(costs - beta) / self.cfg.temperature)
        weights = weights / weights.sum()   # (N,)
        
        # Weighted update to z_mean
        self._z_mean = (weights.unsqueeze(-1).unsqueeze(-1) * z_batch).sum(0)
        
        best_cost = costs.min().item()
        best_z    = self._z_mean.clone()
        
        # Shift for receding horizon
        self._z_mean = torch.cat([self._z_mean[1:], torch.zeros(1, d_z, device=device)])
        
        return best_z, best_cost, {"weight_entropy": -(weights * weights.log()).sum().item()}


class ValueShapingLoss(nn.Module):
    """
    IQL-inspired value shaping for JEPA latent space.
    
    From: Destrade et al. (2025) "Value-guided action planning with JEPA"
    
    Idea: train the encoder so that Euclidean distance between state
    embeddings approximates the negative goal-conditioned value function V(s,g).
    This structures the latent space for better MPC optimization.
    
    Loss:
        L_value = IQL_loss(d_embed(s, g), -V*(s, g))
    
    where V*(s, g) is estimated by implicit Q-learning from offline data.
    """
    
    def __init__(self, d_model: int, d_hidden: int = 256):
        super().__init__()
        # Learned value function head (Q-network)
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, 1),
        )
        # Expectile parameter for IQL
        self.expectile = 0.7   # >0.5 = optimistic (standard IQL)
    
    def forward(
        self,
        s_emb:    torch.Tensor,   # (B, d) state embeddings
        g_emb:    torch.Tensor,   # (B, d) goal embeddings
        returns:  torch.Tensor,   # (B,) discounted returns from offline data
    ) -> torch.Tensor:
        """
        IQL value loss.
        Shapes embedding space so d(s, g) ≈ -V*(s, g).
        """
        sg = torch.cat([s_emb, g_emb], dim=-1)   # (B, 2d)
        v_pred = self.value_head(sg).squeeze(-1)   # (B,)
        
        # Expectile regression (IQL core)
        diff = returns - v_pred
        weight = torch.where(diff >= 0,
                             torch.full_like(diff, self.expectile),
                             torch.full_like(diff, 1 - self.expectile))
        return (weight * diff.pow(2)).mean()
    
    def get_value(self, s_emb: torch.Tensor, g_emb: torch.Tensor) -> torch.Tensor:
        """Inference: V(s, g) for MPC shaping."""
        with torch.no_grad():
            sg = torch.cat([s_emb, g_emb], dim=-1)
            return self.value_head(sg).squeeze(-1)
