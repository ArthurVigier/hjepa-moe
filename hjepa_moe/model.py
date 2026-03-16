"""
H-JEPA-MoE: The full hierarchical model.

Stacks L JEPA levels. At each level:
  1. TemporalEncoder compresses k consecutive lower-level states
  2. MoEPredictor predicts the next state at this level
  3. VICReg/SIGReg loss trained with EMA target

Training follows V-JEPA 2-AC style:
  - EMA (exponential moving average) for target encoder
  - Multi-step rollout loss to reduce error accumulation
  - Teacher forcing for short-horizon + rollout for generalization

The hierarchical loss is a weighted sum across levels:
  L = sum_ℓ w_ℓ * (prediction_loss_ℓ + regularization_ℓ + aux_moe_ℓ)

with w_ℓ typically increasing with level (higher = harder = more weight).
"""

import torch
import torch.nn as nn
import copy
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from hjepa_moe.encoders.temporal import TemporalEncoder, Level0Encoder
from hjepa_moe.predictors.moe_predictor import MoEPredictor
from hjepa_moe.losses.vicreg import VICRegLoss, SIGRegLoss


@dataclass
class LevelConfig:
    """Configuration for one H-JEPA level."""
    d_in:         int           # input dim (from level below)
    d_out:        int           # output dim at this level
    pool_factor:  int           # temporal compression factor
    n_experts:    int = 4       # number of MoE experts
    top_k:        int = 2       # active experts per forward pass
    expert_type:  str = "ffn"   # 'ffn' or 'transformer'
    d_z:          int = 0       # latent variable dim (0 = disabled)
    pooling:      str = "attention"  # temporal pooling strategy
    n_heads:      int = 4       # for transformer experts / attentive pool
    loss_weight:  float = 1.0   # weight of this level's loss in total


@dataclass
class HJEPAMoEConfig:
    """Full H-JEPA-MoE configuration."""
    levels: List[LevelConfig]
    
    # Loss settings
    loss_type:    str   = "vicreg"   # 'vicreg', 'sigreg', or 'infonce'
    sim_coef:     float = 25.0
    var_coef:     float = 25.0
    cov_coef:     float = 1.0
    
    # EMA for target encoder
    ema_decay:    float = 0.996   # typical JEPA EMA value
    
    # Multi-step rollout (V-JEPA 2-AC style)
    n_rollout_steps: int = 2   # additional unrolled steps for robustness
    rollout_weight:  float = 0.5
    
    # Level 0 encoder
    level0_mode:  str = "small"   # 'small' or 'vjepa2'
    d_level0:     int = 256
    img_size:     int = 64
    
    # Example default config (3 levels, like H-JEPA paper figure)
    @classmethod
    def default_3level(cls) -> "HJEPAMoEConfig":
        return cls(
            levels=[
                LevelConfig(d_in=256, d_out=256, pool_factor=4,
                            n_experts=4, top_k=2, expert_type="ffn",
                            loss_weight=1.0),
                LevelConfig(d_in=256, d_out=512, pool_factor=4,
                            n_experts=4, top_k=2, expert_type="transformer",
                            loss_weight=2.0),
                LevelConfig(d_in=512, d_out=512, pool_factor=4,
                            n_experts=2, top_k=1, expert_type="transformer",
                            loss_weight=4.0),
            ],
            loss_type="vicreg",
        )


class HJEPAMoE(nn.Module):
    """
    Hierarchical JEPA with MoE Predictors.
    
    Architecture overview (3 levels):
    
        Video frames
             |
        Level0Encoder (ViT or ConvNet)
             |
        [s_0^0, s_0^1, ..., s_0^T]   <-- level 0 states
             |
        TemporalEncoder_1 (pool 4)
             |
        [s_1^0, ..., s_1^{T/4}]       <-- level 1 states
          |       |
     MoEPredictor_1              (predicts s_1^{t+1} from s_1^t)
             |
        TemporalEncoder_2 (pool 4)
             |
        [s_2^0, ..., s_2^{T/16}]      <-- level 2 states
          |
     MoEPredictor_2              (predicts s_2^{t+1} from s_2^t)
             |
        [s_3^0, ..., s_3^{T/64}]      <-- level 3 states (abstract goals)
          |
     MoEPredictor_3              (top-level planning horizon)
    
    EMA target encoders: Each TemporalEncoder has an EMA copy used
    as the prediction target, preventing representation collapse.
    """
    
    def __init__(self, config: HJEPAMoEConfig, vjepa2_model=None):
        super().__init__()
        self.config = config
        
        # Level 0 encoder
        self.enc0 = Level0Encoder(
            mode      = config.level0_mode,
            d_out     = config.d_level0,
            img_size  = config.img_size,
            vjepa2_model = vjepa2_model,
        )
        
        # Build hierarchy
        self.temporal_encoders = nn.ModuleList()
        self.moe_predictors    = nn.ModuleList()
        self.target_encoders   = []  # EMA copies (not in ModuleList, not trained)
        
        for level_cfg in config.levels:
            enc = TemporalEncoder(
                d_in        = level_cfg.d_in,
                d_out       = level_cfg.d_out,
                pool_factor = level_cfg.pool_factor,
                pooling     = level_cfg.pooling,
                n_heads     = level_cfg.n_heads,
            )
            pred = MoEPredictor(
                d_in        = level_cfg.d_out,
                d_out       = level_cfg.d_out,
                n_experts   = level_cfg.n_experts,
                top_k       = level_cfg.top_k,
                expert_type = level_cfg.expert_type,
                d_z         = level_cfg.d_z,
            )
            self.temporal_encoders.append(enc)
            self.moe_predictors.append(pred)
            
            # EMA target encoder (deep copy, no grad)
            target_enc = copy.deepcopy(enc)
            for p in target_enc.parameters():
                p.requires_grad_(False)
            self.target_encoders.append(target_enc)
        
        # Losses per level
        if config.loss_type == "vicreg":
            self.losses = nn.ModuleList([
                VICRegLoss(config.sim_coef, config.var_coef, config.cov_coef)
                for _ in config.levels
            ])
        elif config.loss_type == "sigreg":
            self.losses = nn.ModuleList([
                SIGRegLoss(d_model=lvl.d_out)
                for lvl in config.levels
            ])
        else:
            raise ValueError(f"Unknown loss_type: {config.loss_type}")
    
    def forward(
        self,
        x: torch.Tensor,
        z_list: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Full forward pass with loss computation.
        
        Args:
            x:      video input (B, T, C, H, W) or (B, T, d_level0)
            z_list: optional list of latent variables, one per level
        Returns:
            total_loss: scalar
            stats:      dict of per-level losses and routing stats
        """
        if z_list is None:
            z_list = [None] * len(self.config.levels)
        
        stats = {}
        total_loss = torch.tensor(0.0, device=x.device)
        
        # Level 0: encode frames
        if x.dim() == 5:  # video: (B, T, C, H, W)
            B, T, C, H, W = x.shape
            states = self.enc0(x.reshape(B*T, C, H, W))  # (B*T, N, d0)
            if states.dim() == 3:
                states = states.mean(dim=1)  # pool spatial -> (B*T, d0)
            states = states.reshape(B, T, -1)  # (B, T, d0)
        else:
            states = x  # already encoded externally
        
        # Walk up the hierarchy
        for ℓ, (enc, target_enc, pred, loss_fn, level_cfg) in enumerate(zip(
            self.temporal_encoders,
            self.target_encoders,
            self.moe_predictors,
            self.losses,
            self.config.levels,
        )):
            # Online encoder: compress states from level below
            states_ℓ = enc(states)   # (B, T//k, d_out)
            
            # Target encoder (EMA — no grad): same compression
            with torch.no_grad():
                target_ℓ = target_enc(states)  # (B, T//k, d_out)
            
            # Predict next state at this level
            # Context: all steps except last
            ctx    = states_ℓ[:, :-1]   # (B, T//k - 1, d_out)
            target = target_ℓ[:, 1:]    # (B, T//k - 1, d_out)
            
            # Flatten for predictor
            B_, Tl, d = ctx.shape
            ctx_flat    = ctx.reshape(B_ * Tl, d)
            target_flat = target.reshape(B_ * Tl, d)
            
            z = z_list[ℓ]
            if z is not None and z.dim() == 2:
                # Expand z to match flattened batch
                z = z.unsqueeze(1).expand(-1, Tl, -1).reshape(B_ * Tl, -1)
            
            pred_flat, aux_loss = pred(ctx_flat, z)
            
            # Level loss
            level_loss, level_stats = loss_fn(pred_flat, target_flat.detach())
            
            # Multi-step rollout (V-JEPA 2-AC style)
            if self.config.n_rollout_steps > 0 and self.training:
                rollout_loss = self._multi_step_rollout(
                    pred, ctx_flat, target_ℓ.reshape(B_ * Tl, d),
                    n_steps=self.config.n_rollout_steps, z=z,
                )
                level_loss = level_loss + self.config.rollout_weight * rollout_loss
                level_stats["loss_rollout"] = rollout_loss.item()
            
            # Add aux MoE load balancing loss
            if aux_loss is not None:
                level_loss = level_loss + aux_loss
                level_stats["loss_moe_aux"] = aux_loss.item()
            
            total_loss = total_loss + level_cfg.loss_weight * level_loss
            stats[f"level_{ℓ}"] = level_stats
            
            # States at this level become input to next level
            states = states_ℓ.detach()
        
        stats["loss_total"] = total_loss.item()
        return total_loss, stats
    
    def _multi_step_rollout(
        self,
        pred:   MoEPredictor,
        ctx:    torch.Tensor,
        target: torch.Tensor,
        n_steps: int,
        z:      Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Unrolled multi-step prediction loss.
        Train the predictor to maintain accuracy over n_steps autonomous rolls.
        This is key to preventing error accumulation during planning.
        """
        current = ctx
        rollout_loss = torch.tensor(0.0, device=ctx.device)
        
        for step in range(n_steps):
            current, _ = pred(current, z)
            # Compare against shifted targets (if available)
            if step < target.shape[0] - 1:
                rollout_loss = rollout_loss + (
                    (current - target[step+1:step+2].expand_as(current))
                    .pow(2).mean()
                )
        
        return rollout_loss / n_steps
    
    @torch.no_grad()
    def update_ema(self):
        """
        Update EMA target encoders after each optimizer step.
        Call this after loss.backward() + optimizer.step().
        """
        decay = self.config.ema_decay
        for enc, target_enc in zip(self.temporal_encoders, self.target_encoders):
            for p_online, p_target in zip(enc.parameters(), target_enc.parameters()):
                p_target.data = decay * p_target.data + (1 - decay) * p_online.data
    
    def get_level_states(
        self, x: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Inference helper: return all level representations.
        Useful for probing, visualization, or downstream planning.
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 5:
                B, T, C, H, W = x.shape
                states = self.enc0(x.reshape(B*T, C, H, W))
                if states.dim() == 3:
                    states = states.mean(dim=1)
                states = states.reshape(B, T, -1)
            else:
                states = x
            
            level_states = [states]
            for enc in self.temporal_encoders:
                states = enc(states)
                level_states.append(states)
            
            return level_states
    
    def plan(
        self,
        current_state: torch.Tensor,
        goal_state:    torch.Tensor,
        level:         int = -1,
        n_steps:       int = 10,
        n_samples:     int = 100,
    ) -> Tuple[torch.Tensor, float]:
        """
        Simple CEM (Cross-Entropy Method) planning at a given level.
        Follows V-JEPA 2-AC MPC approach.
        
        Finds a sequence of latent actions z that minimizes
        distance between predicted states and goal.
        
        Args:
            current_state: (1, d_level) embedding
            goal_state:    (1, d_level) goal embedding
            level:         which JEPA level to plan at (-1 = top)
            n_steps:       planning horizon
            n_samples:     CEM population size
        Returns:
            best_z_seq: (n_steps, d_z) — sequence of latent variables
            best_cost:  scalar — final goal distance
        """
        if level < 0:
            level = len(self.moe_predictors) - 1
        
        pred     = self.moe_predictors[level]
        d_z      = self.config.levels[level].d_z
        d_model  = self.config.levels[level].d_out
        
        if d_z == 0:
            # No latent variable: single deterministic rollout
            s = current_state
            for _ in range(n_steps):
                s, _ = pred(s)
            cost = (s - goal_state).pow(2).mean().item()
            return torch.zeros(n_steps, 0), cost
        
        # CEM: iteratively refine latent z distribution
        z_mean = torch.zeros(n_steps, d_z, device=current_state.device)
        z_std  = torch.ones(n_steps, d_z, device=current_state.device)
        
        for cem_iter in range(5):
            # Sample population
            z_samples = (z_mean.unsqueeze(0) +
                         z_std.unsqueeze(0) * torch.randn(n_samples, n_steps, d_z,
                                                           device=current_state.device))
            
            costs = []
            for i in range(n_samples):
                s = current_state.expand(1, -1)
                for t in range(n_steps):
                    s, _ = pred(s, z_samples[i, t:t+1])
                cost = (s - goal_state).pow(2).mean()
                costs.append(cost.item())
            
            # Keep top-20% (elites)
            costs_t = torch.tensor(costs)
            elite_idx = costs_t.argsort()[:max(1, n_samples // 5)]
            elites = z_samples[elite_idx]
            z_mean = elites.mean(0)
            z_std  = elites.std(0).clamp(min=1e-4)
        
        best_cost = min(costs)
        return z_mean, best_cost
