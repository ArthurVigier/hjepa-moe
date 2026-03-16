"""
MoE Predictor for H-JEPA

Each JEPA level has one MoEPredictor that maps:
    (s_ℓ^t, z_ℓ) -> ŝ_ℓ^{t+Δ}

where:
  s_ℓ^t  : current state at level ℓ  (B, d_ℓ)
  z_ℓ    : optional latent variable   (B, d_z) — captures uncertainty
  ŝ_ℓ    : predicted next state       (B, d_ℓ)

Architecture:
  - Router: linear(d_ℓ) -> N logits -> softmax (or sparse top-k)
  - N experts: each a small Transformer MLP or GRU
  - Weighted sum of active expert outputs
  - Load-balancing auxiliary loss (DeepSeek-style)

Key design choices aligned with FAIR/AMI Labs style:
  - SwiGLU FFN inside each expert (as in Llama3 layers)
  - RMSNorm pre/post
  - Auxiliary-loss-free load balancing option (DeepSeek V3 style)
  - Latent variable z injected via FiLM conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SwiGLUExpert(nn.Module):
    """Single MoE expert: SwiGLU FFN (used in Llama3 / DeepSeek)."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj   = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.drop      = nn.Dropout(dropout)
        self.norm      = nn.RMSNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (gate ⊙ SiLU) * up, then project down
        gate = F.silu(self.gate_proj(x))
        up   = self.up_proj(x)
        h    = self.drop(gate * up)
        return self.norm(self.down_proj(h))


class TransformerExpert(nn.Module):
    """
    Single MoE expert: 2-layer Transformer block.
    More expressive than FFN-only — used for higher levels
    where temporal dependencies matter.
    """
    
    def __init__(self, d_model: int, n_heads: int = 4, d_ff: int = None,
                 dropout: float = 0.0):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.norm1 = nn.RMSNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                           batch_first=True)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn   = SwiGLUExpert(d_model, d_ff, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T_local, d_model)  — small temporal window
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + residual
        residual = x
        x = self.ffn(self.norm2(x))
        return x + residual


class FiLMConditioner(nn.Module):
    """
    Feature-wise Linear Modulation for latent variable injection.
    z -> (scale, shift) that modulate the hidden state.
    Used to inject uncertainty signal z_ℓ into each expert.
    """
    
    def __init__(self, d_z: int, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_z, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model * 2),
        )
    
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # x: (B, d_model), z: (B, d_z)
        gamma_beta = self.proj(z)          # (B, 2 * d_model)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return x * (1 + gamma) + beta


class MoERouter(nn.Module):
    """
    Token router: maps input state to expert selection probabilities.
    
    Supports two modes:
      - 'softmax': dense mixture (all experts contribute, weighted)
      - 'topk': sparse routing (only top-k experts active)
    
    Load balancing loss follows DeepSeek-V3 auxiliary-loss-free approach:
      We track bias terms per expert that shift routing probabilities
      to maintain balance, rather than adding a separate auxiliary loss.
      Optionally can use classic Shazeer-style auxiliary loss.
    """
    
    def __init__(self, d_model: int, n_experts: int,
                 top_k: int = 2,
                 routing_mode: str = "topk",
                 use_aux_loss: bool = True,
                 aux_loss_coef: float = 0.01):
        super().__init__()
        assert routing_mode in ("softmax", "topk")
        self.n_experts    = n_experts
        self.top_k        = top_k
        self.routing_mode = routing_mode
        self.use_aux_loss = use_aux_loss
        self.aux_loss_coef = aux_loss_coef
        
        self.router = nn.Linear(d_model, n_experts, bias=False)
        # Expert bias for load balancing (DeepSeek-V3 style)
        self.expert_bias = nn.Parameter(torch.zeros(n_experts))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, d_model)
        Returns:
            gates:       (B, n_experts) — full gate weights (sparse if topk)
            indices:     (B, top_k) — selected expert indices
            aux_loss:    scalar or None
        """
        logits = self.router(x)   # (B, n_experts)
        
        if self.routing_mode == "softmax":
            gates = F.softmax(logits, dim=-1)
            indices = gates.topk(self.top_k, dim=-1).indices
            aux_loss = None
        else:  # topk
            # Biased routing for load balancing (DeepSeek V3 style)
            biased_logits = logits + self.expert_bias
            top_vals, indices = biased_logits.topk(self.top_k, dim=-1)
            
            # Renormalize gate weights using ORIGINAL logits (not biased)
            selected_logits = logits.gather(-1, indices)
            gates_topk = F.softmax(selected_logits, dim=-1)
            
            # Scatter back to full gates tensor (zeros for non-selected)
            gates = torch.zeros_like(logits)
            gates.scatter_(-1, indices, gates_topk)
            
            # Auxiliary load balancing loss (optional)
            aux_loss = None
            if self.use_aux_loss and self.training:
                # Classic Shazeer: L_aux = N * sum_i(f_i * P_i)
                # f_i = fraction of tokens routed to expert i
                # P_i = mean gate probability for expert i
                f_i = (gates > 0).float().mean(0)   # (n_experts,)
                P_i = F.softmax(logits, dim=-1).mean(0)  # (n_experts,)
                aux_loss = self.aux_loss_coef * self.n_experts * (f_i * P_i).sum()
        
        return gates, indices, aux_loss


class MoEPredictor(nn.Module):
    """
    MoE Predictor for one JEPA level.
    
    Maps: (context_state, latent_z) -> predicted_next_state
    
    Architecture:
        1. Input projection (d_in -> d_model)
        2. RMSNorm
        3. MoE Router -> select top-k experts
        4. Run selected experts (with optional FiLM from z)
        5. Weighted sum of expert outputs
        6. Output projection (d_model -> d_out)
    
    Args:
        d_in:         input dimension (output of temporal encoder)
        d_out:        output dimension (target embedding space)
        d_model:      internal MoE hidden dim
        n_experts:    number of experts
        top_k:        number of active experts per token
        expert_type:  'ffn' (SwiGLU) or 'transformer' (2-layer)
        d_z:          latent variable dimension (0 = no latent)
        routing_mode: 'topk' or 'softmax'
    """
    
    def __init__(
        self,
        d_in:         int,
        d_out:        int,
        d_model:      int = 256,
        n_experts:    int = 4,
        top_k:        int = 2,
        expert_type:  str = "ffn",
        d_z:          int = 0,
        routing_mode: str = "topk",
        n_heads:      int = 4,
        d_ff:         int = None,
        dropout:      float = 0.0,
        use_aux_loss: bool = True,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.d_in    = d_in
        self.d_out   = d_out
        self.d_model = d_model
        self.d_z     = d_z
        self.top_k   = top_k
        
        # Input projection + norm
        self.input_proj = nn.Linear(d_in, d_model, bias=False)
        self.input_norm = nn.RMSNorm(d_model)
        
        # FiLM conditioner for latent variable
        self.film = FiLMConditioner(d_z, d_model) if d_z > 0 else None
        
        # Router
        self.router = MoERouter(
            d_model, n_experts, top_k, routing_mode,
            use_aux_loss, aux_loss_coef
        )
        
        # Experts
        if expert_type == "ffn":
            self.experts = nn.ModuleList([
                SwiGLUExpert(d_model, d_ff, dropout) for _ in range(n_experts)
            ])
        elif expert_type == "transformer":
            self.experts = nn.ModuleList([
                TransformerExpert(d_model, n_heads, d_ff, dropout)
                for _ in range(n_experts)
            ])
        else:
            raise ValueError(f"Unknown expert_type: {expert_type}")
        
        # Output
        self.output_norm = nn.RMSNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_out, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
    
    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            s: context state (B, d_in) or (B, T, d_in) for transformer experts
            z: latent variable (B, d_z) or None
        Returns:
            s_hat: predicted state (B, d_out)
            aux_loss: scalar load-balancing loss or None
        """
        # Handle sequence input for transformer experts
        is_sequence = s.dim() == 3
        if is_sequence:
            B, T, _ = s.shape
            s_flat = s.reshape(B * T, -1)
        else:
            s_flat = s
        
        # Project input
        h = self.input_norm(self.input_proj(s_flat))  # (B', d_model)
        
        # Apply FiLM conditioning from latent variable
        if self.film is not None and z is not None:
            z_flat = z.reshape(-1, z.shape[-1]) if is_sequence else z
            h = self.film(h, z_flat)
        
        # Route
        gates, indices, aux_loss = self.router(h)  # gates: (B', n_experts)
        
        # Run active experts and aggregate
        output = torch.zeros_like(h)
        for expert_idx, expert in enumerate(self.experts):
            # Mask for tokens routed to this expert
            expert_mask = (gates[:, expert_idx] > 0)
            if not expert_mask.any():
                continue
            
            h_expert = h[expert_mask]   # (M, d_model)
            if isinstance(expert, TransformerExpert):
                # Transformer expert expects (M, 1, d_model) — local window
                h_expert = h_expert.unsqueeze(1)
                out = expert(h_expert).squeeze(1)
            else:
                out = expert(h_expert)   # (M, d_model)
            
            output[expert_mask] += gates[expert_mask, expert_idx:expert_idx+1] * out
        
        # Output
        out_h = self.output_norm(output)
        s_hat = self.output_proj(out_h)   # (B', d_out)
        
        # Reshape back if sequence
        if is_sequence:
            s_hat = s_hat.reshape(B, T, -1)
        
        return s_hat, aux_loss
    
    def get_routing_stats(self, s: torch.Tensor) -> dict:
        """Debug helper: returns per-expert routing statistics."""
        with torch.no_grad():
            h = self.input_norm(self.input_proj(s))
            gates, indices, _ = self.router(h)
            expert_usage = (gates > 0).float().mean(0)
            return {
                "expert_usage": expert_usage.cpu().numpy(),
                "mean_gate_value": gates[gates > 0].mean().item(),
                "active_fraction": (gates > 0).float().mean().item(),
            }
