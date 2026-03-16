"""
H-JEPA-MoE: Probe-only / architectural test mode with frozen V-JEPA 2.

Usage (no training required):

    from hjepa_moe.model_vjepa2 import load_probe_model, extract_all_levels
    import torch

    # 1. Load V-JEPA 2 from facebookresearch/vjepa2 (MIT license)
    vjepa2 = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vitl')

    # 2. Build H-JEPA-MoE with frozen V-JEPA 2 as Level 0 encoder
    model = load_probe_model(vjepa2)

    # 3. Run forward on real video — zero training needed
    video = torch.randn(1, 64, 3, 256, 256)   # (B, T, C, H, W)
    states = extract_all_levels(model, video)

    # states[0]: (1, 64, 1024->1024) raw V-JEPA 2 patch tokens
    # states[1]: (1, 16, 1024)       level 1 — motion scale (4 frames pooled)
    # states[2]: (1, 4,  1024)       level 2 — action scale (16 frames pooled)
    # states[3]: (1, 1,  1024)       level 3 — goal scale   (64 frames pooled)

What you can test architecturally without any training:
  - Geometry of V-JEPA 2 representations at each temporal scale
  - Routing behavior of uninitialised MoE predictors (baseline entropy)
  - CEM planning feasibility in V-JEPA 2 latent space
  - Whether temporal pooling preserves or destroys geometric structure
  - Cosine similarity / CKA between levels
  - Nearest-neighbor retrieval across levels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from hjepa_moe.encoders.temporal import TemporalEncoder
from hjepa_moe.predictors.moe_predictor import MoEPredictor
from hjepa_moe.planners.cem import CEMPlanner
from hjepa_moe.utils import routing_entropy
import numpy as np


# ── V-JEPA 2 constants ────────────────────────────────────────
VJEPA2_VITL_DIM = 1024   # ViT-L output dim
VJEPA2_VITH_DIM = 1280   # ViT-H output dim (if using ViT-H variant)


# ── Config for probe mode ─────────────────────────────────────

@dataclass
class ProbeConfig:
    """
    Configuration for architectural probe experiments.
    All encoders are frozen V-JEPA 2 representations.
    Only TemporalEncoders + MoEPredictors have learnable params
    (but we don't train them here — we just test the forward pass).
    """
    d_model:       int   = VJEPA2_VITL_DIM   # match V-JEPA 2 ViT-L
    pool_factors:  List[int] = None           # temporal compression per level
    n_experts:     List[int] = None           # MoE experts per level
    top_k:         List[int] = None           # active experts per level
    expert_types:  List[str] = None           # 'ffn' or 'transformer' per level
    d_z:           List[int] = None           # latent dim per level (0 = disabled)

    def __post_init__(self):
        # Defaults: 3 levels, pool 4x each → 4^3 = 64 total compression
        if self.pool_factors is None:
            self.pool_factors = [4, 4, 4]
        if self.n_experts is None:
            self.n_experts = [4, 4, 2]
        if self.top_k is None:
            self.top_k = [2, 2, 1]
        if self.expert_types is None:
            self.expert_types = ["ffn", "transformer", "transformer"]
        if self.d_z is None:
            self.d_z = [0, 32, 64]

    @property
    def n_levels(self) -> int:
        return len(self.pool_factors)

    @classmethod
    def vjepa2_vitl_3level(cls) -> "ProbeConfig":
        """Standard 3-level config on top of V-JEPA 2 ViT-L."""
        return cls(d_model=VJEPA2_VITL_DIM)

    @classmethod
    def vjepa2_vitl_2level_fast(cls) -> "ProbeConfig":
        """2-level config for faster iteration."""
        return cls(
            d_model      = VJEPA2_VITL_DIM,
            pool_factors = [4, 4],
            n_experts    = [4, 2],
            top_k        = [2, 1],
            expert_types = ["ffn", "transformer"],
            d_z          = [0, 32],
        )


# ── Main probe model ──────────────────────────────────────────

class HJEPAMoEProbe(nn.Module):
    """
    H-JEPA-MoE in probe / architectural test mode.

    Level 0 encoder: frozen V-JEPA 2 ViT-L (no grad, no update).
    Levels 1..L:     TemporalEncoder + MoEPredictor (randomly initialized,
                     can be trained later but not required for probe tests).

    Key methods:
        get_level_states(video)     → list of (B, T_ℓ, d) per level
        routing_profile(video)      → per-level expert usage stats
        cosine_sim_across_levels()  → geometric alignment between levels
        plan(s0, s_goal, level)     → CEM planning in latent space
        probe_nearest_neighbors()   → retrieval test
    """

    def __init__(self, vjepa2_model: nn.Module, config: ProbeConfig):
        super().__init__()
        self.config = config

        # ── Level 0: frozen V-JEPA 2 ─────────────────────────
        self.enc0 = vjepa2_model
        for p in self.enc0.parameters():
            p.requires_grad_(False)
        self.enc0.eval()

        # Optional projection if V-JEPA 2 dim != d_model
        # (V-JEPA 2 ViT-L = 1024, our d_model = 1024 by default → identity)
        vjepa2_dim = self._probe_vjepa2_dim()
        if vjepa2_dim != config.d_model:
            self.level0_proj = nn.Sequential(
                nn.Linear(vjepa2_dim, config.d_model, bias=False),
                nn.RMSNorm(config.d_model),
            )
        else:
            self.level0_proj = nn.Identity()

        # ── Levels 1..L: TemporalEncoder + MoEPredictor ──────
        self.temporal_encoders = nn.ModuleList()
        self.moe_predictors    = nn.ModuleList()

        for ℓ in range(config.n_levels):
            enc = TemporalEncoder(
                d_in        = config.d_model,
                d_out       = config.d_model,
                pool_factor = config.pool_factors[ℓ],
                pooling     = "attention",
            )
            pred = MoEPredictor(
                d_in        = config.d_model,
                d_out       = config.d_model,
                d_model     = config.d_model,
                n_experts   = config.n_experts[ℓ],
                top_k       = config.top_k[ℓ],
                expert_type = config.expert_types[ℓ],
                d_z         = config.d_z[ℓ],
            )
            self.temporal_encoders.append(enc)
            self.moe_predictors.append(pred)

    # ── Core: encode video through hierarchy ─────────────────

    @torch.no_grad()
    def encode_level0(self, video: torch.Tensor) -> torch.Tensor:
        """
        Run frozen V-JEPA 2 on video frames.

        Args:
            video: (B, T, C, H, W)  — T frames, typically 256x256
        Returns:
            tokens: (B, T, d_model)  — one embedding per frame
                    (spatial patch tokens averaged → single frame vector)
        """
        self.enc0.eval()
        B, T, C, H, W = video.shape

        # V-JEPA 2 expects (B, C, T, H, W) or (B*T, C, H, W) depending on version
        # We use frame-by-frame encoding and pool spatial patches
        frames = video.reshape(B * T, C, H, W)

        # V-JEPA 2 ViT forward: returns (B*T, N_patches, d)
        try:
            tokens = self.enc0(frames)                         # (B*T, N, d)
        except Exception:
            # Some V-JEPA 2 variants expect video shape — try that
            tokens = self.enc0(video.permute(0, 2, 1, 3, 4))  # (B, C, T, H, W)
            if tokens.dim() == 4:   # (B, T, N, d)
                tokens = tokens.reshape(B * T, -1, tokens.shape[-1])

        if tokens.dim() == 3:
            tokens = tokens.mean(dim=1)   # pool spatial: (B*T, d)

        tokens = self.level0_proj(tokens)         # (B*T, d_model)
        tokens = tokens.reshape(B, T, -1)         # (B, T, d_model)
        return tokens

    @torch.no_grad()
    def get_level_states(self, video: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract representations at all levels.

        Returns list of length n_levels + 1:
            states[0]: (B, T,       d_model)  level 0 — raw V-JEPA 2
            states[1]: (B, T/4,     d_model)  level 1 — motion scale
            states[2]: (B, T/16,    d_model)  level 2 — action scale
            states[3]: (B, T/64,    d_model)  level 3 — goal scale
        """
        states = self.encode_level0(video)
        all_states = [states]

        for enc in self.temporal_encoders:
            states = enc(states)
            all_states.append(states)

        return all_states

    # ── Geometric analysis (no training required) ────────────

    @torch.no_grad()
    def routing_profile(self, video: torch.Tensor) -> Dict[str, dict]:
        """
        Run MoE routing on uninitialised predictors and return
        per-expert usage statistics at each level.

        This is the baseline routing entropy before training.
        Uniform usage (high entropy) = good initialisation.
        Collapsed usage (low entropy) = initialisation problem.
        """
        all_states = self.get_level_states(video)
        profile = {}

        for ℓ, (pred, states) in enumerate(
            zip(self.moe_predictors, all_states[1:])
        ):
            # Use first-step context tokens
            s_flat = states[:, :-1].reshape(-1, states.shape[-1])
            stats  = pred.get_routing_stats(s_flat)
            stats["routing_entropy"] = routing_entropy(stats["expert_usage"])
            stats["n_experts"]       = self.config.n_experts[ℓ]
            stats["max_entropy"]     = float(np.log(self.config.n_experts[ℓ]))
            profile[f"level_{ℓ+1}"] = stats

        return profile

    @torch.no_grad()
    def cosine_similarity_across_levels(
        self, video: torch.Tensor
    ) -> Dict[str, float]:
        """
        Measure cosine similarity between consecutive levels
        (after projecting to same dim via mean pooling).

        High similarity = temporal pooling preserves geometry.
        Low similarity = too much information is lost per level.
        Useful diagnostic before committing to a pool_factor.
        """
        all_states = self.get_level_states(video)
        sims = {}

        for ℓ in range(len(all_states) - 1):
            s_low  = all_states[ℓ].mean(dim=1)    # (B, d) — pool time
            s_high = all_states[ℓ+1].mean(dim=1)  # (B, d)
            sim = F.cosine_similarity(
                F.normalize(s_low,  dim=-1),
                F.normalize(s_high, dim=-1),
            ).mean().item()
            sims[f"cos_sim_L{ℓ}_L{ℓ+1}"] = sim

        return sims

    @torch.no_grad()
    def nearest_neighbor_retrieval(
        self,
        query_video:    torch.Tensor,   # (1, T, C, H, W)
        database_video: torch.Tensor,   # (N, T, C, H, W)
        level:          int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Test whether level-ℓ representations support retrieval.
        Returns (ranked_indices, cosine_scores) for query against database.
        """
        q_states = self.get_level_states(query_video)[level]     # (1, T_ℓ, d)
        db_states = self.get_level_states(database_video)[level] # (N, T_ℓ, d)

        q  = F.normalize(q_states.mean(dim=1),  dim=-1)  # (1, d)
        db = F.normalize(db_states.mean(dim=1), dim=-1)  # (N, d)

        scores  = (q @ db.T).squeeze(0)  # (N,)
        ranked  = scores.argsort(descending=True)
        return ranked, scores[ranked]

    # ── Planning test ────────────────────────────────────────

    @torch.no_grad()
    def test_cem_planning(
        self,
        video:      torch.Tensor,    # (B, T, C, H, W)
        level:      int = -1,
        horizon:    int = 5,
        n_samples:  int = 50,
        n_iters:    int = 3,
    ) -> Dict[str, float]:
        """
        Test CEM planning feasibility in V-JEPA 2 latent space.
        Uses first clip as 'current state', last clip as 'goal'.
        Reports planning cost and wall time.
        """
        import time
        if level < 0:
            level = self.config.n_levels - 1

        all_states = self.get_level_states(video)
        states_ℓ = all_states[level + 1]  # +1 because index 0 = level 0

        s0     = states_ℓ[:1, 0]   # (1, d) — first timestep
        s_goal = states_ℓ[:1, -1]  # (1, d) — last timestep

        d_z = self.config.d_z[level]
        planner = CEMPlanner(
            self.moe_predictors[level],
            d_z       = d_z,
            horizon   = horizon,
            n_samples = n_samples,
            n_iters   = n_iters,
            device    = str(s0.device),
        )

        t0 = time.time()
        z_seq, cost = planner.plan(s0, s_goal)
        elapsed = time.time() - t0

        return {
            "level":          level,
            "d_z":            d_z,
            "planning_cost":  cost,
            "planning_time_s": elapsed,
            "horizon":        horizon,
        }

    # ── Helpers ───────────────────────────────────────────────

    def _probe_vjepa2_dim(self) -> int:
        """Infer V-JEPA 2 output dimension with a small probe forward pass."""
        try:
            dummy = torch.zeros(1, 3, 64, 64)
            with torch.no_grad():
                out = self.enc0(dummy)
            if out.dim() == 3:
                return out.shape[-1]
            elif out.dim() == 2:
                return out.shape[-1]
        except Exception:
            pass
        return VJEPA2_VITL_DIM   # fallback

    def summary(self) -> str:
        lines = ["HJEPAMoEProbe"]
        lines.append(f"  Level 0: V-JEPA 2 (frozen, d={self.config.d_model})")
        for ℓ in range(self.config.n_levels):
            lines.append(
                f"  Level {ℓ+1}: pool={self.config.pool_factors[ℓ]}x  "
                f"experts={self.config.n_experts[ℓ]}  top_k={self.config.top_k[ℓ]}  "
                f"d_z={self.config.d_z[ℓ]}"
            )
        n_probe = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lines.append(f"  Trainable params (probe only): {n_probe:,}")
        return "\n".join(lines)


# ── Factory functions ────────────────────────────────────────

def load_probe_model(
    vjepa2_model: nn.Module,
    config: Optional[ProbeConfig] = None,
) -> HJEPAMoEProbe:
    """
    Build H-JEPA-MoE probe model from a frozen V-JEPA 2 checkpoint.

    Args:
        vjepa2_model: loaded V-JEPA 2 model (from facebookresearch/vjepa2)
        config:       ProbeConfig (defaults to 3-level ViT-L config)
    Returns:
        HJEPAMoEProbe ready for forward passes
    """
    if config is None:
        config = ProbeConfig.vjepa2_vitl_3level()
    model = HJEPAMoEProbe(vjepa2_model, config)
    model.eval()
    return model


def extract_all_levels(
    model:  HJEPAMoEProbe,
    video:  torch.Tensor,
    device: str = "cpu",
) -> List[torch.Tensor]:
    """
    Convenience wrapper: extract all level representations.
    Handles device placement automatically.
    """
    video = video.to(device)
    model = model.to(device)
    return model.get_level_states(video)


# ── Quick diagnostic runner ──────────────────────────────────

def run_diagnostics(
    vjepa2_model: nn.Module,
    video:        torch.Tensor,
    device:       str = "cpu",
) -> None:
    """
    Full architectural diagnostic in one call.
    Prints routing entropy, cosine similarities, and planning test.

    Example:
        vjepa2 = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vitl')
        video  = torch.randn(2, 64, 3, 256, 256)
        run_diagnostics(vjepa2, video)
    """
    model = load_probe_model(vjepa2_model)
    model = model.to(device)
    video = video.to(device)

    print(model.summary())
    print()

    # 1. Level state shapes
    print("── Level state shapes ──────────────────")
    states = model.get_level_states(video)
    for ℓ, s in enumerate(states):
        print(f"  Level {ℓ}: {tuple(s.shape)}")
    print()

    # 2. Routing profile
    print("── MoE routing profile (uninitialised) ─")
    profile = model.routing_profile(video)
    for level_name, stats in profile.items():
        H     = stats["routing_entropy"]
        H_max = stats["max_entropy"]
        print(f"  {level_name}: entropy={H:.3f} / {H_max:.3f} "
              f"({'uniform' if H > 0.9 * H_max else 'collapsed'})")
        print(f"    usage: {stats['expert_usage']}")
    print()

    # 3. Cosine similarity across levels
    print("── Cosine similarity across levels ─────")
    sims = model.cosine_similarity_across_levels(video)
    for k, v in sims.items():
        print(f"  {k}: {v:.4f}")
    print()

    # 4. Planning test (level with d_z > 0)
    for ℓ in range(model.config.n_levels):
        if model.config.d_z[ℓ] > 0:
            print(f"── CEM planning test at level {ℓ+1} ─────────")
            result = model.test_cem_planning(video, level=ℓ, horizon=3,
                                             n_samples=20, n_iters=2)
            for k, v in result.items():
                print(f"  {k}: {v}")
            break
    print()
    print("Diagnostics complete.")
