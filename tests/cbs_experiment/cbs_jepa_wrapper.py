"""
cbs_jepa_wrapper.py
====================
Wrapper JEPA générique + CBS.

Design :
    - JEPAWithCBS wrape n'importe quel JEPA conforme à l'interface MinimalJEPA
    - Le CBS est branché via le régime choisi
    - L'interface est compatible avec hjepa_moe

Interface MinimalJEPA attendue :
    jepa.encode(x) -> context_embeddings     # (B, D)
    jepa.predict(context_emb) -> predictions # (B, D)
    jepa.target_encode(x) -> targets         # (B, D) — EMA ou frozen
    jepa.loss(predictions, targets) -> scalar

Si ton JEPA ne suit pas exactement cette interface, sous-classe
JEPAWithCBS et override `_jepa_forward`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Any
import math

from .cbs_network import CBSNetwork, CBSConfig
from .cbs_regimes import CBSRegime, RegimeFactory, RegimeConfig, compute_activation_entropy


@dataclass
class JEPAWithCBSConfig:
    cbs: CBSConfig = field(default_factory=CBSConfig)
    regime: str = "C"               # "A" | "B" | "C"
    regime_cfg: RegimeConfig = field(default_factory=RegimeConfig)

    # Logging
    log_every_n_steps: int = 50


class MinimalJEPA(nn.Module):
    """
    Interface JEPA minimale pour les tests.
    Dans la pratique, remplace par ton HJEPAMoE ou tout JEPA hjepa_moe-compatible.

    Architecture : deux encodeurs MLP + predictor MLP.
    Pas de masquage sophistiqué — just the bare minimum pour tester le CBS.
    """

    def __init__(self, dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.dim = dim

        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim),
        )

        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim),
        )

        # Target encoder (EMA du context encoder)
        self.target_encoder = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim),
        )

        # Copie les poids initiaux dans le target encoder
        self._copy_context_to_target()

        # EMA momentum
        self.ema_momentum = 0.996

    def _copy_context_to_target(self):
        for p_ctx, p_tgt in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            p_tgt.data.copy_(p_ctx.data)
            p_tgt.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update du target encoder."""
        m = self.ema_momentum
        for p_ctx, p_tgt in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            p_tgt.data.mul_(m).add_((1 - m) * p_ctx.data)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.context_encoder(x)

    def predict(self, context_emb: torch.Tensor) -> torch.Tensor:
        return self.predictor(context_emb)

    @torch.no_grad()
    def target_encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(x)

    def loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # VICReg-style : invariance + variance + covariance
        # Simplifié ici : juste smooth L1 + variance regularization
        inv_loss = F.smooth_l1_loss(predictions, targets)
        var_loss = torch.relu(1 - predictions.std(dim=0)).mean()
        return inv_loss + 0.1 * var_loss


class JEPAWithCBS(nn.Module):
    """
    Wrapper qui ajoute un CBS à n'importe quel JEPA.

    Usage:
        jepa = MinimalJEPA(dim=256)
        cfg = JEPAWithCBSConfig(regime="C")
        model = JEPAWithCBS(jepa, cfg)

        # Training step
        loss, info = model.training_step(x_context, x_target)

        # Evaluation (CBS toujours consulté)
        emb, info = model.encode_with_cbs(x)
    """

    def __init__(self, jepa: nn.Module, cfg: JEPAWithCBSConfig = None):
        super().__init__()
        self.cfg = cfg or JEPAWithCBSConfig()
        self.jepa = jepa

        # CBS
        self.cbs = CBSNetwork(self.cfg.cbs)

        # Régime
        self.regime = RegimeFactory.build(self.cfg.regime, self.cfg.regime_cfg)

        # Step counter pour logging
        self._step = 0
        self._log_buffer = []

    def training_step(
        self,
        x_context: torch.Tensor,
        x_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Un step complet JEPA + CBS.

        1. Encode context → context_emb
        2. (optionnel) Enrichir context_emb via CBS selon le régime
        3. Predict → predictions
        4. Target encode → targets
        5. Loss JEPA
        6. (optionnel) CBS update step

        Returns:
            loss: scalar tensor (avec gradient, pour .backward() du main network)
            info: dict de métriques
        """
        self.regime.tick()
        info = {}

        # ── 1. Encode context ──────────────────────────────────────────────
        context_emb = self.jepa.encode(x_context)  # (B, D)

        # ── 2. Consultation CBS (si le régime l'autorise) ──────────────────
        main_entropy = compute_activation_entropy(context_emb)
        info["main_entropy"] = main_entropy

        cbs_intensity = self.regime.get_cbs_intensity(main_entropy)
        info["cbs_intensity"] = cbs_intensity

        if self.regime.should_consult_cbs(main_entropy):
            # Modulation de gate_alpha par l'intensité du régime
            original_alpha = self.cbs.cfg.gate_alpha
            self.cbs.cfg.gate_alpha = original_alpha * cbs_intensity

            enriched_emb, cbs_info = self.cbs.consult(context_emb, x_context)
            info.update({f"cbs_{k}": v for k, v in cbs_info.items()})

            self.cbs.cfg.gate_alpha = original_alpha  # restore
        else:
            enriched_emb = context_emb
            info["cbs_gate_ratio"] = 0.0

        # ── 3. Predict ─────────────────────────────────────────────────────
        predictions = self.jepa.predict(enriched_emb)

        # ── 4. Target encode (no grad) ─────────────────────────────────────
        with torch.no_grad():
            targets = self.jepa.target_encode(x_target)

        # ── 5. Loss JEPA ───────────────────────────────────────────────────
        jepa_loss = self.jepa.loss(predictions, targets)
        info["jepa_loss"] = jepa_loss.item()

        # ── 6. CBS update step (séparé du graphe JEPA) ────────────────────
        if self.regime.should_run_cbs_update(main_entropy):
            # Le CBS update est COMPLÈTEMENT séparé du graphe JEPA
            # On utilise x_context comme signal d'input pour le CBS
            with torch.no_grad():
                cbs_input = x_context.detach()
            cbs_metrics = self.cbs.cbs_update_step(cbs_input)
            info.update({f"cbs_update_{k}": v for k, v in cbs_metrics.items()})

        # ── 7. EMA update target encoder ──────────────────────────────────
        if hasattr(self.jepa, "update_target_encoder"):
            self.jepa.update_target_encoder()

        self._step += 1

        # Logging périodique
        if self._step % self.cfg.log_every_n_steps == 0:
            cbs_stats = self.cbs.get_stats()
            info.update({f"cbs_stat_{k}": v for k, v in cbs_stats.items()})
            info["regime"] = self.regime.describe()

        return jepa_loss, info

    @torch.no_grad()
    def encode_with_cbs(
        self,
        x: torch.Tensor,
        force_consult: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Encode avec consultation CBS forcée (pour évaluation).
        En régime A, le CBS peut être consulté ici même s'il ne l'est pas
        pendant le training.

        Args:
            x: input tensor (B, D)
            force_consult: si True, consulte le CBS indépendamment du régime

        Returns:
            enriched_emb: (B, D)
            info: dict
        """
        context_emb = self.jepa.encode(x)

        if force_consult or self.regime.should_consult_cbs():
            enriched_emb, cbs_info = self.cbs.consult(context_emb, x)
        else:
            enriched_emb = context_emb
            cbs_info = {}

        return enriched_emb, cbs_info

    def get_full_stats(self) -> Dict:
        return {
            "step": self._step,
            "regime": self.cfg.regime,
            **self.cbs.get_stats(),
        }
