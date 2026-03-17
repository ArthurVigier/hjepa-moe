"""
cbs_network.py
==============
Architecture du Chaotic Background Subnetwork (CBS).

Design :
    - MLP résiduel simple (pas de attention, volontairement sobre)
    - Poids mis à jour UNIQUEMENT via perturbation chaotique + entropie max loss
    - JAMAIS dans le graphe de gradient du réseau principal
    - Gate de corrélation cosinus asymétrique pour la consultation

Invariant central :
    Tout ce qui sort du CBS vers le réseau principal passe par .detach().
    Il n'existe aucun chemin de gradient du loss principal vers les poids du CBS.

Analogie :
    Le CBS est comme le Default Mode Network — actif quand le réseau
    principal ne "demande" pas, consultable mais sans rétroaction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple
import math

from .chaotic_dynamics import ChaoticDynamicsFactory


@dataclass
class CBSConfig:
    # Dimensions
    input_dim: int = 256            # Doit matcher la dim du JEPA latent space
    hidden_dim: int = 512           # CBS peut être plus large que le main network
    output_dim: int = 256           # Doit matcher input_dim pour la gate cosinus
    n_layers: int = 3

    # Dynamique chaotique
    attractor: str = "lorenz"
    # Options :
    #   "lorenz"                 → R³, 1 exposant positif, exploration large
    #   "rossler"                → R³, 1 exposant positif, exploration fine
    #   "rossler_hyper"          → R⁴, 2 exposants positifs (hyperchaos)
    #   "rossler_hyper_per_dim"  → N×R⁴, ~2N exposants positifs (stress-test)
    chaos_noise_scale: float = 1e-3 # Amplitude de perturbation des poids
    chaos_dt: float = 0.01
    perturb_every_n_steps: int = 10 # Fréquence de perturbation chaotique

    # Gate de consultation
    cosine_threshold: float = 0.3   # Seuil au-dessous duquel la gate est fermée
    gate_alpha: float = 0.1         # Amplitude de l'influence CBS sur main activations

    # Objectif propre du CBS : maximisation d'entropie des activations
    entropy_loss_weight: float = 0.01
    cbs_lr: float = 1e-4            # Optimizer propre du CBS

    # Seed
    seed: int = 42


class CBSBlock(nn.Module):
    """Bloc résiduel du CBS. Pre-LayerNorm, activation SiLU."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.SiLU()

        # Init légèrement bruitée — le CBS commence déjà dans un état non-standard
        nn.init.normal_(self.fc1.weight, mean=0, std=0.02)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x + residual


class CBSNetwork(nn.Module):
    """
    Chaotic Background Subnetwork.

    Ne PAS appeler .backward() sur les outputs de ce réseau depuis
    le loss du réseau principal. Utiliser uniquement cbs.read(x) qui
    retourne des activations détachées.

    Le CBS a son propre optimizer et son propre objectif (entropie max).
    """

    def __init__(self, cfg: CBSConfig = None):
        super().__init__()
        if cfg is None:
            cfg = CBSConfig()
        self.cfg = cfg
        self._step_count = 0

        # Architecture
        self.input_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.blocks = nn.ModuleList([
            CBSBlock(cfg.hidden_dim, cfg.hidden_dim * 2)
            for _ in range(cfg.n_layers)
        ])
        self.output_proj = nn.Linear(cfg.hidden_dim, cfg.output_dim)
        self.output_norm = nn.LayerNorm(cfg.output_dim)

        # Dynamique chaotique
        self.dynamics = ChaoticDynamicsFactory.build(
            attractor=cfg.attractor,
            dim=cfg.hidden_dim,   # Perturbation dans l'espace hidden du CBS
            noise_scale=cfg.chaos_noise_scale,
            seed=cfg.seed,
        )

        # Optimizer propre — indépendant de celui du réseau principal
        self._optimizer = torch.optim.AdamW(
            self.parameters(), lr=cfg.cbs_lr, weight_decay=0.0
        )

        # Stats pour monitoring
        self.register_buffer("_gate_activations", torch.tensor(0.0))
        self.register_buffer("_gate_total", torch.tensor(0.0))
        self.register_buffer("_mean_cosine_sim", torch.tensor(0.0))

    # ─────────────────────────────────────────────
    # Forward interne (avec gradient, pour CBS loss)
    # ─────────────────────────────────────────────

    def _forward_internal(self, x: torch.Tensor) -> torch.Tensor:
        """Forward avec gradient — UNIQUEMENT pour le CBS loss interne."""
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        out = self.output_proj(h)
        out = self.output_norm(out)
        return out

    # ─────────────────────────────────────────────
    # Interface publique : lecture asymétrique
    # ─────────────────────────────────────────────

    @torch.no_grad()
    def read(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lit les activations du CBS sans aucun gradient.
        C'est la SEULE façon dont le réseau principal accède au CBS.

        Returns:
            Tensor (batch, output_dim) — toujours détaché
        """
        out = self._forward_internal(x)
        return out.detach()  # INVARIANT : jamais de gradient vers le CBS depuis main

    def consult(
        self,
        main_activations: torch.Tensor,
        x: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Gate de corrélation cosinus asymétrique.

        Le CBS est consulté uniquement si la similarité cosinus entre
        les activations du réseau principal et celles du CBS dépasse le seuil.

        Args:
            main_activations: activations du réseau principal, shape (B, D)
            x: input original (optionnel, pour calculer les activations CBS)
               Si None, on réutilise main_activations comme input CBS

        Returns:
            enriched: main_activations enrichies (ou non), shape (B, D)
            info: dict de monitoring
        """
        cbs_input = x if x is not None else main_activations
        cbs_acts = self.read(cbs_input)  # (B, D), toujours détaché

        # Similarité cosinus par sample
        cos_sim = F.cosine_similarity(
            main_activations.detach(), cbs_acts, dim=-1
        )  # (B,)

        # Gate binaire par sample (ou soft gate)
        gate = (cos_sim.abs() > self.cfg.cosine_threshold).float()  # (B,)
        gate = gate.unsqueeze(-1)  # (B, 1) pour broadcast

        # Enrichissement — le CBS ne contribue que là où la gate est ouverte
        # .detach() sur cbs_acts est déjà garanti par read()
        enriched = main_activations + gate * self.cfg.gate_alpha * cbs_acts

        # Stats
        with torch.no_grad():
            self._gate_activations += gate.sum()
            self._gate_total += gate.numel()
            self._mean_cosine_sim = cos_sim.mean()

        info = {
            "gate_ratio": gate.mean().item(),
            "mean_cosine_sim": cos_sim.mean().item(),
            "cbs_norm": cbs_acts.norm(dim=-1).mean().item(),
        }

        return enriched, info

    # ─────────────────────────────────────────────
    # Update interne du CBS
    # ─────────────────────────────────────────────

    def cbs_update_step(self, x: torch.Tensor) -> dict:
        """
        Un pas d'update du CBS :
        1. Calcul du loss d'entropie maximale (le CBS veut être "surprenant")
        2. Perturbation chaotique des poids
        3. Mise à jour via son optimizer propre

        Ne PAS appeler pendant le backward du réseau principal.
        Appeler séparément, idéalement en mode idle ou entre batches.

        Args:
            x: input représentatif (batch ou sous-batch)

        Returns:
            dict de métriques
        """
        self._optimizer.zero_grad()

        # Forward CBS avec gradient (pour son propre loss)
        acts = self._forward_internal(x)

        # Loss d'entropie maximale sur les activations
        # Objectif : maximiser H(softmax(acts)) — le CBS veut couvrir l'espace
        entropy_loss = -self._activation_entropy(acts)
        loss = self.cfg.entropy_loss_weight * entropy_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self._optimizer.step()

        # Perturbation chaotique périodique
        self._step_count += 1
        if self._step_count % self.cfg.perturb_every_n_steps == 0:
            self._apply_chaotic_perturbation()

        return {
            "cbs_entropy_loss": entropy_loss.item(),
            "cbs_total_loss": loss.item(),
            "chaos_lyapunov_proxy": self.dynamics.lyapunov_proxy()
            if hasattr(self.dynamics, "lyapunov_proxy") else 0.0,
        }

    def _activation_entropy(self, acts: torch.Tensor) -> torch.Tensor:
        """
        Entropie des activations softmax sur le batch.
        Maximiser ceci = le CBS produit des distributions diversifiées.
        """
        # acts: (B, D)
        probs = F.softmax(acts, dim=-1)  # (B, D)
        log_probs = F.log_softmax(acts, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # (B,)
        return entropy.mean()

    def _apply_chaotic_perturbation(self):
        """
        Injecte la perturbation chaotique directement dans les poids
        du CBS. La direction vient de l'attracteur (Lorenz/Rössler),
        projetée dans l'espace des poids de chaque couche.
        """
        perturb = self.dynamics.step()  # (hidden_dim,)

        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad and param.dim() >= 2:
                    # Perturbation outer product : direction chaotique × vecteur aléatoire
                    if param.shape[-1] == self.cfg.hidden_dim:
                        # Perturbation cohérente avec la direction chaotique
                        p_vec = perturb.to(param.device)
                        noise = torch.randn(param.shape[0], device=param.device)
                        delta = torch.outer(noise, p_vec) * self.cfg.chaos_noise_scale
                        if delta.shape == param.shape:
                            param.add_(delta)
                    else:
                        # Perturbation isotrope pour les autres couches
                        param.add_(
                            torch.randn_like(param) * self.cfg.chaos_noise_scale * 0.1
                        )

    # ─────────────────────────────────────────────
    # Monitoring
    # ─────────────────────────────────────────────

    def get_stats(self) -> dict:
        gate_ratio = (
            (self._gate_activations / (self._gate_total + 1e-8)).item()
        )
        return {
            "gate_open_ratio": gate_ratio,
            "mean_cosine_sim": self._mean_cosine_sim.item(),
            "chaos_state_norm": self.dynamics.get_state().norm().item(),
            "n_steps": self._step_count,
        }

    def reset_stats(self):
        self._gate_activations.zero_()
        self._gate_total.zero_()
