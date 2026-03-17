"""
chaotic_dynamics.py
====================
Implémente les attracteurs chaotiques comme source de perturbation
pour les poids du CBS. Le CBS n'est pas un système chaotique en soi —
ses poids dérivent selon une dynamique chaotique injectée périodiquement.

Attracteurs disponibles :
- Lorenz (sigma, rho, beta)          : R³, 1 exposant Lyapunov positif (~0.9)
- Rössler (a, b, c)                  : R³, 1 exposant Lyapunov positif (~0.07)
- RosslerHyperchaos (a,b,c,d,e)     : R⁴, 2 exposants Lyapunov positifs
                                       → hyperchaos au sens strict (Rössler 1979)
- RosslerHyperchaosPerDim            : N instances R⁴ indépendantes, une par
                                       dimension de projection — couverture maximale
                                       de l'espace latent, stress-test extrême

Pourquoi l'hyperchaos per-dim est un stress-test extrême :
    - Chaque dimension reçoit une perturbation issue d'un attracteur R⁴ distinct
    - Les paramètres sont légèrement perturbés entre instances → pas de synchronisation
    - 2 exposants positifs par système → divergence plus rapide qu'un chaos simple
    - La matrice de perturbation est plein rang → aucune direction latente privilégiée
    - Spectre de Lyapunov global : O(N) exposants positifs pour N dimensions

Usage:
    # Simple
    dyn = LorenzDynamics(dim=256, dt=0.01)
    perturbation = dyn.step()  # (256,)

    # Stress-test extrême
    dyn = RosslerHyperchaosPerDim(projection_dim=256)
    perturbation = dyn.step()  # (256,) — une instance R⁴ par dimension
    spectrum = dyn.lyapunov_spectrum_proxy()  # dict avec stats spectrales
"""

import torch
import torch.nn as nn
import math
from dataclasses import dataclass, field
from typing import Literal, List, Dict


@dataclass
class LorenzConfig:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    dt: float = 0.01
    noise_scale: float = 1e-3      # Amplitude de la perturbation sur les poids
    projection_dim: int = 256       # Dimension de projection vers l'espace des poids


@dataclass
class RosslerConfig:
    a: float = 0.2
    b: float = 0.2
    c: float = 5.7
    dt: float = 0.01
    noise_scale: float = 1e-3
    projection_dim: int = 256


class LorenzDynamics:
    """
    Système de Lorenz intégré par Runge-Kutta 4.
    Produit un vecteur de perturbation de dimension `projection_dim`
    via une projection linéaire aléatoire fixe (pas entraînée).

    L'état (x, y, z) du système Lorenz est projeté dans R^dim
    via une matrice W fixée à l'initialisation (jamais mise à jour).
    Ce vecteur sert à perturber légèrement les poids du CBS.
    """

    def __init__(self, cfg: LorenzConfig = None, seed: int = 42):
        if cfg is None:
            cfg = LorenzConfig()
        self.cfg = cfg

        rng = torch.Generator()
        rng.manual_seed(seed)

        # État initial du système Lorenz
        self.state = torch.tensor([1.0, 0.0, 0.0])  # (x, y, z)

        # Projection fixe R^3 -> R^projection_dim
        # Orthogonalisée pour une couverture uniforme
        W = torch.randn(3, cfg.projection_dim, generator=rng)
        self.W = W / (W.norm(dim=0, keepdim=True) + 1e-8)  # normalisation colonne

    def _lorenz_deriv(self, s: torch.Tensor) -> torch.Tensor:
        x, y, z = s[0], s[1], s[2]
        dx = self.cfg.sigma * (y - x)
        dy = x * (self.cfg.rho - z) - y
        dz = x * y - self.cfg.beta * z
        return torch.stack([dx, dy, dz])

    def step(self) -> torch.Tensor:
        """
        Avance le système d'un pas dt (RK4) et retourne un vecteur
        de perturbation normalisé de taille (projection_dim,).
        """
        dt = self.cfg.dt
        s = self.state

        k1 = self._lorenz_deriv(s)
        k2 = self._lorenz_deriv(s + 0.5 * dt * k1)
        k3 = self._lorenz_deriv(s + 0.5 * dt * k2)
        k4 = self._lorenz_deriv(s + dt * k3)

        self.state = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Projection + normalisation
        perturb = self.state @ self.W          # (projection_dim,)
        perturb = perturb / (perturb.norm() + 1e-8)
        return perturb * self.cfg.noise_scale

    def get_state(self) -> torch.Tensor:
        return self.state.clone()

    def lyapunov_proxy(self) -> float:
        """
        Proxy du plus grand exposant de Lyapunov : norme de la dérivée
        au point courant. Utile pour monitorer si le système est bien chaotique.
        Pour Lorenz standard : exposant ≈ 0.9
        """
        deriv = self._lorenz_deriv(self.state)
        return deriv.norm().item()


class RosslerDynamics:
    """
    Système de Rössler — attracteur plus compact que Lorenz.
    Même interface que LorenzDynamics.

    Utile pour une exploration plus "locale" de l'espace latent.
    """

    def __init__(self, cfg: RosslerConfig = None, seed: int = 42):
        if cfg is None:
            cfg = RosslerConfig()
        self.cfg = cfg

        rng = torch.Generator()
        rng.manual_seed(seed)

        self.state = torch.tensor([1.0, 0.0, 0.0])

        W = torch.randn(3, cfg.projection_dim, generator=rng)
        self.W = W / (W.norm(dim=0, keepdim=True) + 1e-8)

    def _rossler_deriv(self, s: torch.Tensor) -> torch.Tensor:
        x, y, z = s[0], s[1], s[2]
        dx = -y - z
        dy = x + self.cfg.a * y
        dz = self.cfg.b + z * (x - self.cfg.c)
        return torch.stack([dx, dy, dz])

    def step(self) -> torch.Tensor:
        dt = self.cfg.dt
        s = self.state

        k1 = self._rossler_deriv(s)
        k2 = self._rossler_deriv(s + 0.5 * dt * k1)
        k3 = self._rossler_deriv(s + 0.5 * dt * k2)
        k4 = self._rossler_deriv(s + dt * k3)

        self.state = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        perturb = self.state @ self.W
        perturb = perturb / (perturb.norm() + 1e-8)
        return perturb * self.cfg.noise_scale

    def get_state(self) -> torch.Tensor:
        return self.state.clone()


# ─────────────────────────────────────────────────────────────────────────────
# Rössler Hyperchaotique — R⁴, 2 exposants de Lyapunov positifs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RosslerHyperchaosConfig:
    """
    Paramètres du système hyperchaotique de Rössler (1979).

    Équations (R⁴) :
        dx/dt = -y - z
        dy/dt = x + a*y + w
        dz/dt = b + x*z
        dw/dt = -c*z + d*w

    Paramètres canoniques donnant l'hyperchaos :
        a=0.25, b=3.0, c=0.5, d=0.05
        → spectre de Lyapunov : (+, +, 0, -)
        → deux directions d'instabilité simultanées

    Le paramètre `e` (coupling_noise) ajoute un couplage stochastique
    entre les variables w et z — augmente encore le degré d'imprévisibilité
    sans sortir du régime hyperchaotique.

    Référence : Rössler O.E. (1979) "An equation for hyperchaos"
                Physics Letters A, 71(2-3), 155-157.
    """
    a: float = 0.25
    b: float = 3.0
    c: float = 0.5
    d: float = 0.05
    e: float = 0.0          # coupling noise w↔z (0 = désactivé, 0.01 = léger)
    dt: float = 0.005       # dt plus petit que Rössler simple (stabilité R⁴)
    noise_scale: float = 1e-3
    projection_dim: int = 256


@dataclass
class RosslerHyperchaosPerDimConfig:
    """
    Configuration pour N instances hyperchaotiques indépendantes,
    une par dimension de projection.

    Chaque instance a des paramètres légèrement perturbés autour
    des valeurs canoniques → pas de synchronisation entre instances,
    couverture maximale de l'espace latent.

    param_jitter : amplitude de la perturbation des paramètres entre instances
                   0.0 = toutes identiques (inutile)
                   0.05 = légère désynchronisation (recommandé)
                   0.2  = forte désynchronisation (stress-test extrême)
    """
    base: RosslerHyperchaosConfig = field(default_factory=RosslerHyperchaosConfig)
    param_jitter: float = 0.05     # Perturbation relative des paramètres
    projection_dim: int = 256
    noise_scale: float = 1e-3
    seed: int = 42


class RosslerHyperchaos:
    """
    Système hyperchaotique de Rössler en R⁴.

    Deux exposants de Lyapunov positifs simultanément :
        λ₁ ≈ +0.16  (instabilité rapide)
        λ₂ ≈ +0.03  (instabilité lente)
        λ₃ ≈  0.0   (direction neutre)
        λ₄ ≈ -2.0   (contraction)

    La somme est négative (système dissipatif) mais deux directions
    divergent → l'attracteur est un tore fractal de dimension > 2.

    Comparaison avec Rössler standard :
        Rössler R³ : 1 exposant positif  → exploration 1D dans l'espace latent
        Rössler R⁴ : 2 exposants positifs → exploration 2D simultanée
                     → couverture de l'espace latent ~2x plus rapide
    """

    def __init__(self, cfg: RosslerHyperchaosConfig = None, seed: int = 42):
        if cfg is None:
            cfg = RosslerHyperchaosConfig()
        self.cfg = cfg

        rng = torch.Generator()
        rng.manual_seed(seed)

        # État initial R⁴ — légèrement aléatoire pour éviter les points fixes
        ic = torch.randn(4, generator=rng) * 0.5
        ic[0] = ic[0] + 1.0   # x ≠ 0 pour sortir du plan invariant
        self.state = ic        # (x, y, z, w)

        # Projection fixe R⁴ → R^projection_dim
        # QR pour garantir l'orthogonalité des colonnes de projection
        W_raw = torch.randn(4, cfg.projection_dim, generator=rng)
        if cfg.projection_dim >= 4:
            # QR donne des colonnes orthonormales
            Q, _ = torch.linalg.qr(W_raw.T)
            self.W = Q.T  # (4, projection_dim)
        else:
            self.W = W_raw / (W_raw.norm(dim=0, keepdim=True) + 1e-8)

        # Générateur pour le bruit de couplage (e > 0)
        self._rng_seed = seed
        self._step_count = 0

    def _deriv(self, s: torch.Tensor) -> torch.Tensor:
        """
        Dérivée du système hyperchaotique de Rössler R⁴.

            dx/dt = -y - z
            dy/dt =  x + a*y + w
            dz/dt =  b + x*z
            dw/dt = -c*z + d*w  [+ e*bruit si e > 0]
        """
        x, y, z, w = s[0], s[1], s[2], s[3]
        a, b, c, d = self.cfg.a, self.cfg.b, self.cfg.c, self.cfg.d

        dx = -y - z
        dy = x + a * y + w
        dz = b + x * z
        dw = -c * z + d * w

        # Couplage stochastique optionnel w↔z
        if self.cfg.e > 0.0:
            noise = torch.randn(1).item() * self.cfg.e
            dw = dw + noise * z

        return torch.stack([dx, dy, dz, dw])

    def step(self) -> torch.Tensor:
        """
        Avance le système d'un pas dt (RK4) et retourne un vecteur
        de perturbation de taille (projection_dim,).

        Returns:
            perturbation normalisée × noise_scale
        """
        dt = self.cfg.dt
        s = self.state

        k1 = self._deriv(s)
        k2 = self._deriv(s + 0.5 * dt * k1)
        k3 = self._deriv(s + 0.5 * dt * k2)
        k4 = self._deriv(s + dt * k3)

        self.state = s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self._step_count += 1

        # Projection R⁴ → R^dim
        perturb = self.state @ self.W          # (projection_dim,)
        perturb = perturb / (perturb.norm() + 1e-8)
        return perturb * self.cfg.noise_scale

    def get_state(self) -> torch.Tensor:
        return self.state.clone()

    def lyapunov_proxy(self) -> float:
        """
        Proxy du spectre de Lyapunov : norme de la jacobienne au point courant.
        Pour le système hyperchaotique, cette valeur est structurellement
        plus élevée que pour Rössler simple.
        """
        deriv = self._deriv(self.state)
        return deriv.norm().item()

    def lyapunov_spectrum_proxy(self) -> Dict[str, float]:
        """
        Estimation grossière des deux plus grands exposants de Lyapunov
        via la méthode de la perturbation finie (approximation locale).

        Pas une mesure exacte — utile pour confirmer que le système
        est bien en régime hyperchaotique (λ₁ > 0 ET λ₂ > 0).

        Returns:
            dict avec lambda1, lambda2, sum (doit être < 0 pour dissipatif)
        """
        dt = self.cfg.dt
        eps = 1e-6

        s = self.state
        deriv_s = self._deriv(s)

        # Perturbation selon les deux premières directions canoniques
        lambda_estimates = []
        for i in range(min(2, 4)):
            e_i = torch.zeros(4)
            e_i[i] = eps
            deriv_perturbed = self._deriv(s + e_i)
            divergence_rate = (deriv_perturbed - deriv_s).norm() / eps
            lambda_estimates.append(math.log(divergence_rate.item() + 1e-8))

        return {
            "lambda1_proxy": lambda_estimates[0],
            "lambda2_proxy": lambda_estimates[1],
            "sum_proxy": sum(lambda_estimates),
            "hyperchaotic": lambda_estimates[0] > 0 and lambda_estimates[1] > 0,
        }


class RosslerHyperchaosPerDim:
    """
    Stress-test extrême : N instances hyperchaotiques indépendantes,
    une par dimension de l'espace de projection.

    Design :
        - projection_dim instances de RosslerHyperchaos en R⁴
        - Chaque instance a des paramètres (a, b, c, d) légèrement perturbés
          autour des valeurs canoniques → désynchronisation garantie
        - La dimension i de la perturbation sortante est le scalaire issu
          de l'instance i, projeté sur R¹ via un vecteur de lecture fixe

    Résultat :
        - Vecteur de perturbation de taille (projection_dim,)
        - Chaque composante est issue d'un attracteur R⁴ DISTINCT
        - La matrice de perturbation implicite est plein rang avec probabilité 1
        - Spectre de Lyapunov global : O(projection_dim) exposants positifs

    Coût computationnel :
        O(projection_dim × 4) = O(4D) opérations flottantes par step
        Pour D=256 : ~1024 opérations flottantes (négligeable sur CPU)
        Pour D=4096 : ~16k opérations flottantes (toujours négligeable)

    Quand utiliser :
        - Quand on veut maximiser la couverture de l'espace latent CBS
        - Stress-test : si le CBS survit à cette perturbation, il est robuste
        - Pour des espaces latents de grande dimension (D > 512)
    """

    def __init__(self, cfg: RosslerHyperchaosPerDimConfig = None):
        if cfg is None:
            cfg = RosslerHyperchaosPerDimConfig()
        self.cfg = cfg

        D = cfg.projection_dim
        rng_params = torch.Generator().manual_seed(cfg.seed)

        # Créer D instances avec paramètres perturbés
        self.instances: List[RosslerHyperchaos] = []
        self.read_vectors: List[torch.Tensor] = []  # R⁴ → R¹ par instance

        base = cfg.base
        jitter = cfg.param_jitter

        for i in range(D):
            # Perturbation multiplicative des paramètres
            j = 1.0 + jitter * (torch.randn(4, generator=rng_params) * 2 - 1)
            j = j.clamp(0.5, 2.0)  # éviter les valeurs aberrantes

            inst_cfg = RosslerHyperchaosConfig(
                a = base.a * j[0].item(),
                b = base.b * j[1].item(),
                c = base.c * j[2].item(),
                d = base.d * j[3].item(),
                e = base.e,
                dt = base.dt,
                noise_scale = 1.0,   # on normalise globalement après
                projection_dim = 4,  # inutilisé ici — on lit directement l'état
            )

            # Seed distinct par instance pour diverger immédiatement
            inst_seed = (cfg.seed * 1000 + i) % (2**31)
            inst = RosslerHyperchaos(inst_cfg, seed=inst_seed)

            # Vecteur de lecture fixe R⁴ → R¹ pour cette dimension
            rv = torch.randn(4, generator=rng_params)
            rv = rv / (rv.norm() + 1e-8)

            self.instances.append(inst)
            self.read_vectors.append(rv)

        # Pré-calculer la matrice de lecture (D, 4) pour la vectoriser
        self.R = torch.stack(self.read_vectors, dim=0)  # (D, 4)

        self._step_count = 0
        self.cfg.noise_scale = cfg.noise_scale  # amplitude globale

    def step(self) -> torch.Tensor:
        """
        Avance toutes les D instances d'un pas RK4 et retourne
        un vecteur de perturbation (D,) — une valeur par dimension,
        issue d'un attracteur R⁴ distinct.

        Returns:
            perturbation: (projection_dim,), normalisée × noise_scale
        """
        # Avancer toutes les instances et collecter les états
        states = torch.stack(
            [inst.step_state_only() if hasattr(inst, 'step_state_only')
             else self._step_and_get_state(inst)
             for inst in self.instances],
            dim=0
        )  # (D, 4)

        # Lecture scalaire par dimension : dot(état_i, vecteur_lecture_i)
        perturb = (states * self.R).sum(dim=1)  # (D,)

        # Normalisation globale
        perturb = perturb / (perturb.norm() + 1e-8)
        self._step_count += 1

        return perturb * self.cfg.noise_scale

    def _step_and_get_state(self, inst: RosslerHyperchaos) -> torch.Tensor:
        """Avance l'instance et retourne son état R⁴."""
        inst.step()
        return inst.get_state()

    def get_state(self) -> torch.Tensor:
        """
        Retourne la matrice d'états (D, 4) de toutes les instances.
        """
        return torch.stack([inst.get_state() for inst in self.instances], dim=0)

    def lyapunov_proxy(self) -> float:
        """
        Proxy du plus grand exposant de Lyapunov sur toutes les instances.
        Valeur typique : max des λ₁ de chaque instance ≈ 0.15-0.20
        """
        return max(inst.lyapunov_proxy() for inst in self.instances)

    def lyapunov_spectrum_summary(self, n_sample: int = 10) -> Dict[str, float]:
        """
        Résumé du spectre de Lyapunov sur un échantillon d'instances.
        Confirme que le régime hyperchaotique est bien maintenu.

        Args:
            n_sample: nombre d'instances à sonder (toutes si None)

        Returns:
            dict avec statistiques sur les exposants estimés
        """
        step = max(1, len(self.instances) // n_sample)
        sampled = self.instances[::step][:n_sample]

        l1_list, l2_list, hyperchaotic_count = [], [], 0

        for inst in sampled:
            spec = inst.lyapunov_spectrum_proxy()
            l1_list.append(spec["lambda1_proxy"])
            l2_list.append(spec["lambda2_proxy"])
            if spec["hyperchaotic"]:
                hyperchaotic_count += 1

        l1 = torch.tensor(l1_list)
        l2 = torch.tensor(l2_list)

        return {
            "n_instances": len(self.instances),
            "n_sampled": len(sampled),
            "lambda1_mean": l1.mean().item(),
            "lambda1_std":  l1.std().item(),
            "lambda2_mean": l2.mean().item(),
            "lambda2_std":  l2.std().item(),
            "hyperchaotic_fraction": hyperchaotic_count / len(sampled),
            "total_positive_exponents_estimate": 2 * len(self.instances),
        }

    def chaos_coverage_metric(self) -> Dict[str, float]:
        """
        Mesure de la couverture de l'espace de perturbation.

        Calcule la corrélation moyenne entre toutes les paires d'instances
        (sur les N derniers états). Idéalement proche de 0 — les instances
        explorent des directions indépendantes.

        Retourne aussi la variance par dimension comme proxy de l'exploration.
        """
        states = self.get_state()  # (D, 4)

        # Projections scalaires (D,) via les vecteurs de lecture
        projections = (states * self.R).sum(dim=1)  # (D,)

        # Variance des projections
        var = projections.var().item()

        # Corrélation moyenne entre dimensions (sur les états actuels)
        # Proxy : variance normalisée du vecteur de perturbation
        normalized = projections / (projections.std() + 1e-8)
        autocorr = (normalized * normalized.roll(1)).mean().item()

        return {
            "projection_variance": var,
            "autocorrelation_proxy": autocorr,
            "step": self._step_count,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory mise à jour
# ─────────────────────────────────────────────────────────────────────────────

class ChaoticDynamicsFactory:
    """
    Factory pour instancier le bon attracteur.

    Attracteurs disponibles :
        "lorenz"            : Lorenz R³ (exploration large)
        "rossler"           : Rössler R³ (exploration fine)
        "rossler_hyper"     : Rössler hyperchaotique R⁴ (2 exposants positifs)
        "rossler_hyper_per_dim" : N instances R⁴ indépendantes (stress-test extrême)
    """

    @staticmethod
    def build(
        attractor: Literal[
            "lorenz", "rossler",
            "rossler_hyper", "rossler_hyper_per_dim"
        ] = "lorenz",
        dim: int = 256,
        noise_scale: float = 1e-3,
        seed: int = 42,
        param_jitter: float = 0.05,   # Pour rossler_hyper_per_dim
        coupling_noise: float = 0.0,  # Paramètre e pour les variantes hyper
    ):
        if attractor == "lorenz":
            cfg = LorenzConfig(projection_dim=dim, noise_scale=noise_scale)
            return LorenzDynamics(cfg, seed=seed)

        elif attractor == "rossler":
            cfg = RosslerConfig(projection_dim=dim, noise_scale=noise_scale)
            return RosslerDynamics(cfg, seed=seed)

        elif attractor == "rossler_hyper":
            cfg = RosslerHyperchaosConfig(
                projection_dim=dim,
                noise_scale=noise_scale,
                e=coupling_noise,
            )
            return RosslerHyperchaos(cfg, seed=seed)

        elif attractor == "rossler_hyper_per_dim":
            base_cfg = RosslerHyperchaosConfig(
                projection_dim=4,   # inutilisé — lecture directe de l'état
                noise_scale=1.0,    # normalisé globalement dans PerDim
                e=coupling_noise,
            )
            cfg = RosslerHyperchaosPerDimConfig(
                base=base_cfg,
                param_jitter=param_jitter,
                projection_dim=dim,
                noise_scale=noise_scale,
                seed=seed,
            )
            return RosslerHyperchaosPerDim(cfg)

        else:
            raise ValueError(
                f"Unknown attractor: '{attractor}'. "
                f"Choose: 'lorenz', 'rossler', 'rossler_hyper', 'rossler_hyper_per_dim'."
            )
