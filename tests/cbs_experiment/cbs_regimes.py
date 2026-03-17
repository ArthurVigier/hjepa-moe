"""
cbs_regimes.py
==============
Trois régimes d'activation du CBS, correspondant aux trois hypothèses
discutées :

Régime A — Séparé (cerveau-fidèle)
    Le CBS tourne uniquement quand le réseau principal est en mode "idle".
    Pendant l'inférence active, CBS gelé. Exploration entre les batches.
    Hypothèse : exploration plus large mais aveugle.

Régime B — Parallèle (computationnellement coûteux)
    Le CBS tourne en continu, même pendant l'inférence active.
    La gate filtre ce qui remonte.
    Hypothèse : exploration contextuelle, attracteurs proches de l'activité courante.

Régime C — Hybride entropie-dépendant (proposition originale)
    Le duty cycle du CBS est proportionnel à l'entropie de représentation
    du réseau principal. Quand le main network est "sûr" (faible entropie),
    CBS tourne à fond. Quand incertain, CBS réduit.
    Hypothèse : meilleur compromis exploration/exploitation.

Interface commune :
    regime.should_run_cbs(step, main_entropy) -> bool
    regime.should_consult(step, main_entropy) -> bool
    regime.get_cbs_intensity(main_entropy) -> float [0, 1]
"""

import torch
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class RegimeConfig:
    # Commun
    warmup_steps: int = 100         # Pas de CBS pendant le warmup du main network

    # Régime A
    idle_every_n_batches: int = 5   # CBS actif 1 batch sur N

    # Régime B
    # (pas de paramètre supplémentaire — toujours actif)

    # Régime C
    entropy_high_threshold: float = 2.0   # Au-dessus → CBS à fond
    entropy_low_threshold: float = 0.5    # En-dessous → CBS minimal
    min_cbs_intensity: float = 0.1        # CBS tourne toujours au moins un peu
    max_cbs_intensity: float = 1.0


class CBSRegime(ABC):
    """Interface commune pour tous les régimes."""

    def __init__(self, cfg: RegimeConfig = None):
        self.cfg = cfg or RegimeConfig()
        self._step = 0

    def tick(self):
        """Appeler à chaque step du réseau principal."""
        self._step += 1

    @abstractmethod
    def should_run_cbs_update(self, main_entropy: Optional[float] = None) -> bool:
        """Est-ce que le CBS doit faire son propre update step maintenant ?"""
        ...

    @abstractmethod
    def should_consult_cbs(self, main_entropy: Optional[float] = None) -> bool:
        """Est-ce que le réseau principal doit consulter le CBS maintenant ?"""
        ...

    @abstractmethod
    def get_cbs_intensity(self, main_entropy: Optional[float] = None) -> float:
        """
        Intensité de l'influence CBS ∈ [0, 1].
        Multiplie gate_alpha dans cbs_network.consult().
        """
        ...

    def _in_warmup(self) -> bool:
        return self._step < self.cfg.warmup_steps

    def describe(self) -> str:
        return f"{self.__class__.__name__} | step={self._step}"


class RegimeA(CBSRegime):
    """
    Régime A — Séparé.

    CBS update : tous les N batches (mode "entre les batches")
    CBS consult : JAMAIS pendant l'inférence active
                  (seulement en eval ou sur demande explicite)

    Simule le comportement du DMN — actif en mode repos, inhibé en tâche.
    """

    def should_run_cbs_update(self, main_entropy: Optional[float] = None) -> bool:
        if self._in_warmup():
            return False
        return self._step % self.cfg.idle_every_n_batches == 0

    def should_consult_cbs(self, main_entropy: Optional[float] = None) -> bool:
        # En régime A, on ne consulte JAMAIS pendant le training actif.
        # La consultation se fait uniquement en évaluation (géré à l'extérieur).
        return False

    def get_cbs_intensity(self, main_entropy: Optional[float] = None) -> float:
        return 1.0 if self.should_run_cbs_update(main_entropy) else 0.0


class RegimeB(CBSRegime):
    """
    Régime B — Parallèle.

    CBS update : à chaque step (en parallèle du main network)
    CBS consult : à chaque forward pass

    Hypothèse : les attracteurs explorés par le CBS sont corrélés
    avec l'activité courante → enrichissement contextuel.
    """

    def should_run_cbs_update(self, main_entropy: Optional[float] = None) -> bool:
        if self._in_warmup():
            return False
        return True  # Toujours

    def should_consult_cbs(self, main_entropy: Optional[float] = None) -> bool:
        if self._in_warmup():
            return False
        return True  # Toujours

    def get_cbs_intensity(self, main_entropy: Optional[float] = None) -> float:
        return 1.0


class RegimeC(CBSRegime):
    """
    Régime C — Hybride entropie-dépendant.

    CBS intensity = f(entropie du réseau principal)
    Quand le réseau principal est "confiant" (basse entropie) → CBS à fond
    Quand le réseau principal est "incertain" (haute entropie) → CBS réduit

    Inverse du réflexe habituel : on explore activement PENDANT la stabilité,
    pas pendant l'incertitude (qui nécessite toute l'attention du main network).

    L'intensité est calculée via une fonction sigmoïde inversée sur l'entropie.
    """

    def __init__(self, cfg: RegimeConfig = None):
        super().__init__(cfg)
        self._last_intensity = 0.5

    def _compute_intensity(self, main_entropy: Optional[float]) -> float:
        if main_entropy is None:
            return self._last_intensity

        cfg = self.cfg
        # Intensité inversement proportionnelle à l'entropie
        # entropy_high → intensity_min, entropy_low → intensity_max
        t = (main_entropy - cfg.entropy_low_threshold) / (
            cfg.entropy_high_threshold - cfg.entropy_low_threshold + 1e-8
        )
        t = max(0.0, min(1.0, t))  # clamp [0, 1]

        # Sigmoïde inversée : haute entropie → basse intensité
        intensity = cfg.max_cbs_intensity - t * (
            cfg.max_cbs_intensity - cfg.min_cbs_intensity
        )
        self._last_intensity = intensity
        return intensity

    def should_run_cbs_update(self, main_entropy: Optional[float] = None) -> bool:
        if self._in_warmup():
            return False
        intensity = self._compute_intensity(main_entropy)
        # Probabiliste : tourne avec probabilité = intensity
        return torch.rand(1).item() < intensity

    def should_consult_cbs(self, main_entropy: Optional[float] = None) -> bool:
        if self._in_warmup():
            return False
        intensity = self._compute_intensity(main_entropy)
        return intensity > self.cfg.min_cbs_intensity

    def get_cbs_intensity(self, main_entropy: Optional[float] = None) -> float:
        return self._compute_intensity(main_entropy)


class RegimeFactory:
    """Factory."""

    @staticmethod
    def build(regime: str, cfg: RegimeConfig = None) -> CBSRegime:
        cfg = cfg or RegimeConfig()
        if regime == "A":
            return RegimeA(cfg)
        elif regime == "B":
            return RegimeB(cfg)
        elif regime == "C":
            return RegimeC(cfg)
        else:
            raise ValueError(f"Unknown regime: {regime}. Choose 'A', 'B', or 'C'.")


def compute_activation_entropy(activations: torch.Tensor) -> float:
    """
    Calcule l'entropie des activations d'un réseau.
    Utilisé par RegimeC pour moduler l'intensité CBS.

    Args:
        activations: (B, D) tensor d'activations

    Returns:
        entropie scalaire (float)
    """
    with torch.no_grad():
        probs = torch.softmax(activations.detach(), dim=-1)
        log_probs = torch.log(probs + 1e-8)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
    return entropy.item()
