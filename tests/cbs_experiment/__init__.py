"""
CBS — Chaotic Background Subnetwork
====================================
Expérimentation dans hjepa_moe/experiments/cbs/

Structure:
    cbs_experiment/
    ├── __init__.py
    ├── chaotic_dynamics.py     # Lorenz, Rössler, attracteurs
    ├── cbs_network.py          # Architecture CBS + gate asymétrique
    ├── cbs_regimes.py          # Régimes A (alternant) / B (parallèle) / C (hybride)
    ├── cbs_jepa_wrapper.py     # Wrapper JEPA générique + CBS
    ├── few_shot_ood_eval.py    # Evaluation few-shot OOD
    └── smoke_test.py           # Smoke test complet

Principe:
    - Le CBS tourne en fond avec une dynamique chaotique injectée
    - Il n'est JAMAIS dans le graphe de gradient du réseau principal
    - La consultation se fait via gate cosinus + .detach() strict
    - Trois régimes testables : A (séparé), B (parallèle), C (hybride entropie-dépendant)
"""

from .cbs_network import CBSNetwork, CBSConfig
from .cbs_regimes import CBSRegime, RegimeA, RegimeB, RegimeC
from .cbs_jepa_wrapper import JEPAWithCBS
from .chaotic_dynamics import (
    LorenzDynamics, RosslerDynamics,
    RosslerHyperchaos, RosslerHyperchaosPerDim,
    ChaoticDynamicsFactory,
)

__all__ = [
    "CBSNetwork", "CBSConfig",
    "CBSRegime", "RegimeA", "RegimeB", "RegimeC",
    "JEPAWithCBS",
    "LorenzDynamics", "RosslerDynamics",
    "RosslerHyperchaos", "RosslerHyperchaosPerDim",
    "ChaoticDynamicsFactory",
]
