"""
H-JEPA-MoE: Hierarchical Joint-Embedding Predictive Architecture
             with Mixture-of-Experts Predictors

Architecture inspired by:
  - LeCun (2022) "A Path Towards Autonomous Machine Intelligence"
  - Assran et al. (2025) V-JEPA 2
  - Balestriero & LeCun (2025) LeJEPA / SIGReg
  - Lei et al. (2025) M3-JEPA (MoE predictor)
  - Terver et al. (2026) EB-JEPA

Core idea: Stack L JEPA levels. Each level ℓ has:
  - A temporal encoder Enc_ℓ that pools k_ℓ states from level ℓ-1
  - A MoE predictor with N experts and top-k routing
  - A regularized loss (VICReg or SIGReg) per level

The MoE routing is the key innovation:
  Different experts specialize in different temporal dynamics
  (fast/slow, rigid/deformable, contact/free-flight, etc.)
"""

from hjepa_moe.model import HJEPAMoE
from hjepa_moe.encoders.temporal import TemporalEncoder
from hjepa_moe.predictors.moe_predictor import MoEPredictor
from hjepa_moe.losses.vicreg import VICRegLoss
from hjepa_moe.losses.sigreg import SIGRegLoss

__version__ = "0.1.0"
__all__ = [
    "HJEPAMoE",
    "TemporalEncoder",
    "MoEPredictor",
    "VICRegLoss",
    "SIGRegLoss",
]
