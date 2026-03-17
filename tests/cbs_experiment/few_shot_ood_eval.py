"""
few_shot_ood_eval.py
=====================
Évaluation few-shot OOD — le test central pour valider le CBS.

Hypothèse testée :
    Un JEPA+CBS adapte mieux à des distributions hors-training
    qu'un JEPA sans CBS, particulièrement avec peu d'exemples (few-shot).

Protocole :
    1. Train un JEPA (avec et sans CBS) sur une distribution "source"
    2. Créer des tâches OOD de complexité croissante
    3. Pour chaque tâche OOD, mesurer l'adaptation avec k=1,2,5,10,20 shots
    4. Métrique : MSE de reconstruction après fine-tuning k-shot

Les distributions OOD sont construites comme transformations de la source :
    - Rotation dans l'espace latent (OOD légère)
    - Changement de structure de covariance (OOD modérée)
    - Distribution complètement différente (OOD sévère)

Comparaison :
    baseline_jepa       : JEPA sans CBS
    jepa_regime_A       : JEPA + CBS régime A (séparé)
    jepa_regime_B       : JEPA + CBS régime B (parallèle)
    jepa_regime_C       : JEPA + CBS régime C (hybride)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────────────────────
# Distributions synthétiques
# ─────────────────────────────────────────────────────────────

class SourceDistribution:
    """Distribution source : gaussienne avec structure de covariance."""

    def __init__(self, dim: int = 256, n_modes: int = 4, seed: int = 0):
        self.dim = dim
        self.n_modes = n_modes
        rng = torch.Generator().manual_seed(seed)

        # Centres des modes
        self.centers = torch.randn(n_modes, dim, generator=rng) * 2.0

        # Covariances diagonales par mode
        self.scales = torch.rand(n_modes, dim, generator=rng) * 0.5 + 0.1

    def sample(self, n: int, device: str = "cpu") -> torch.Tensor:
        """Sample n exemples de la distribution source."""
        # Choisir un mode par sample
        mode_ids = torch.randint(0, self.n_modes, (n,))
        centers = self.centers[mode_ids].to(device)
        scales = self.scales[mode_ids].to(device)
        noise = torch.randn(n, self.dim, device=device)
        return centers + scales * noise


class OODDistribution:
    """
    Distributions OOD de complexité croissante.
    Construites à partir de la distribution source.
    """

    @staticmethod
    def rotation(source: SourceDistribution, angle_fraction: float = 0.1) -> "OODDistribution":
        """OOD légère : rotation dans un sous-espace 2D."""
        ood = OODDistribution()
        ood._source = source
        ood._type = "rotation"
        ood._angle = angle_fraction * math.pi
        return ood

    @staticmethod
    def covariance_shift(source: SourceDistribution, scale_factor: float = 3.0) -> "OODDistribution":
        """OOD modérée : changement de scale des variances."""
        ood = OODDistribution()
        ood._source = source
        ood._type = "covariance_shift"
        ood._scale_factor = scale_factor
        return ood

    @staticmethod
    def full_shift(dim: int = 256, seed: int = 99) -> "OODDistribution":
        """OOD sévère : distribution complètement différente."""
        ood = OODDistribution()
        ood._source = None
        ood._type = "full_shift"
        ood._dim = dim
        ood._seed = seed
        # Distribution laplace + structure différente
        rng = torch.Generator().manual_seed(seed)
        ood._center = torch.randn(dim, generator=rng) * 5.0
        return ood

    def sample(self, n: int, device: str = "cpu") -> torch.Tensor:
        if self._type == "rotation":
            x = self._source.sample(n, device)
            # Rotation dans le plan (0, 1)
            angle = self._angle
            x_rot = x.clone()
            x_rot[:, 0] = x[:, 0] * math.cos(angle) - x[:, 1] * math.sin(angle)
            x_rot[:, 1] = x[:, 0] * math.sin(angle) + x[:, 1] * math.cos(angle)
            return x_rot

        elif self._type == "covariance_shift":
            x = self._source.sample(n, device)
            # Amplifier les dimensions paires
            mask = torch.zeros(x.shape[-1], device=device)
            mask[::2] = self._scale_factor
            mask[1::2] = 1.0 / self._scale_factor
            return x * mask.unsqueeze(0)

        elif self._type == "full_shift":
            # Laplace + centre décalé
            u = torch.rand(n, self._dim, device=device) - 0.5
            x = -torch.sign(u) * torch.log(1 - 2 * u.abs() + 1e-8)
            x = x + self._center.to(device)
            return x

        raise ValueError(f"Unknown type: {self._type}")


# ─────────────────────────────────────────────────────────────
# Évaluation few-shot
# ─────────────────────────────────────────────────────────────

@dataclass
class FewShotEvalConfig:
    k_shots: List[int] = field(default_factory=lambda: [1, 2, 5, 10, 20])
    n_eval_tasks: int = 20          # Nombre de tâches OOD par type
    n_test_per_task: int = 200      # Exemples de test par tâche
    adaptation_lr: float = 1e-3
    adaptation_steps: int = 5       # Steps de fine-tuning few-shot
    dim: int = 256
    device: str = "cpu"


def few_shot_adapt_and_eval(
    model: nn.Module,
    support_x: torch.Tensor,
    query_x: torch.Tensor,
    cfg: FewShotEvalConfig,
    use_cbs: bool = False,
) -> float:
    """
    Adapte un modèle sur `support_x` (k shots) et évalue sur `query_x`.

    Pour CBS : on utilise encode_with_cbs pour encoder.
    Pour baseline : on utilise encode directement.

    Retourne le MSE moyen sur query_x après adaptation.
    """
    # Clone du modèle pour ne pas affecter le modèle original
    model_copy = copy.deepcopy(model)
    model_copy.train()

    # Optimizer pour l'adaptation few-shot
    # On adapte seulement le predictor / les couches finales
    if hasattr(model_copy, 'jepa'):
        params = list(model_copy.jepa.predictor.parameters())
    else:
        params = list(model_copy.parameters())

    opt = torch.optim.Adam(params, lr=cfg.adaptation_lr)

    # ── Adaptation ────────────────────────────────────────────────────────
    for _ in range(cfg.adaptation_steps):
        opt.zero_grad()

        if use_cbs and hasattr(model_copy, 'encode_with_cbs'):
            with torch.no_grad():
                emb, _ = model_copy.encode_with_cbs(support_x, force_consult=True)
        else:
            with torch.no_grad():
                if hasattr(model_copy, 'jepa'):
                    emb = model_copy.jepa.encode(support_x)
                else:
                    emb = model_copy.encode(support_x)

        # Reconstruction self-supervised : predict ≈ encode
        if hasattr(model_copy, 'jepa'):
            pred = model_copy.jepa.predict(emb)
        else:
            pred = model_copy.predict(emb)

        loss = F.mse_loss(pred, emb.detach())
        loss.backward()
        opt.step()

    # ── Évaluation ────────────────────────────────────────────────────────
    model_copy.eval()
    with torch.no_grad():
        if use_cbs and hasattr(model_copy, 'encode_with_cbs'):
            query_emb, _ = model_copy.encode_with_cbs(query_x, force_consult=True)
        else:
            if hasattr(model_copy, 'jepa'):
                query_emb = model_copy.jepa.encode(query_x)
            else:
                query_emb = model_copy.encode(query_x)

        if hasattr(model_copy, 'jepa'):
            query_pred = model_copy.jepa.predict(query_emb)
        else:
            query_pred = model_copy.predict(query_emb)

        mse = F.mse_loss(query_pred, query_emb).item()

    return mse


def run_ood_evaluation(
    models: Dict[str, nn.Module],
    source: SourceDistribution,
    cfg: FewShotEvalConfig = None,
) -> Dict:
    """
    Évaluation complète OOD pour tous les modèles et tous les régimes.

    Args:
        models: dict {"baseline": model, "regime_A": model, ...}
        source: distribution source d'entraînement
        cfg: config d'évaluation

    Returns:
        results: dict structuré avec les MSE par modèle / OOD type / k
    """
    if cfg is None:
        cfg = FewShotEvalConfig()

    device = cfg.device

    # Créer les distributions OOD
    ood_distributions = {
        "rotation_light": OODDistribution.rotation(source, angle_fraction=0.1),
        "rotation_heavy": OODDistribution.rotation(source, angle_fraction=0.4),
        "covariance_shift": OODDistribution.covariance_shift(source, scale_factor=3.0),
        "full_shift": OODDistribution.full_shift(dim=cfg.dim, seed=99),
    }

    results = {
        model_name: {
            ood_name: {k: [] for k in cfg.k_shots}
            for ood_name in ood_distributions
        }
        for model_name in models
    }

    print(f"\n{'='*60}")
    print(f"Few-shot OOD Evaluation")
    print(f"Models: {list(models.keys())}")
    print(f"OOD types: {list(ood_distributions.keys())}")
    print(f"k-shots: {cfg.k_shots}")
    print(f"{'='*60}\n")

    for ood_name, ood_dist in ood_distributions.items():
        print(f"  OOD: {ood_name}")

        for task_idx in range(cfg.n_eval_tasks):
            # Générer query set commun à tous les modèles
            query_x = ood_dist.sample(cfg.n_test_per_task, device=device)

            for k in cfg.k_shots:
                # Support set (k shots)
                support_x = ood_dist.sample(k, device=device)

                for model_name, model in models.items():
                    use_cbs = (model_name != "baseline")

                    mse = few_shot_adapt_and_eval(
                        model=model,
                        support_x=support_x,
                        query_x=query_x,
                        cfg=cfg,
                        use_cbs=use_cbs,
                    )
                    results[model_name][ood_name][k].append(mse)

    # Agréger les résultats
    aggregated = {}
    for model_name in models:
        aggregated[model_name] = {}
        for ood_name in ood_distributions:
            aggregated[model_name][ood_name] = {}
            for k in cfg.k_shots:
                vals = results[model_name][ood_name][k]
                t = torch.tensor(vals)
                aggregated[model_name][ood_name][k] = {
                    "mean": t.mean().item(),
                    "std": t.std().item(),
                    "min": t.min().item(),
                }

    return aggregated


def print_results_table(results: Dict):
    """Affichage formaté des résultats."""
    models = list(results.keys())
    ood_types = list(results[models[0]].keys())
    k_shots = list(results[models[0]][ood_types[0]].keys())

    print(f"\n{'='*80}")
    print("RESULTS — Mean MSE (lower is better)")
    print(f"{'='*80}")

    for ood_name in ood_types:
        print(f"\n  OOD: {ood_name}")
        print(f"  {'k':>6}", end="")
        for model in models:
            print(f"  {model:>15}", end="")
        print()

        for k in k_shots:
            print(f"  {k:>6}", end="")
            for model in models:
                mean = results[model][ood_name][k]["mean"]
                std = results[model][ood_name][k]["std"]
                print(f"  {mean:.4f}±{std:.4f}", end="")
            print()

    # Résumé : CBS vs baseline
    print(f"\n{'='*80}")
    print("SUMMARY — CBS improvement over baseline (negative = CBS better)")
    print(f"{'='*80}")
    if "baseline" in results:
        for model in models:
            if model == "baseline":
                continue
            improvements = []
            for ood_name in ood_types:
                for k in k_shots:
                    baseline_mse = results["baseline"][ood_name][k]["mean"]
                    model_mse = results[model][ood_name][k]["mean"]
                    rel_improvement = (model_mse - baseline_mse) / (baseline_mse + 1e-8)
                    improvements.append(rel_improvement)
            mean_imp = sum(improvements) / len(improvements)
            print(f"  {model}: {mean_imp:+.2%} relative to baseline")
