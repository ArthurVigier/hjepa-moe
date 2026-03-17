"""
smoke_test.py
=============
Smoke test complet du CBS experiment.

Vérifie :
    1. Chaotic dynamics (Lorenz, Rössler) — determinisme, stabilité
    2. CBSNetwork — forward, read, consult, update step
    3. Invariant central : aucun gradient ne traverse vers CBS depuis main
    4. Régimes A/B/C — scheduling correct
    5. JEPAWithCBS — training step complet
    6. Few-shot OOD eval — pipeline end-to-end
    7. Comparaison des 3 régimes vs baseline (mini version)

Usage:
    python smoke_test.py
    python smoke_test.py --device cuda  # si GPU disponible
    python smoke_test.py --full         # version longue avec vraie comparaison

Durée attendue : ~30s CPU (smoke), ~2min CPU (full)
"""

import torch
import torch.nn as nn
import argparse
import time
import sys
import math

# ── Imports locaux ────────────────────────────────────────────────────────────
from chaotic_dynamics import (
    LorenzDynamics, LorenzConfig,
    RosslerDynamics, RosslerConfig,
    RosslerHyperchaos, RosslerHyperchaosConfig,
    RosslerHyperchaosPerDim, RosslerHyperchaosPerDimConfig,
    ChaoticDynamicsFactory,
)
from cbs_network import CBSNetwork, CBSConfig
from cbs_regimes import (
    RegimeA, RegimeB, RegimeC,
    RegimeConfig, RegimeFactory,
    compute_activation_entropy,
)
from cbs_jepa_wrapper import (
    JEPAWithCBS, JEPAWithCBSConfig,
    MinimalJEPA,
)
from few_shot_ood_eval import (
    SourceDistribution, OODDistribution,
    FewShotEvalConfig, run_ood_evaluation, print_results_table,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

def ok(msg: str):
    print(f"  ✓  {msg}")

def fail(msg: str):
    print(f"  ✗  {msg}")
    sys.exit(1)

def check(condition: bool, msg_ok: str, msg_fail: str = ""):
    if condition:
        ok(msg_ok)
    else:
        fail(msg_fail or msg_ok)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 : Dynamiques chaotiques
# ─────────────────────────────────────────────────────────────────────────────

def test_chaotic_dynamics():
    section("1. Chaotic Dynamics")

    DIM = 128

    for cls, cfg_cls, name in [
        (LorenzDynamics, LorenzConfig, "Lorenz"),
        (RosslerDynamics, RosslerConfig, "Rössler"),
    ]:
        cfg = cfg_cls(projection_dim=DIM, noise_scale=1e-3)
        dyn = cls(cfg, seed=42)

        # Determinisme : deux instances avec le même seed → même séquence
        dyn2 = cls(cfg, seed=42)
        p1 = dyn.step()
        p2 = dyn2.step()
        check(
            torch.allclose(p1, p2),
            f"{name}: reproductibilité seed OK",
            f"{name}: séquences divergent avec même seed"
        )

        # Shape
        check(p1.shape == (DIM,), f"{name}: shape perturbation = ({DIM},)")

        # Norme ≈ noise_scale (à un facteur près)
        check(
            p1.norm().item() < 0.1,
            f"{name}: norme perturbation raisonnable ({p1.norm().item():.2e})"
        )

        # Divergence avec seed différent
        dyn3 = cls(cfg, seed=123)
        p3 = dyn3.step()
        check(
            not torch.allclose(p1, p3),
            f"{name}: seeds différents → perturbations différentes"
        )

        # 100 steps sans NaN
        for _ in range(100):
            p = dyn.step()
        check(
            not p.isnan().any() and not p.isinf().any(),
            f"{name}: stable après 100 steps"
        )

    # ── Rössler Hyperchaotique R⁴ ────────────────────────────────────────────
    section("1b. Rössler Hyperchaotique R⁴")

    DIM = 64
    hyper_cfg = RosslerHyperchaosConfig(projection_dim=DIM, noise_scale=1e-3)
    dyn_h = RosslerHyperchaos(hyper_cfg, seed=42)

    # Shape et reproductibilité
    dyn_h2 = RosslerHyperchaos(hyper_cfg, seed=42)
    p_h1 = dyn_h.step()
    p_h2 = dyn_h2.step()
    check(p_h1.shape == (DIM,), f"Hyperchaos R⁴: shape = ({DIM},)")
    check(torch.allclose(p_h1, p_h2), "Hyperchaos R⁴: reproductibilité seed OK")

    # 200 steps sans divergence (R⁴ peut être plus instable)
    stable = True
    for _ in range(200):
        p = dyn_h.step()
        if p.isnan().any() or p.isinf().any():
            stable = False
            break
    check(stable, "Hyperchaos R⁴: stable après 200 steps")

    # Vérification spectre de Lyapunov
    spec = dyn_h.lyapunov_spectrum_proxy()
    check("lambda1_proxy" in spec, "Hyperchaos R⁴: spectrum proxy calculable")
    check("hyperchaotic" in spec, "Hyperchaos R⁴: clé 'hyperchaotic' présente")
    print(f"      λ₁≈{spec['lambda1_proxy']:.3f}, λ₂≈{spec['lambda2_proxy']:.3f}, "
          f"hyperchaotic={spec['hyperchaotic']}")

    # Coupling noise (paramètre e)
    hyper_cfg_e = RosslerHyperchaosConfig(projection_dim=DIM, e=0.01)
    dyn_he = RosslerHyperchaos(hyper_cfg_e, seed=42)
    for _ in range(50):
        p = dyn_he.step()
    check(
        not p.isnan().any(),
        "Hyperchaos R⁴ + coupling noise (e=0.01): stable après 50 steps"
    )

    # ── Rössler Hyperchaos Per-Dimension (stress-test extrême) ────────────────
    section("1c. Rössler Hyperchaos Per-Dim (stress-test extrême)")

    DIM_PD = 32   # Réduit pour le smoke test (D=256+ pour la vraie expérience)
    base_cfg = RosslerHyperchaosConfig(noise_scale=1.0)
    pd_cfg = RosslerHyperchaosPerDimConfig(
        base=base_cfg,
        param_jitter=0.05,
        projection_dim=DIM_PD,
        noise_scale=1e-3,
        seed=42,
    )
    dyn_pd = RosslerHyperchaosPerDim(pd_cfg)

    check(len(dyn_pd.instances) == DIM_PD,
          f"PerDim: {DIM_PD} instances créées")
    check(dyn_pd.R.shape == (DIM_PD, 4),
          f"PerDim: matrice de lecture shape ({DIM_PD}, 4)")

    # Step
    p_pd = dyn_pd.step()
    check(p_pd.shape == (DIM_PD,), f"PerDim: step() shape = ({DIM_PD},)")
    check(not p_pd.isnan().any(), "PerDim: pas de NaN")

    # 100 steps
    stable_pd = True
    for _ in range(100):
        p = dyn_pd.step()
        if p.isnan().any() or p.isinf().any():
            stable_pd = False
            break
    check(stable_pd, "PerDim: stable après 100 steps")

    # Spectre de Lyapunov résumé
    summary = dyn_pd.lyapunov_spectrum_summary(n_sample=5)
    check(summary["n_sampled"] <= DIM_PD, "PerDim: spectrum summary OK")
    print(f"      λ₁_mean≈{summary['lambda1_proxy_mean'] if 'lambda1_proxy_mean' in summary else summary['lambda1_mean']:.3f}, "
          f"λ₂_mean≈{summary['lambda2_mean']:.3f}, "
          f"hyperchaotic_frac={summary['hyperchaotic_fraction']:.0%}, "
          f"total_positive_exponents≈{summary['total_positive_exponents_estimate']}")

    # Désynchronisation : vérifier que les instances divergent
    states_before = dyn_pd.get_state().clone()   # (D, 4)
    for _ in range(50):
        dyn_pd.step()
    states_after = dyn_pd.get_state()
    diffs = (states_after - states_before).norm(dim=1)  # (D,) distance par instance
    # Les instances doivent avoir divergé différemment
    check(
        diffs.std().item() > 0,
        f"PerDim: instances désynchronisées (std des distances = {diffs.std().item():.3f})"
    )

    # Coverage metric
    coverage = dyn_pd.chaos_coverage_metric()
    check("projection_variance" in coverage, "PerDim: coverage metric OK")
    print(f"      projection_variance={coverage['projection_variance']:.4f}, "
          f"autocorr={coverage['autocorrelation_proxy']:.4f}")

    # Factory — tous les attracteurs
    for name, kwargs in [
        ("lorenz",               {}),
        ("rossler",              {}),
        ("rossler_hyper",        {"coupling_noise": 0.0}),
        ("rossler_hyper_per_dim", {"param_jitter": 0.05}),
    ]:
        d = ChaoticDynamicsFactory.build(name, dim=32, noise_scale=1e-3, seed=0, **kwargs)
        p = d.step()
        check(p.shape == (32,), f"Factory '{name}': shape OK")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 : CBSNetwork
# ─────────────────────────────────────────────────────────────────────────────

def test_cbs_network(device: str = "cpu"):
    section("2. CBSNetwork")

    B, D = 4, 256
    cfg = CBSConfig(
        input_dim=D, hidden_dim=128, output_dim=D,
        n_layers=2, attractor="lorenz",
        chaos_noise_scale=1e-3, perturb_every_n_steps=5,
    )
    cbs = CBSNetwork(cfg).to(device)
    x = torch.randn(B, D, device=device)

    # read() retourne bien du .detach()
    out = cbs.read(x)
    check(out.shape == (B, D), "read() shape correcte")
    check(not out.requires_grad, "read() : pas de requires_grad (invariant)")

    # consult() : enriched a le même shape que main_activations
    main_acts = torch.randn(B, D, device=device, requires_grad=True)
    enriched, info = cbs.consult(main_acts, x)
    check(enriched.shape == (B, D), "consult() shape correcte")
    check("gate_ratio" in info, "consult() retourne gate_ratio")

    # INVARIANT CENTRAL : gradient du loss enriched ne doit PAS atteindre CBS
    loss = enriched.sum()
    loss.backward()
    cbs_grad_exists = any(
        p.grad is not None and p.grad.abs().max() > 0
        for p in cbs.parameters()
    )
    check(
        not cbs_grad_exists,
        "INVARIANT : aucun gradient main → CBS ✓",
        "INVARIANT VIOLÉ : gradient du main a atteint le CBS !"
    )

    # cbs_update_step() — a son propre gradient
    cbs.zero_grad()
    metrics = cbs.cbs_update_step(x)
    check("cbs_entropy_loss" in metrics, "cbs_update_step() retourne metrics")

    # Perturbation chaotique : les poids changent après perturb_every_n_steps
    w_before = cbs.blocks[0].fc1.weight.data.clone()
    for i in range(cfg.perturb_every_n_steps + 1):
        cbs.cbs_update_step(x)
    w_after = cbs.blocks[0].fc1.weight.data
    check(
        not torch.allclose(w_before, w_after),
        "Perturbation chaotique : poids modifiés après N steps"
    )

    # get_stats()
    stats = cbs.get_stats()
    check("gate_open_ratio" in stats, "get_stats() OK")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 : Régimes
# ─────────────────────────────────────────────────────────────────────────────

def test_regimes():
    section("3. CBS Régimes A / B / C")

    cfg = RegimeConfig(warmup_steps=5, idle_every_n_batches=3)

    # ── Régime A ──
    r_a = RegimeA(cfg)
    # Pendant warmup : jamais
    for _ in range(5):
        r_a.tick()
        check(not r_a.should_run_cbs_update(), "Régime A : warmup OK")
    # Après warmup : tous les N steps
    r_a.tick()  # step 6
    check(r_a.should_run_cbs_update(), "Régime A : update au step N=6 (6%3==0)")
    check(not r_a.should_consult_cbs(), "Régime A : pas de consultation pendant training")

    # ── Régime B ──
    r_b = RegimeB(cfg)
    for _ in range(5):
        r_b.tick()
    r_b.tick()
    check(r_b.should_run_cbs_update(), "Régime B : toujours actif après warmup")
    check(r_b.should_consult_cbs(), "Régime B : consultation toujours active")
    check(r_b.get_cbs_intensity() == 1.0, "Régime B : intensité = 1.0")

    # ── Régime C ──
    r_c = RegimeC(cfg)
    for _ in range(6):
        r_c.tick()

    # Basse entropie → haute intensité
    intensity_low_entropy = r_c.get_cbs_intensity(main_entropy=0.1)
    # Haute entropie → basse intensité
    intensity_high_entropy = r_c.get_cbs_intensity(main_entropy=3.0)

    check(
        intensity_low_entropy > intensity_high_entropy,
        f"Régime C : intensité inversement proportionnelle à l'entropie "
        f"(low_e={intensity_low_entropy:.2f} > high_e={intensity_high_entropy:.2f})"
    )

    # Factory
    for regime_name in ["A", "B", "C"]:
        r = RegimeFactory.build(regime_name, cfg)
        check(r is not None, f"RegimeFactory.build('{regime_name}') OK")

    # compute_activation_entropy
    acts = torch.randn(8, 64)
    entropy = compute_activation_entropy(acts)
    check(entropy > 0, f"compute_activation_entropy = {entropy:.3f} > 0")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 : JEPAWithCBS — training step complet
# ─────────────────────────────────────────────────────────────────────────────

def test_jepa_with_cbs(device: str = "cpu"):
    section("4. JEPAWithCBS — training step")

    B, D = 8, 64

    for regime_name in ["A", "B", "C"]:
        jepa = MinimalJEPA(dim=D, hidden_dim=128).to(device)
        cbs_cfg = CBSConfig(
            input_dim=D, hidden_dim=128, output_dim=D,
            n_layers=2, perturb_every_n_steps=3,
        )
        regime_cfg = RegimeConfig(warmup_steps=2)
        cfg = JEPAWithCBSConfig(
            cbs=cbs_cfg, regime=regime_name, regime_cfg=regime_cfg
        )
        model = JEPAWithCBS(jepa, cfg).to(device)

        opt = torch.optim.Adam(model.jepa.parameters(), lr=1e-3)

        # Simulate training
        for step in range(10):
            x_ctx = torch.randn(B, D, device=device)
            x_tgt = torch.randn(B, D, device=device)

            opt.zero_grad()
            loss, info = model.training_step(x_ctx, x_tgt)
            loss.backward()
            opt.step()

            # Vérifier que le loss est scalaire fini
            assert loss.ndim == 0 and not loss.isnan(), \
                f"Régime {regime_name}: loss invalide au step {step}"

        check(True, f"Régime {regime_name}: 10 steps sans erreur, loss={loss.item():.4f}")

        # encode_with_cbs
        x = torch.randn(B, D, device=device)
        emb, cbs_info = model.encode_with_cbs(x, force_consult=True)
        check(emb.shape == (B, D), f"Régime {regime_name}: encode_with_cbs shape OK")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 : Few-shot OOD — pipeline end-to-end
# ─────────────────────────────────────────────────────────────────────────────

def test_few_shot_ood_pipeline(device: str = "cpu"):
    section("5. Few-shot OOD — pipeline end-to-end (mini)")

    DIM = 64
    source = SourceDistribution(dim=DIM, n_modes=3, seed=0)

    # Modèles (non entraînés pour ce smoke test)
    def make_model(regime):
        jepa = MinimalJEPA(dim=DIM, hidden_dim=128)
        cbs_cfg = CBSConfig(input_dim=DIM, hidden_dim=64, output_dim=DIM, n_layers=1)
        cfg = JEPAWithCBSConfig(cbs=cbs_cfg, regime=regime)
        return JEPAWithCBS(jepa, cfg)

    models = {
        "baseline": MinimalJEPA(dim=DIM, hidden_dim=128),
        "regime_B": make_model("B"),
    }

    eval_cfg = FewShotEvalConfig(
        k_shots=[1, 5],
        n_eval_tasks=3,          # Mini pour le smoke test
        n_test_per_task=50,
        adaptation_steps=2,
        dim=DIM,
        device=device,
    )

    results = run_ood_evaluation(models, source, eval_cfg)

    # Vérifier la structure des résultats
    for model_name in models:
        for ood_name in results[model_name]:
            for k in eval_cfg.k_shots:
                check(
                    "mean" in results[model_name][ood_name][k],
                    f"Results structure OK: {model_name}/{ood_name}/k={k}"
                )

    print_results_table(results)


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 (FULL) : Comparaison complète avec training
# ─────────────────────────────────────────────────────────────────────────────

def test_full_comparison(device: str = "cpu", n_train_steps: int = 500):
    section(f"6. Comparaison complète (training {n_train_steps} steps)")

    DIM = 128
    B = 32
    source = SourceDistribution(dim=DIM, n_modes=4, seed=0)

    def make_and_train(regime: str = None, n_steps: int = n_train_steps):
        """Crée et entraîne un modèle (avec ou sans CBS)."""
        jepa = MinimalJEPA(dim=DIM, hidden_dim=256).to(device)

        if regime is None:
            model = jepa
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        else:
            cbs_cfg = CBSConfig(
                input_dim=DIM, hidden_dim=256, output_dim=DIM, n_layers=2,
                perturb_every_n_steps=10, chaos_noise_scale=5e-4,
            )
            regime_cfg = RegimeConfig(warmup_steps=50)
            cfg = JEPAWithCBSConfig(cbs=cbs_cfg, regime=regime, regime_cfg=regime_cfg)
            model = JEPAWithCBS(jepa, cfg).to(device)
            opt = torch.optim.Adam(model.jepa.parameters(), lr=1e-3)

        print(f"    Training {regime or 'baseline'} ({n_steps} steps)...", end=" ", flush=True)
        t0 = time.time()

        for step in range(n_steps):
            x = source.sample(B, device=device)
            x_shifted = x + 0.1 * torch.randn_like(x)

            opt.zero_grad()
            if regime is None:
                emb = model.encode(x)
                pred = model.predict(emb)
                with torch.no_grad():
                    tgt = model.target_encode(x_shifted)
                loss = model.loss(pred, tgt)
                model.update_target_encoder()
            else:
                loss, _ = model.training_step(x, x_shifted)

            loss.backward()
            opt.step()

        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s, final loss={loss.item():.4f})")
        return model

    # Entraîner tous les modèles
    baseline = make_and_train(regime=None)
    model_a = make_and_train(regime="A")
    model_b = make_and_train(regime="B")
    model_c = make_and_train(regime="C")

    models = {
        "baseline": baseline,
        "regime_A": model_a,
        "regime_B": model_b,
        "regime_C": model_c,
    }

    # Évaluation OOD
    eval_cfg = FewShotEvalConfig(
        k_shots=[1, 5, 20],
        n_eval_tasks=10,
        n_test_per_task=200,
        adaptation_steps=5,
        dim=DIM,
        device=device,
    )

    print("\n  Évaluation OOD en cours...")
    results = run_ood_evaluation(models, source, eval_cfg)
    print_results_table(results)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="cpu | cuda")
    parser.add_argument("--full", action="store_true", help="Run full comparison with training")
    parser.add_argument("--train-steps", type=int, default=500)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  CBS Experiment — Smoke Test")
    print(f"  device={args.device} | full={args.full}")
    print(f"{'='*60}")

    t_start = time.time()

    test_chaotic_dynamics()
    test_cbs_network(args.device)
    test_regimes()
    test_jepa_with_cbs(args.device)
    test_few_shot_ood_pipeline(args.device)

    if args.full:
        test_full_comparison(args.device, n_train_steps=args.train_steps)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  ✓ Tous les tests passés en {elapsed:.1f}s")
    print(f"{'='*60}\n")
