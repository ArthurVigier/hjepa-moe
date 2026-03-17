# CBS — Chaotic Background Subnetwork
## `hjepa_moe/experiments/cbs/`

Expérimentation autonome. Ne pas intégrer dans le main network avant validation.

---

## Concept

Un sous-réseau délibérément non-contraint, animé par une dynamique chaotique
(Lorenz ou Rössler), qui tourne en parallèle du JEPA principal sans jamais
recevoir de gradient de son loss.

**Analogie biologique** : Default Mode Network — actif pendant le repos,
inhibe pendant la tâche, maintient accessibles des configurations que le
cortex actif n'explore plus.

**Invariant central** :
```python
# Tout ce qui sort du CBS est .detach() — sans exception
enriched = main_acts + gate * alpha * cbs.read(x)  # cbs.read() → toujours detach()
```

---

## Structure

```
cbs_experiment/
├── __init__.py
├── chaotic_dynamics.py     # Lorenz, Rössler (RK4), perturbation des poids
├── cbs_network.py          # Architecture CBS + gate cosinus asymétrique
├── cbs_regimes.py          # Régimes A / B / C
├── cbs_jepa_wrapper.py     # Wrapper JEPA générique
├── few_shot_ood_eval.py    # Evaluation few-shot OOD
└── smoke_test.py           # Tests complets
```

---

## Usage rapide

```bash
# Smoke test (30s CPU)
cd hjepa_moe/experiments/cbs
python smoke_test.py

# Avec GPU
python smoke_test.py --device cuda

# Comparaison complète avec training (2min CPU / 30s GPU)
python smoke_test.py --full --train-steps 500
```

---

## Trois régimes

| Régime | CBS update | Consultation | Hypothèse |
|--------|-----------|--------------|-----------|
| **A** | Tous les N batches | Jamais pendant training | Exploration large mais aveugle |
| **B** | À chaque step | À chaque forward | Exploration contextuelle |
| **C** | Proportionnel à `1/entropy` | Si intensity > seuil | Explore pendant la stabilité |

---

## Brancher sur ton JEPA existant

```python
from experiments.cbs import JEPAWithCBS, JEPAWithCBSConfig, CBSConfig

# Ton JEPA doit avoir : encode(), predict(), target_encode(), loss()
my_jepa = HJEPAMoE(...)

cfg = JEPAWithCBSConfig(
    cbs=CBSConfig(input_dim=256, hidden_dim=512, output_dim=256),
    regime="C",  # recommandé pour commencer
)
model = JEPAWithCBS(my_jepa, cfg)

# Training loop inchangé
loss, info = model.training_step(x_context, x_target)
loss.backward()
optimizer.step()
```

---

## Métriques à surveiller

- `cbs_gate_ratio` : fraction des samples qui consultent le CBS (0.3-0.7 = bon)
- `cbs_mean_cosine_sim` : similarité cosinus main/CBS (trop proche = CBS inutile, trop loin = gate jamais ouverte)
- `cbs_update_cbs_entropy_loss` : entropie interne du CBS (doit rester élevée)
- `chaos_lyapunov_proxy` : proxy exposant Lyapunov (doit rester > 0 pour confirmer régime chaotique)

---

## Questions ouvertes à tester

1. **Régime A vs B vs C** sur few-shot OOD : lequel améliore le plus ?
2. **Attractor choice** : Lorenz (exploration large) vs Rössler (exploration fine) ?
3. **Duty cycle naturel** : le CBS améliore-t-il en parallèle ou entre les batches ?
4. **Anti-collapse** : est-ce que le CBS réduit le representational collapse du predictor JEPA ?
