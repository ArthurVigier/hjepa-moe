# hjepa-moe

Experimental implementation of **Hierarchical JEPA with Mixture-of-Experts predictors**.

This is a personal research sandbox, not a production library. The goal is to have
a clean, runnable codebase that stays close to what AMI Labs / Meta FAIR are likely
building internally — close enough to run real architectural experiments, loose enough
to iterate fast.

---

## What this is

JEPA (Joint Embedding Predictive Architecture, LeCun 2022) predicts in **embedding space**
instead of pixel/token space. H-JEPA stacks multiple JEPA levels with different temporal
resolutions. This repo adds **MoE predictors** at each level so different experts can
specialize on different dynamic regimes (fast motion, slow background, contact events, etc.).

None of this is published. It is a plausible extrapolation from:
- LeCun (2022) — H-JEPA blueprint
- V-JEPA 2 (Assran et al., 2025) — EMA training, multi-step rollout, MPC planning
- LeJEPA / SIGReg (Balestriero & LeCun, 2025) — anti-collapse theory
- M3-JEPA (Lei et al., ICML 2025) — MoE as JEPA predictor
- EB-JEPA (Terver, Rabbat, LeCun et al., 2026) — modular codebase style

---

## Install

```bash
git clone <this-repo>
cd hjepa_moe
pip install -e ".[dev]"
python smoke_test.py        # validate everything runs on CPU
```

---

## Repo structure

```
hjepa_moe/
│
├── hjepa_moe/                    core package
│   ├── __init__.py               public exports
│   ├── model.py                  HJEPAMoE — main model, full hierarchy
│   ├── model_vjepa2.py           probe mode with frozen V-JEPA 2 encoder
│   │
│   ├── encoders/
│   │   └── temporal.py           TemporalEncoder (attention/mean/conv pooling)
│   │                             Level0Encoder (small ConvNet or V-JEPA 2)
│   │
│   ├── predictors/
│   │   └── moe_predictor.py      MoEPredictor, SwiGLUExpert, TransformerExpert
│   │                             MoERouter (top-k + load balancing)
│   │                             FiLMConditioner (latent variable injection)
│   │
│   ├── losses/
│   │   └── vicreg.py             VICRegLoss, SIGRegLoss, InfoNCELoss
│   │
│   ├── planners/
│   │   ├── cem.py                CEMPlanner, MPPIPlanner
│   │   └── cem_mppi.py           extended MPPI with gradient planning
│   │
│   └── utils/
│       ├── __init__.py           cosine_schedule, AverageMeter, AttentiveProbe
│       └── eval.py               downstream evaluation helpers
│
├── configs/
│   └── video_jepa_moe.yaml       3-level config with ablation sweep grid
│
├── examples/
│   ├── image_jepa/main.py        1-level image SSL (CIFAR-style, ~30min A100)
│   ├── video_jepa/main.py        3-level video prediction (Moving MNIST, ~4h A100)
│   └── ac_video_jepa/main.py     action-conditioned, enc0 frozen + CEM planning
│
├── train_ablate_scale.py         3-in-1 script:
│                                   train   → single/multi-GPU full run
│                                   ablate  → systematic param grid sweep → CSV
│                                   scale   → torchrun DDP entry point
│                                             (1 GPU → 4 → 8 → multi-node configs built-in)
│
├── universal_jepa.py             train JEPA on ANY data shape/modality:
│                                   tabular, sequence, image, video, audio,
│                                   graph, point cloud, HuggingFace datasets
│                                   includes full analysis suite (geometry,
│                                   routing entropy, CKA matrix, kNN, retrieval)
│
├── smoke_test.py                 CPU validation, ~30s, no GPU needed
├── tests/test_hjepa_moe.py       ~35 pytest unit tests
└── pyproject.toml                pip-installable package
```

---

## Quick start

```bash
# Validate repo (CPU, ~30s)
python smoke_test.py

# Train on synthetic video (single GPU)
python train_ablate_scale.py train --cfg configs/video_jepa_moe.yaml

# Run ablation sweep
python train_ablate_scale.py ablate --cfg configs/video_jepa_moe.yaml --steps 2000

# Train on any data (auto-detects shape)
python universal_jepa.py --data mydata.npy
python universal_jepa.py --data mydata.csv --modality tabular
python universal_jepa.py --data ./images   --modality image
python universal_jepa.py --hf_dataset speech_commands --modality audio

# Just run geometry analysis (no training)
python universal_jepa.py --data mydata.npy --probe_only

# Scale to multi-GPU
torchrun --nproc_per_node=4 train_ablate_scale.py scale --scale 4gpu
```

---

## Connecting to V-JEPA 2 (zero training)

```python
import torch
from hjepa_moe.model_vjepa2 import load_probe_model, run_diagnostics

# Load frozen V-JEPA 2 from facebookresearch/vjepa2 (MIT license)
vjepa2 = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vitl')

# Build H-JEPA-MoE on top — no training needed
model = load_probe_model(vjepa2)

# Run full architectural diagnostic
video = torch.randn(2, 64, 3, 256, 256)
run_diagnostics(vjepa2, video)
```

---

## Key ablations to run

Edit `configs/video_jepa_moe.yaml` `sweep.param_grid` then:

```bash
python train_ablate_scale.py ablate --cfg configs/video_jepa_moe.yaml --steps 5000
```

Suggested experiments:
- `n_experts: [1, 2, 4, 8]` — does MoE help vs dense predictor?
- `loss_type: [vicreg, sigreg]` — SIGReg vs VICReg stability
- `pool_factor: [2, 4, 8]` — temporal compression ratio
- `expert_type: [ffn, transformer]` — expert capacity

Results saved to `ablation_results.csv`.

---

## References

- LeCun (2022). A Path Towards Autonomous Machine Intelligence. OpenReview.
- Assran et al. (2025). V-JEPA 2. arXiv:2506.09985
- Balestriero & LeCun (2025). LeJEPA. arXiv:2511.08544
- Lei et al. (2025). M3-JEPA. arXiv:2409.05929. ICML 2025.
- Terver et al. (2026). EB-JEPA. arXiv:2602.03604
- Bardes et al. (2022). VICReg. ICLR 2022.
- Destrade et al. (2025). Value-guided JEPA planning. arXiv:2601.00844
