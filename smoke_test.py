#!/usr/bin/env python
"""
Smoke test: validates the full repo runs end-to-end on CPU in <30 seconds.
Run: python smoke_test.py

Checks:
  1. All imports resolve
  2. MoEPredictor forward (ffn + transformer)
  3. TemporalEncoder forward (all 3 pooling modes)
  4. VICRegLoss + SIGRegLoss
  5. HJEPAMoE full forward + backward
  6. EMA update
  7. CEMPlanner (d_z=0 and d_z>0)
  8. AverageMeter
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch

PASS = "  [OK]"
FAIL = "  [FAIL]"

def check(name, fn):
    try:
        fn()
        print(f"{PASS}  {name}")
        return True
    except Exception as e:
        print(f"{FAIL}  {name}: {e}")
        return False

results = []
t0 = time.time()

# 1. Imports
def _imports():
    from hjepa_moe import HJEPAMoE, TemporalEncoder, MoEPredictor, VICRegLoss, SIGRegLoss
    from hjepa_moe.planners.cem import CEMPlanner, MPPIPlanner
    from hjepa_moe.utils import AverageMeter, routing_entropy, cosine_schedule

results.append(check("Imports", _imports))

# 2. MoEPredictor FFN
def _moe_ffn():
    from hjepa_moe.predictors.moe_predictor import MoEPredictor
    p = MoEPredictor(d_in=32, d_out=32, d_model=32, n_experts=4, top_k=2)
    out, _ = p(torch.randn(4, 32))
    assert out.shape == (4, 32)

results.append(check("MoEPredictor FFN", _moe_ffn))

# 3. MoEPredictor Transformer
def _moe_transformer():
    from hjepa_moe.predictors.moe_predictor import MoEPredictor
    p = MoEPredictor(d_in=32, d_out=64, d_model=32, n_experts=2, top_k=1,
                     expert_type="transformer")
    out, _ = p(torch.randn(4, 32))
    assert out.shape == (4, 64)

results.append(check("MoEPredictor Transformer", _moe_transformer))

# 4. TemporalEncoder all modes
def _temporal_enc():
    from hjepa_moe.encoders.temporal import TemporalEncoder
    for mode in ["attention", "mean", "conv"]:
        enc = TemporalEncoder(32, 64, pool_factor=4, pooling=mode)
        out = enc(torch.randn(2, 8, 32))
        assert out.shape == (2, 2, 64), f"{mode}: {out.shape}"

results.append(check("TemporalEncoder (all modes)", _temporal_enc))

# 5. Losses
def _losses():
    from hjepa_moe.losses.vicreg import VICRegLoss, SIGRegLoss, InfoNCELoss
    z1, z2 = torch.randn(8, 32), torch.randn(8, 32)
    l1, _ = VICRegLoss()(z1, z2);  assert torch.isfinite(l1)
    l2, _ = SIGRegLoss(d_model=32)(z1, z2); assert torch.isfinite(l2)
    l3, _ = InfoNCELoss()(z1, z2); assert torch.isfinite(l3)

results.append(check("Losses (VICReg + SIGReg + InfoNCE)", _losses))

# 6. Full forward + backward
def _full_forward():
    from hjepa_moe.model import HJEPAMoE, HJEPAMoEConfig, LevelConfig
    cfg = HJEPAMoEConfig(
        levels=[
            LevelConfig(d_in=32, d_out=32, pool_factor=4, n_experts=2,
                        top_k=1, expert_type="ffn", d_z=0),
        ],
        loss_type="vicreg", d_level0=32, img_size=16, n_rollout_steps=0,
    )
    model = HJEPAMoE(cfg)
    x = torch.randn(2, 4, 3, 16, 16)
    loss, stats = model(x)
    assert torch.isfinite(loss), f"loss={loss.item()}"
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())

results.append(check("HJEPAMoE forward + backward", _full_forward))

# 7. EMA update
def _ema():
    from hjepa_moe.model import HJEPAMoE, HJEPAMoEConfig, LevelConfig
    cfg = HJEPAMoEConfig(
        levels=[LevelConfig(d_in=32, d_out=32, pool_factor=4, n_experts=2, top_k=1)],
        loss_type="vicreg", d_level0=32, img_size=16, n_rollout_steps=0,
    )
    model = HJEPAMoE(cfg)
    before = [p.data.clone() for p in model.target_encoders[0].parameters()]
    loss, _ = model(torch.randn(2, 4, 3, 16, 16))
    loss.backward()
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    model.update_ema()
    after = [p.data for p in model.target_encoders[0].parameters()]
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))

results.append(check("EMA update", _ema))

# 8. CEM planner
def _cem():
    from hjepa_moe.model import HJEPAMoE, HJEPAMoEConfig, LevelConfig
    from hjepa_moe.planners.cem import CEMPlanner
    cfg = HJEPAMoEConfig(
        levels=[
            LevelConfig(d_in=32, d_out=32, pool_factor=4, n_experts=2, top_k=1, d_z=0),
            LevelConfig(d_in=32, d_out=64, pool_factor=4, n_experts=2, top_k=1, d_z=8),
        ],
        loss_type="vicreg", d_level0=32, img_size=16, n_rollout_steps=0,
    )
    model = HJEPAMoE(cfg)
    # d_z=0
    p0 = CEMPlanner(model.moe_predictors[0], d_z=0, horizon=2, n_samples=4, n_iters=2)
    z, cost = p0.plan(torch.randn(1, 32), torch.randn(1, 32))
    assert isinstance(cost, float)
    # d_z=8
    p1 = CEMPlanner(model.moe_predictors[1], d_z=8, horizon=2, n_samples=4, n_iters=2)
    z, cost = p1.plan(torch.randn(1, 64), torch.randn(1, 64))
    assert z.shape == (2, 8) and cost >= 0

results.append(check("CEMPlanner (d_z=0 and d_z=8)", _cem))

# 9. AverageMeter
def _meter():
    from hjepa_moe.utils import AverageMeter
    m = AverageMeter()
    m.update({"loss": 1.0, "acc": 0.9})
    m.update({"loss": 3.0, "acc": 0.7})
    avg = m.avg()
    assert abs(avg["loss"] - 2.0) < 1e-6

results.append(check("AverageMeter", _meter))

# ── Summary ───────────────────────────────────────────────────
elapsed = time.time() - t0
n_pass  = sum(results)
n_total = len(results)
print(f"\n{'='*40}")
print(f"Smoke test: {n_pass}/{n_total} passed in {elapsed:.1f}s")
if n_pass == n_total:
    print("All checks passed. Repo is functional.")
else:
    print("Some checks failed — see above.")
    sys.exit(1)
