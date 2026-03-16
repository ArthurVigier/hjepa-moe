"""
Unit tests for H-JEPA-MoE. Run: pytest tests/test_hjepa_moe.py -v
"""
import pytest
import torch
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hjepa_moe.predictors.moe_predictor import MoEPredictor, MoERouter
from hjepa_moe.encoders.temporal import TemporalEncoder, Level0Encoder
from hjepa_moe.losses.vicreg import VICRegLoss, SIGRegLoss, InfoNCELoss
from hjepa_moe.model import HJEPAMoE, HJEPAMoEConfig, LevelConfig
from hjepa_moe.planners.cem import CEMPlanner, MPPIPlanner
from hjepa_moe.utils import AverageMeter, routing_entropy

B = 4

@pytest.fixture
def tiny_config():
    return HJEPAMoEConfig(
        levels=[
            LevelConfig(d_in=64, d_out=64,  pool_factor=4, n_experts=2, top_k=1,
                        expert_type="ffn",         d_z=0,  loss_weight=1.0),
            LevelConfig(d_in=64, d_out=128, pool_factor=4, n_experts=2, top_k=1,
                        expert_type="transformer", d_z=16, loss_weight=2.0),
        ],
        loss_type="vicreg", level0_mode="small", d_level0=64, img_size=32,
        n_rollout_steps=1,
    )

# ── MoEPredictor ──────────────────────────────────────────────

def test_moe_ffn_shape():
    pred = MoEPredictor(d_in=64, d_out=64, d_model=64, n_experts=4, top_k=2, expert_type="ffn")
    out, _ = pred(torch.randn(B, 64))
    assert out.shape == (B, 64)

def test_moe_transformer_shape():
    pred = MoEPredictor(d_in=64, d_out=128, d_model=64, n_experts=2, top_k=1, expert_type="transformer")
    out, _ = pred(torch.randn(B, 64))
    assert out.shape == (B, 128)

def test_moe_with_latent():
    pred = MoEPredictor(d_in=64, d_out=64, n_experts=2, top_k=1, d_z=16)
    out, _ = pred(torch.randn(B, 64), torch.randn(B, 16))
    assert out.shape == (B, 64)

def test_moe_aux_loss_train():
    pred = MoEPredictor(d_in=32, d_out=32, n_experts=4, top_k=2, use_aux_loss=True)
    pred.train()
    _, aux = pred(torch.randn(B, 32))
    assert aux is not None and aux.shape == torch.Size([])

def test_moe_no_aux_eval():
    pred = MoEPredictor(d_in=32, d_out=32, n_experts=4, top_k=2, use_aux_loss=True)
    pred.eval()
    _, aux = pred(torch.randn(B, 32))
    assert aux is None

def test_moe_routing_stats():
    pred = MoEPredictor(d_in=32, d_out=32, n_experts=4, top_k=2)
    stats = pred.get_routing_stats(torch.randn(8, 32))
    assert len(stats["expert_usage"]) == 4

def test_moe_gradients_flow():
    pred = MoEPredictor(d_in=32, d_out=32, n_experts=2, top_k=1)
    x = torch.randn(B, 32, requires_grad=True)
    out, _ = pred(x)
    out.sum().backward()
    assert x.grad is not None

def test_router_gates_sum_to_one():
    router = MoERouter(32, 4, top_k=2, routing_mode="topk", use_aux_loss=False)
    gates, _, _ = router(torch.randn(8, 32))
    assert torch.allclose(gates.sum(-1), torch.ones(8), atol=1e-5)

# ── TemporalEncoder ───────────────────────────────────────────

@pytest.mark.parametrize("mode", ["attention", "mean", "conv"])
def test_temporal_encoder_shape(mode):
    enc = TemporalEncoder(d_in=64, d_out=128, pool_factor=4, pooling=mode)
    out = enc(torch.randn(B, 16, 64))
    assert out.shape == (B, 4, 128), f"{mode}: got {out.shape}"

def test_temporal_encoder_bad_T():
    enc = TemporalEncoder(d_in=32, d_out=32, pool_factor=4)
    with pytest.raises(AssertionError):
        enc(torch.randn(B, 7, 32))

# ── Level0Encoder ─────────────────────────────────────────────

def test_level0_image():
    enc = Level0Encoder(mode="small", d_out=64, img_size=32)
    out = enc(torch.randn(B, 3, 32, 32))
    assert out.shape[0] == B and out.shape[-1] == 64

def test_level0_video():
    enc = Level0Encoder(mode="small", d_out=64, img_size=32, patch_size=8)
    out = enc(torch.randn(B, 4, 3, 32, 32))
    assert out.shape[0] == B and out.shape[-1] == 64

# ── Losses ────────────────────────────────────────────────────

def test_vicreg():
    fn = VICRegLoss()
    loss, stats = fn(torch.randn(B, 64), torch.randn(B, 64))
    assert loss.shape == torch.Size([]) and torch.isfinite(loss)
    assert all(k in stats for k in ["loss_inv", "loss_var", "loss_cov"])

def test_sigreg():
    fn = SIGRegLoss(d_model=64)
    loss, stats = fn(torch.randn(B, 64), torch.randn(B, 64))
    assert torch.isfinite(loss) and "loss_sigreg" in stats

def test_infonce():
    fn = InfoNCELoss()
    loss, stats = fn(torch.randn(B, 64), torch.randn(B, 64))
    assert torch.isfinite(loss) and "acc_top1" in stats

def test_vicreg_detects_collapse():
    fn = VICRegLoss(var_coef=25.0)
    z_bad = torch.ones(32, 64) * 0.5
    _, stats = fn(z_bad, torch.randn(32, 64))
    assert stats["loss_var"] > 1.0

def test_sigreg_sketch_not_param():
    fn = SIGRegLoss(d_model=32, n_sketches=16)
    assert "sketch" not in [n for n, _ in fn.named_parameters()]

# ── HJEPAMoE ─────────────────────────────────────────────────

def test_full_forward(tiny_config):
    model = HJEPAMoE(tiny_config)
    loss, stats = model(torch.randn(2, 16, 3, 32, 32))
    assert loss.shape == torch.Size([])
    assert "level_0" in stats and "level_1" in stats

def test_loss_finite(tiny_config):
    model = HJEPAMoE(tiny_config)
    loss, _ = model(torch.randn(2, 16, 3, 32, 32))
    assert torch.isfinite(loss)

def test_backward(tiny_config):
    model = HJEPAMoE(tiny_config)
    loss, _ = model(torch.randn(2, 16, 3, 32, 32))
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())

def test_ema_update(tiny_config):
    model = HJEPAMoE(tiny_config)
    before = [p.data.clone() for p in model.target_encoders[0].parameters()]
    loss, _ = model(torch.randn(2, 16, 3, 32, 32))
    loss.backward()
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    model.update_ema()
    after = [p.data for p in model.target_encoders[0].parameters()]
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))

def test_get_level_states(tiny_config):
    model = HJEPAMoE(tiny_config)
    states = model.get_level_states(torch.randn(2, 16, 3, 32, 32))
    assert len(states) == len(tiny_config.levels) + 1

def test_sigreg_config():
    cfg = HJEPAMoEConfig(
        levels=[LevelConfig(d_in=32, d_out=32, pool_factor=4, n_experts=2, top_k=1)],
        loss_type="sigreg", d_level0=32, img_size=32,
    )
    loss, _ = HJEPAMoE(cfg)(torch.randn(2, 4, 3, 32, 32))
    assert torch.isfinite(loss)

def test_default_3level_config():
    cfg = HJEPAMoEConfig.default_3level()
    assert len(cfg.levels) == 3
    model = HJEPAMoE(cfg)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n > 0

# ── Planners ──────────────────────────────────────────────────

def test_cem_no_latent(tiny_config):
    model = HJEPAMoE(tiny_config)
    planner = CEMPlanner(model.moe_predictors[0], d_z=0, horizon=3, n_samples=5)
    z, cost = planner.plan(torch.randn(1, 64), torch.randn(1, 64))
    assert isinstance(cost, float)

def test_cem_with_latent(tiny_config):
    model = HJEPAMoE(tiny_config)
    planner = CEMPlanner(model.moe_predictors[1], d_z=16, horizon=3,
                         n_samples=5, n_iters=2)
    z, cost = planner.plan(torch.randn(1, 128), torch.randn(1, 128))
    assert z.shape == (3, 16) and cost >= 0

def test_mppi(tiny_config):
    model = HJEPAMoE(tiny_config)
    planner = MPPIPlanner(model.moe_predictors[1], d_z=16, horizon=3)
    z, cost = planner.plan(torch.randn(1, 128), torch.randn(1, 128), n_iters=3)
    assert z.shape == (3, 16)

# ── Utils ─────────────────────────────────────────────────────

def test_average_meter():
    m = AverageMeter()
    m.update({"loss": 1.0}); m.update({"loss": 3.0})
    assert abs(m.avg()["loss"] - 2.0) < 1e-6

def test_average_meter_reset():
    m = AverageMeter(); m.update({"x": 5.0}); m.reset()
    assert len(m.avg()) == 0

def test_routing_entropy_uniform():
    assert abs(routing_entropy(np.array([.25,.25,.25,.25])) - np.log(4)) < 1e-4

def test_routing_entropy_collapsed():
    assert routing_entropy(np.array([1.0, 0.0, 0.0, 0.0])) < 0.01

def test_smoke_imports():
    from hjepa_moe import HJEPAMoE, TemporalEncoder, MoEPredictor
    from hjepa_moe import VICRegLoss, SIGRegLoss
    assert True
