"""
H-JEPA-MoE: Action-conditioned video example.
Mirrors V-JEPA 2-AC post-training: freeze level-0 encoder,
train action-conditioned MoE predictor at level 1.

Run: python -m examples.ac_video_jepa.main
"""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hjepa_moe.model import HJEPAMoE, HJEPAMoEConfig, LevelConfig
from hjepa_moe.planners.cem import CEMPlanner
from hjepa_moe.utils import AverageMeter, cosine_schedule, set_lr

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2-level model with latent z at level 1 (enables planning)
    cfg = HJEPAMoEConfig(
        levels=[
            LevelConfig(d_in=128, d_out=128, pool_factor=4,
                        n_experts=4, top_k=2, d_z=0,  loss_weight=1.0),
            LevelConfig(d_in=128, d_out=256, pool_factor=4,
                        n_experts=4, top_k=2, d_z=32, loss_weight=2.0,
                        expert_type="transformer"),
        ],
        loss_type="vicreg", level0_mode="small",
        d_level0=128, img_size=64, n_rollout_steps=2, rollout_weight=0.5,
    )
    model = HJEPAMoE(cfg).to(device)

    # Freeze level-0 encoder (AC post-training style)
    for p in model.enc0.parameters():
        p.requires_grad_(False)

    opt   = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=0.05
    )
    meter = AverageMeter()
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for step in range(500):
        lr = cosine_schedule(step, 500, 50, 1e-4)
        set_lr(opt, lr)
        # Replace with real robot trajectory DataLoader (e.g. Droid dataset)
        x = torch.randn(4, 16, 3, 64, 64, device=device)
        loss, stats = model(x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); model.update_ema()
        meter.update({"loss": loss.item()})
        if step % 100 == 0:
            print(f"step {step:3d} | {meter.avg()}")
            meter.reset()

    # Demo: plan from random current state to random goal at level 1
    model.eval()
    planner = CEMPlanner(model.moe_predictors[1], d_z=32, horizon=5,
                         n_samples=50, n_iters=3, device=str(device))
    s0     = torch.randn(1, 256, device=device)
    s_goal = torch.randn(1, 256, device=device)
    z_seq, cost = planner.plan(s0, s_goal)
    print(f"\nPlanning result: horizon={z_seq.shape[0]}, cost={cost:.4f}")

if __name__ == "__main__":
    main()
