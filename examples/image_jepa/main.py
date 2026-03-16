"""
H-JEPA-MoE: Image representation learning example (CIFAR-10 style).
Single level, small MoE predictor. Trains in ~30min on A100.

Run: python -m examples.image_jepa.main
"""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hjepa_moe.model import HJEPAMoE, HJEPAMoEConfig, LevelConfig
from hjepa_moe.utils import AverageMeter, cosine_schedule, set_lr

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = HJEPAMoEConfig(
        levels=[
            LevelConfig(d_in=128, d_out=128, pool_factor=4,
                        n_experts=4, top_k=2, expert_type="ffn",
                        pooling="mean", loss_weight=1.0),
        ],
        loss_type="sigreg", level0_mode="small",
        d_level0=128, img_size=32, n_rollout_steps=0,
    )
    model = HJEPAMoE(cfg).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    meter = AverageMeter()

    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for step in range(2000):
        lr = cosine_schedule(step, 2000, 200, 3e-4)
        set_lr(opt, lr)
        # Placeholder: replace with real CIFAR DataLoader
        x = torch.randn(32, 4, 3, 32, 32, device=device)  # fake seq of 4 frames
        loss, stats = model(x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); model.update_ema()
        meter.update({"loss": loss.item()})
        if step % 200 == 0:
            print(f"step {step:4d} | {meter.avg()}")
            meter.reset()

if __name__ == "__main__":
    main()
