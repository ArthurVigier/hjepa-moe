"""CEM and MPPI planners — see docstrings in each class."""
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class CEMPlanner:
    """
    Cross-Entropy Method planner operating in JEPA latent space.
    Mirrors V-JEPA 2-AC MPC loop (16s/action vs Cosmos 4min/action).
    """
    def __init__(self, predictor, d_z, horizon=10, n_samples=200,
                 n_iters=5, elite_frac=0.1, noise_decay=0.5, device="cpu"):
        self.pred = predictor
        self.d_z = d_z; self.H = horizon
        self.n_samples = n_samples; self.n_iters = n_iters
        self.elite_k = max(1, int(n_samples * elite_frac))
        self.noise_decay = noise_decay; self.device = device

    @torch.no_grad()
    def plan(self, s0, s_goal, value_fn=None):
        if self.d_z == 0:
            s = s0
            for _ in range(self.H): s, _ = self.pred(s)
            return torch.zeros(self.H, 0, device=self.device), F.mse_loss(s, s_goal).item()

        z_mean = torch.zeros(self.H, self.d_z, device=self.device)
        z_std  = torch.ones (self.H, self.d_z, device=self.device)

        for it in range(self.n_iters):
            noise = torch.randn(self.n_samples, self.H, self.d_z, device=self.device)
            z_pop = z_mean.unsqueeze(0) + z_std.unsqueeze(0) * noise
            costs = torch.zeros(self.n_samples, device=self.device)
            for i in range(self.n_samples):
                s = s0.clone()
                for t in range(self.H): s, _ = self.pred(s, z_pop[i, t:t+1])
                costs[i] = F.mse_loss(s, s_goal)
                if value_fn is not None: costs[i] -= 0.1 * value_fn(s, s_goal)
            elite_idx = costs.argsort()[:self.elite_k]
            elites = z_pop[elite_idx]
            z_mean = elites.mean(0)
            z_std  = elites.std(0).clamp(min=1e-4) * self.noise_decay

        return z_mean, costs[elite_idx[0]].item()

    @torch.no_grad()
    def mpc_step(self, s0, s_goal, value_fn=None):
        z_seq, cost = self.plan(s0, s_goal, value_fn)
        z1 = z_seq[0:1]
        s1, _ = self.pred(s0, z1)
        return z1, s1, cost


class MPPIPlanner:
    """Gradient-based MPPI planner through predictor."""
    def __init__(self, predictor, d_z, horizon=10, device="cpu"):
        self.pred = predictor; self.d_z = d_z
        self.H = horizon; self.device = device

    def plan(self, s0, s_goal, n_iters=10, lr=0.05):
        if self.d_z == 0:
            return torch.zeros(self.H, 0, device=self.device), 0.0
        z_seq = torch.zeros(self.H, self.d_z, device=self.device, requires_grad=True)
        opt = torch.optim.Adam([z_seq], lr=lr)
        for _ in range(n_iters):
            s = s0.clone()
            for t in range(self.H): s, _ = self.pred(s, z_seq[t:t+1])
            cost = F.mse_loss(s, s_goal)
            opt.zero_grad(); cost.backward(); opt.step()
        return z_seq.detach(), cost.item()
