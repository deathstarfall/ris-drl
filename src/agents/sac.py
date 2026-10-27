import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from src.agents.common import MLP, ReplayBuffer, RunningNorm, to_tensor


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.net = MLP(obs_dim, 2 * act_dim, hidden)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.act_dim = act_dim

    def forward(self, obs):
        out = self.net(obs)
        mu, log_std = out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        return mu, std, log_std


class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net = MLP(obs_dim + act_dim, 1, hidden)

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))


class SAC:
    def __init__(self, cfg, obs_dim, act_dim, N, env, device="cpu"):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.N = N
        self.env = env
        self.device = torch.device(device)
        self.policy = GaussianPolicy(obs_dim, act_dim, cfg["hidden"]).to(self.device)
        self.q1 = QNet(obs_dim, act_dim, cfg["hidden"]).to(self.device)
        self.q2 = QNet(obs_dim, act_dim, cfg["hidden"]).to(self.device)
        self.q1_t = deepcopy(self.q1)
        self.q2_t = deepcopy(self.q2)
        self.opt_p = torch.optim.Adam(self.policy.parameters(), lr=cfg["lr_actor"])
        self.opt_q1 = torch.optim.Adam(self.q1.parameters(), lr=cfg["lr_critic"])
        self.opt_q2 = torch.optim.Adam(self.q2.parameters(), lr=cfg["lr_critic"])
                # DD-007: T=1 contextual bandit - every transition is terminal, the`n        # bootstrap term never fires ((1 - d) = 0), so the target degenerates`n        # to y = r regardless of this value. Kept explicit for clarity.`n        self.gamma = 1.0
        self.tau = float(cfg.get("tau", 0.005))
        self.alpha_auto = bool(cfg.get("alpha_auto", True))
        self.target_entropy = -float(act_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg.get("lr_alpha", 3e-4))
        self.alpha = self.log_alpha.exp().item()
        self.buffer = ReplayBuffer(cfg["buffer"], obs_dim, act_dim)
        self.norm = RunningNorm(obs_dim) if cfg.get("obs_norm", True) else None

    def preprocess(self, obs):
        import numpy as _np
        obs = obs.cpu().numpy() if hasattr(obs, "cpu") else obs
        obs = _np.asarray(obs, dtype=_np.float32).reshape(-1)
        if self.norm is not None:
            self.norm.update(obs)
            obs = self.norm.normalize(obs)
        return obs

    def _sample_action(self, obs_t, deterministic=False):
        mu, std, log_std = self.policy(obs_t)
        if deterministic:
            a = torch.tanh(mu)
            logp = None
        else:
            eps = torch.randn_like(mu)
            a_raw = mu + std * eps
            a = torch.tanh(a_raw)
            logp = (torch.distributions.Normal(mu, std).log_prob(a_raw) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=-1, keepdim=True)
        return a, logp

    def act(self, obs, deterministic=False):
        o = self.preprocess(obs)
        with torch.no_grad():
            a, _ = self._sample_action(to_tensor(o, self.device), deterministic=deterministic)
            a = a.cpu().numpy().reshape(-1)
        # map tanh [-1,1] to [0, 2π]
        a = (a + 1.0) / 2.0 * 2.0 * np.pi
        a = np.clip(a, 0.0, 2.0 * np.pi)
        return a.astype(np.float32)

    def act_deterministic(self, obs):
        return self.act(obs, deterministic=True)

    def store(self, obs, act, rew, obs2, done):
        o = self.preprocess(obs)
        o2 = self.preprocess(obs2)
        a = np.asarray(act, dtype=np.float32).reshape(-1)
        # store in tanh space approx for learning: act_rad -> [-1,1]
        a_n = a / (2.0 * np.pi) * 2.0 - 1.0
        a_n = np.clip(a_n, -1.0, 1.0)
        # better to store the actual policy action; simplified
        self.buffer.store(o, a_n, float(rew), o2, float(done))

    def update(self):
        if self.buffer.size < self.cfg.get("batch", 256):
            return
        batch = self.buffer.sample(self.cfg.get("batch", 256))
        o, a, r, o2, d = [b.to(self.device) for b in batch]
        self.alpha = self.log_alpha.exp().item()
        # critic
        with torch.no_grad():
            a2, logp2 = self._sample_action(o2)
            q1t = self.q1_t(o2, a2)
            q2t = self.q2_t(o2, a2)
            qt = torch.min(q1t, q2t) - self.alpha * logp2
            y = r + self.gamma * (1 - d) * qt
        q1 = self.q1(o, a)
        q2 = self.q2(o, a)
        loss_q = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_q1.zero_grad(); self.opt_q2.zero_grad()
        loss_q.backward()
        self.opt_q1.step(); self.opt_q2.step()
        # actor
        a_pi, logp_pi = self._sample_action(o)
        q1_pi = self.q1(o, a_pi)
        q2_pi = self.q2(o, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        loss_p = (self.alpha * logp_pi - q_pi).mean()
        self.opt_p.zero_grad()
        loss_p.backward()
        self.opt_p.step()
        # alpha
        if self.alpha_auto:
            alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
            self.opt_alpha.zero_grad()
            alpha_loss.backward()
            self.opt_alpha.step()
        # soft update
        tau = self.tau
        for p, p_t in zip(self.q1.parameters(), self.q1_t.parameters()):
            p_t.data.copy_(tau * p.data + (1 - tau) * p_t.data)
        for p, p_t in zip(self.q2.parameters(), self.q2_t.parameters()):
            p_t.data.copy_(tau * p.data + (1 - tau) * p_t.data)

    def state_dict(self):
        return dict(
            policy=self.policy.state_dict(), q1=self.q1.state_dict(), q2=self.q2.state_dict(),
            q1_t=self.q1_t.state_dict(), q2_t=self.q2_t.state_dict(),
            log_alpha=self.log_alpha.detach().cpu(),
            norm=self.norm.state_dict() if self.norm else None, cfg=self.cfg,
        )

    def load_state_dict(self, sd):
        self.policy.load_state_dict(sd["policy"])
        self.q1.load_state_dict(sd["q1"]); self.q2.load_state_dict(sd["q2"])
        if sd.get("q1_t"): self.q1_t.load_state_dict(sd["q1_t"])
        if sd.get("q2_t"): self.q2_t.load_state_dict(sd["q2_t"])
        if "log_alpha" in sd:
            self.log_alpha = sd["log_alpha"].to(self.device).requires_grad_(True)
            self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=self.cfg.get("lr_alpha", 3e-4))
        if sd.get("norm") and self.norm: self.norm.load_state_dict(sd["norm"])

