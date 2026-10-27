import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from src.agents.common import MLP, ReplayBuffer, RunningNorm, to_tensor
from src.agents.ddpg import Actor, Critic


class TD3:
    def __init__(self, cfg, obs_dim, act_dim, N, env, device="cpu"):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.N = N
        self.env = env
        self.device = torch.device(device)
        self.actor = Actor(obs_dim, act_dim, cfg["hidden"]).to(self.device)
        self.critic1 = Critic(obs_dim, act_dim, cfg["hidden"]).to(self.device)
        self.critic2 = Critic(obs_dim, act_dim, cfg["hidden"]).to(self.device)
        self.actor_t = deepcopy(self.actor)
        self.critic1_t = deepcopy(self.critic1)
        self.critic2_t = deepcopy(self.critic2)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg["lr_actor"])
        self.opt_c1 = torch.optim.Adam(self.critic1.parameters(), lr=cfg["lr_critic"])
        self.opt_c2 = torch.optim.Adam(self.critic2.parameters(), lr=cfg["lr_critic"])
                # DD-007: T=1 contextual bandit - every transition is terminal, the`n        # bootstrap term never fires ((1 - d) = 0), so the target degenerates`n        # to y = r regardless of this value. Kept explicit for clarity.`n        self.gamma = 1.0
        self.tau = float(cfg.get("tau", 0.005))
        self.expl_noise = float(cfg.get("expl_noise", 0.1))
        self.policy_delay = int(cfg.get("policy_delay", 2))
        self.target_noise = float(cfg.get("target_noise", 0.2))
        self.noise_clip = float(cfg.get("noise_clip", 0.5))
        self.buffer = ReplayBuffer(cfg["buffer"], obs_dim, act_dim)
        self.norm = RunningNorm(obs_dim) if cfg.get("obs_norm", True) else None
        self.it = 0

    def preprocess(self, obs):
        import numpy as _np
        obs = obs.cpu().numpy() if hasattr(obs, "cpu") else obs
        obs = _np.asarray(obs, dtype=_np.float32).reshape(-1)
        if self.norm is not None:
            self.norm.update(obs)
            obs = self.norm.normalize(obs)
        return obs

    def act(self, obs, deterministic=False):
        o = self.preprocess(obs)
        with torch.no_grad():
            a = self.actor(to_tensor(o, self.device)).cpu().numpy().reshape(-1)
        if not deterministic:
            a = a + self.expl_noise * np.random.randn(*a.shape)
        a = (a + 1.0) / 2.0 * 2.0 * np.pi
        a = np.clip(a, 0.0, 2.0 * np.pi)
        return a.astype(np.float32)

    def act_deterministic(self, obs):
        return self.act(obs, deterministic=True)

    def store(self, obs, act, rew, obs2, done):
        o = self.preprocess(obs)
        o2 = self.preprocess(obs2)
        a = np.asarray(act, dtype=np.float32).reshape(-1)
        a_n = a / (2.0 * np.pi) * 2.0 - 1.0
        a_n = np.clip(a_n, -1.0, 1.0)
        self.buffer.store(o, a_n, float(rew), o2, float(done))

    def update(self):
        if self.buffer.size < self.cfg.get("batch", 256):
            return
        batch = self.buffer.sample(self.cfg.get("batch", 256))
        o, a, r, o2, d = [b.to(self.device) for b in batch]
        with torch.no_grad():
            noise = (torch.randn_like(a) * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
            a2 = (self.actor_t(o2) + noise).clamp(-1, 1)
            q1t = self.critic1_t(o2, a2)
            q2t = self.critic2_t(o2, a2)
            qt = torch.min(q1t, q2t)
            y = r + self.gamma * (1 - d) * qt
        q1 = self.critic1(o, a)
        q2 = self.critic2(o, a)
        loss_c = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_c1.zero_grad(); self.opt_c2.zero_grad()
        loss_c.backward()
        self.opt_c1.step(); self.opt_c2.step()
        self.it += 1
        if self.it % self.policy_delay == 0:
            a_pi = self.actor(o)
            loss_a = -self.critic1(o, a_pi).mean()
            self.opt_a.zero_grad()
            loss_a.backward()
            self.opt_a.step()
            self._soft_update(self.actor, self.actor_t)
            self._soft_update(self.critic1, self.critic1_t)
            self._soft_update(self.critic2, self.critic2_t)

    def _soft_update(self, net, net_t):
        tau = self.tau
        for p, p_t in zip(net.parameters(), net_t.parameters()):
            p_t.data.copy_(tau * p.data + (1 - tau) * p_t.data)

    def state_dict(self):
        return dict(
            actor=self.actor.state_dict(), critic1=self.critic1.state_dict(), critic2=self.critic2.state_dict(),
            actor_t=self.actor_t.state_dict(), critic1_t=self.critic1_t.state_dict(), critic2_t=self.critic2_t.state_dict(),
            norm=self.norm.state_dict() if self.norm else None, cfg=self.cfg,
        )

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"])
        self.critic1.load_state_dict(sd["critic1"])
        self.critic2.load_state_dict(sd["critic2"])
        if sd.get("actor_t"): self.actor_t.load_state_dict(sd["actor_t"])
        if sd.get("critic1_t"): self.critic1_t.load_state_dict(sd["critic1_t"])
        if sd.get("critic2_t"): self.critic2_t.load_state_dict(sd["critic2_t"])
        if sd.get("norm") and self.norm: self.norm.load_state_dict(sd["norm"])

