import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from src.agents.common import MLP, ReplayBuffer, to_tensor


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net = MLP(obs_dim, act_dim, hidden)
        # outputs in [-1,1] -> scaled to [0, 2π] by env/action scaling logic outside
    def forward(self, obs):
        a = torch.tanh(self.net(obs))
        return a


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net = MLP(obs_dim + act_dim, 1, hidden)
    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))


class DDPG:
    def __init__(self, cfg, obs_dim, act_dim, N, env, device="cpu"):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.N = N
        self.env = env
        self.device = torch.device(device)
        self.actor = Actor(obs_dim, act_dim, cfg["hidden"]).to(self.device)
        self.critic = Critic(obs_dim, act_dim, cfg["hidden"]).to(self.device)
        self.actor_t = deepcopy(self.actor)
        self.critic_t = deepcopy(self.critic)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg["lr_actor"])
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg["lr_critic"])
        self.gamma = float(cfg.get("gamma", 1.0))
        self.tau = float(cfg.get("tau", 0.005))
        self.expl_noise = float(cfg.get("expl_noise", 0.1))
        # buffer handled by base via train_runner? train_runner stores via agent.store; better attach
        from src.agents.common import ReplayBuffer
        self.buffer = ReplayBuffer(cfg["buffer"], obs_dim, act_dim)
        # running norm
        from src.agents.common import RunningNorm
        self.norm = RunningNorm(obs_dim) if cfg.get("obs_norm", True) else None

    def preprocess(self, obs):
        obs = obs.cpu().numpy() if hasattr(obs, "cpu") else obs
        import numpy as _np
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
        # map from [-1,1] to [0, 2π] (env expects radians)
        a = (a + 1.0) / 2.0 * 2.0 * np.pi
        # clip
        a = np.clip(a, 0.0, 2.0 * np.pi)
        return a.astype(np.float32)

    def act_deterministic(self, obs):
        return self.act(obs, deterministic=True)

    def store(self, obs, act, rew, obs2, done):
        o = self.preprocess(obs)
        o2 = self.preprocess(obs2)
        a = np.asarray(act, dtype=np.float32).reshape(-1)
        # normalize act to [-1,1] for policy? but env takes radians; easier to store normalized
        a_n = a / (2.0 * np.pi) * 2.0 - 1.0
        a_n = np.clip(a_n, -1.0, 1.0)
        self.buffer.store(o, a_n, float(rew), o2, float(done))

    def update(self):
        if self.buffer.size < self.cfg.get("batch", 256):
            return
        batch = self.buffer.sample(self.cfg.get("batch", 256))
        o, a, r, o2, d = [b.to(self.device) for b in batch]
        with torch.no_grad():
            a2 = self.actor_t(o2)
            q2 = self.critic_t(o2, a2)
            y = r + self.gamma * (1 - d) * q2
        q = self.critic(o, a)
        loss_c = F.mse_loss(q, y)
        self.opt_c.zero_grad()
        loss_c.backward()
        self.opt_c.step()
        a_pi = self.actor(o)
        loss_a = -self.critic(o, a_pi).mean()
        self.opt_a.zero_grad()
        loss_a.backward()
        self.opt_a.step()
        self._soft_update(self.actor, self.actor_t)
        self._soft_update(self.critic, self.critic_t)

    def _soft_update(self, net, net_t):
        tau = self.tau
        for p, p_t in zip(net.parameters(), net_t.parameters()):
            p_t.data.copy_(tau * p.data + (1 - tau) * p_t.data)

    def state_dict(self):
        return dict(
            actor=self.actor.state_dict(),
            critic=self.critic.state_dict(),
            actor_t=self.actor_t.state_dict(),
            critic_t=self.critic_t.state_dict(),
            norm=self.norm.state_dict() if self.norm else None,
            cfg=self.cfg,
        )

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        if sd.get("actor_t"): self.actor_t.load_state_dict(sd["actor_t"])
        if sd.get("critic_t"): self.critic_t.load_state_dict(sd["critic_t"])
        if sd.get("norm") and self.norm: self.norm.load_state_dict(sd["norm"])
