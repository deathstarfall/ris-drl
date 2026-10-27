import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from src.agents.common import MLP, ReplayBuffer, RunningNorm, to_tensor


class NHeadDQN(nn.Module):
    def __init__(self, obs_dim, N, num_actions):
        super().__init__()
        self.N = N
        self.num_actions = num_actions
        self.backbone = MLP(obs_dim, 256, [256, 256])
        self.heads = nn.ModuleList([nn.Linear(256, num_actions) for _ in range(N)])

    def forward(self, x):
        z = self.backbone(x)
        # return list of Q values per element
        return [h(z) for h in self.heads]


class DQNCodebook:
    """
    Codebook DQN: per-element codebook, N heads, each of size 2^B.
    Design (a) frozen D0. Action is a vector of discrete codes per element -> converted to phases.
    """
    def __init__(self, cfg, obs_dim, act_dim, N, env, device="cpu"):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.N = N
        self.env = env
        self.device = torch.device(device)
        self.bits = int(cfg.get("bits", 2))
        self.num_actions = 2 ** self.bits
        # phase codebook
        self.codebook = np.linspace(0, 2 * np.pi, self.num_actions, endpoint=False).astype(np.float32)
        self.q = NHeadDQN(obs_dim, N, self.num_actions).to(self.device)
        self.q_t = deepcopy(self.q)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg["lr"])
                # DD-007: T=1 contextual bandit - every transition is terminal, the`n        # bootstrap term never fires ((1 - d) = 0), so the target degenerates`n        # to y = r regardless of this value. Kept explicit for clarity.`n        self.gamma = 1.0
        self.target_update = int(cfg.get("target_update", 1000))
        self.eps_start = float(cfg.get("eps_start", 1.0))
        self.eps_end = float(cfg.get("eps_end", 0.05))
        self.eps_decay_steps = int(cfg.get("eps_decay_steps", 100000))
        self.buffer = ReplayBuffer(cfg["buffer"], obs_dim, N)  # store discrete actions per element
        self.norm = RunningNorm(obs_dim) if cfg.get("obs_norm", True) else None
        self.total_steps = 0

    def preprocess(self, obs):
        import numpy as _np
        obs = obs.cpu().numpy() if hasattr(obs, "cpu") else obs
        obs = _np.asarray(obs, dtype=_np.float32).reshape(-1)
        if self.norm is not None:
            self.norm.update(obs)
            obs = self.norm.normalize(obs)
        return obs

    def eps(self):
        t = min(self.total_steps, self.eps_decay_steps)
        frac = t / self.eps_decay_steps if self.eps_decay_steps > 0 else 1.0
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, obs, deterministic=False):
        o = self.preprocess(obs)
        self.total_steps += 1
        if not deterministic and np.random.rand() < self.eps():
            # random per element
            codes = np.random.randint(0, self.num_actions, size=self.N)
        else:
            with torch.no_grad():
                qs = self.q(to_tensor(o, self.device))
                codes = []
                for qh in qs:
                    codes.append(int(qh.argmax(dim=-1).item()))
                codes = np.array(codes)
        # convert to phases
        phases = self.codebook[codes]
        return phases.astype(np.float32)

    def act_deterministic(self, obs):
        return self.act(obs, deterministic=True)

    def store(self, obs, act, rew, obs2, done):
        o = self.preprocess(obs)
        o2 = self.preprocess(obs2)
        # map phases back to nearest code
        a = np.asarray(act, dtype=np.float32).reshape(-1)
        # find nearest in codebook
        codes = []
        for val in a:
            # wrap
            v = ((val % (2 * np.pi)) + 2 * np.pi) % (2 * np.pi)
            d = np.abs(self.codebook - v)
            codes.append(int(np.argmin(d)))
        codes = np.array(codes, dtype=np.float32)
        self.buffer.store(o, codes, float(rew), o2, float(done))

    def update(self):
        if self.buffer.size < self.cfg.get("batch", 256):
            return
        batch = self.buffer.sample(self.cfg.get("batch", 256))
        o, a_codes, r, o2, d = [b.to(self.device) for b in batch]
        a_codes = a_codes.long()
        with torch.no_grad():
            qs2 = self.q_t(o2)
            # greedy next action per head
            next_codes = []
            for qh in qs2:
                next_codes.append(qh.argmax(dim=-1, keepdim=True))
            next_codes = torch.cat(next_codes, dim=-1)
            # take Q value of next action
            q2 = 0
            for i in range(self.N):
                q2 = q2 + qs2[i].gather(1, next_codes[:, i:i + 1])
            y = r + self.gamma * (1 - d) * q2
        # current Q
        qs = self.q(o)
        q1 = 0
        for i in range(self.N):
            q1 = q1 + qs[i].gather(1, a_codes[:, i:i + 1])
        loss = F.smooth_l1_loss(q1, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        if self.total_steps % self.target_update == 0:
            self.q_t.load_state_dict(self.q.state_dict())

    def state_dict(self):
        return dict(
            q=self.q.state_dict(), q_t=self.q_t.state_dict(),
            norm=self.norm.state_dict() if self.norm else None,
            codebook=self.codebook, bits=self.bits, cfg=self.cfg,
        )

    def load_state_dict(self, sd):
        self.q.load_state_dict(sd["q"])
        if sd.get("q_t"): self.q_t.load_state_dict(sd["q_t"])
        if sd.get("norm") and self.norm: self.norm.load_state_dict(sd["norm"])
        if sd.get("codebook") is not None:
            self.codebook = np.array(sd["codebook"])
        if sd.get("bits"): self.bits = int(sd["bits"]); self.num_actions = 2 ** self.bits

