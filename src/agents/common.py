import math
import os
import time
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RunningNorm:
    """Running mean/std for the frozen observation vector."""
    def __init__(self, dim, eps=1e-5):
        self.dim = dim
        self.eps = eps
        self.count = 0
        self.mu = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)

    def update(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.dim:
            return
        self.count += 1
        d = x - self.mu
        self.mu += d / self.count
        self.M2 += d * (x - self.mu)

    @property
    def var(self):
        if self.count < 2:
            return np.ones(self.dim, dtype=np.float32)
        return (self.M2 / self.count).astype(np.float32)

    @property
    def std(self):
        s = np.sqrt(self.var.astype(np.float64)) + self.eps
        return s.astype(np.float32)

    def normalize(self, x):
        if self.count == 0:
            return np.asarray(x, dtype=np.float32).reshape(-1)
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        return (x - self.mu.astype(np.float32)) / (self.std + self.eps)

    def state_dict(self):
        return dict(count=self.count, mu=self.mu.copy(), M2=self.M2.copy(), dim=self.dim, eps=self.eps)

    def load_state_dict(self, sd):
        self.count = int(sd.get("count", 0))
        self.mu = np.array(sd.get("mu", np.zeros(self.dim)))
        self.M2 = np.array(sd.get("M2", np.zeros(self.dim)))
        self.dim = int(sd.get("dim", self.dim))
        self.eps = float(sd.get("eps", self.eps))


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.obs2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def store(self, obs, act, rew, obs2, done):
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.obs2[self.ptr] = obs2
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch):
        idx = np.random.randint(0, self.size, size=batch)
        return (
            torch.as_tensor(self.obs[idx]),
            torch.as_tensor(self.act[idx]),
            torch.as_tensor(self.rew[idx]),
            torch.as_tensor(self.obs2[idx]),
            torch.as_tensor(self.done[idx]),
        )


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, activation=nn.ReLU):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), activation()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def to_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def write_json(path, obj):
    import json
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w",encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_yaml(path):
    import yaml
    with open(path, "r",encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_policy_fn(agent_name, seed, N):
    """Reconstructs the agent policy from a checkpoint for evaluation (DD-008)."""
    import torch
    import numpy as np
    
    ckpt_path = f"ckpt/{agent_name}_seed{seed}.pt"
    
    # Load checkpoint
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = sd.get("cfg", {})
    
    # Infer obs_dim from the saved RunningNorm state dict, or fallback to a safe estimate
    obs_dim = sd.get("norm", {}).get("dim", N * 4)
    
    # Dynamically import and instantiate the correct agent
    if agent_name == "sac":
        from src.agents.sac import SAC
        agent = SAC(cfg, obs_dim, N, N, env=None, device="cpu")
    elif agent_name == "td3":
        from src.agents.td3 import TD3
        agent = TD3(cfg, obs_dim, N, N, env=None, device="cpu")
    elif agent_name == "ddpg":
        from src.agents.ddpg import DDPG
        agent = DDPG(cfg, obs_dim, N, N, env=None, device="cpu")
    else: # covers dqn, dqn_codebook, etc.
        from src.agents.dqn_codebook import DQNCodebook
        agent = DQNCodebook(cfg, obs_dim, N, N, env=None, device="cpu")
        
    # Load the trained weights
    agent.load_state_dict(sd)
    
    def policy_fn(ch):
        # Reconstruct the observation exactly as DD-004 specifies:
        # obs = concat(|h_r|, ∠h_r, |G|, |h_d|) flat float32
        h_r = ch["h_r"]
        G = ch["G"]
        h_d = ch["h_d"]
        
        obs = np.concatenate([
            np.abs(h_r).flatten(),
            np.angle(h_r).flatten(),
            np.abs(G).flatten(),
            np.abs(h_d).flatten()
        ]).astype(np.float32)
        
        # agent.act returns the action in radians [0, 2π], which sum_rate expects
        return agent.act(obs)
        
    return policy_fn