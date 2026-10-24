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
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_yaml(path):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)
