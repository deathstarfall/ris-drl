"""Gymnasium env. T=1: one channel (context) -> one phase vector (action) -> one reward.
   This is a CONTEXTUAL BANDIT. There is no gamma. The paper says so (DD-007)."""
import gymnasium as gym, numpy as np
from .precoder import sum_rate
from .channel import _cfg

class RISEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, channels, seed=0, N_override=None):
        """channels: list[dict] from channel.load_dataset — pass the LIST, not a path,
        so training and eval can share the same in-memory dataset."""
        self.chs = channels
        self.rng = np.random.default_rng(seed)
        self.N = int(N_override or self.chs[0]["h_r"].shape[1])
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                                (self._obs_size(self.chs[0]),), np.float32)
        self.action_space = gym.spaces.Box(0.0, 2 * np.pi, (self.N,), np.float32)

    @staticmethod
    def _obs_size(ch):
        return ch["h_r"].size * 2 + ch["G"].size + ch["h_d"].size

    @staticmethod
    def _obs(ch):                       # DD-004 contract
        return np.concatenate([np.abs(ch["h_r"]).ravel(),
                               np.angle(ch["h_r"]).ravel(),
                               np.abs(ch["G"]).ravel(),
                               np.abs(ch["h_d"]).ravel()]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ch = self.chs[self.rng.integers(len(self.chs))]   # FIX of old bug #2:
        self.ch["N"] = self.N                                  # channel FIXED for the episode
        return self._obs(self.ch), {}

    def step(self, theta):
        r = sum_rate(np.asarray(theta).reshape(-1), self.ch)
        return self._obs(self.ch), float(r), True, False, {"rate": r}   # T=1: episode ends