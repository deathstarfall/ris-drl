"""Gymnasium env. T=1: one channel (context) -> one phase vector (action) -> one reward.
   This is a CONTEXTUAL BANDIT. There is no gamma. The paper says so (DD-007)."""
import gymnasium as gym, numpy as np
from .precoder import sum_rate
from .channel import _cfg

class RISEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, channels, seed=0, N_override=None, obs_mode="partial"):
        """channels: list[dict] from channel.load_dataset — pass the LIST, not a path,
        so training and eval can share the same in-memory dataset.
        obs_mode (DD-041, audit item 6):
          "partial" (default, matches all published artifacts): |h_r|, angle(h_r), |G|, |h_d| —
          the angles of G and h_d are NOT observed by the agents.
          "full": adds angle(G) and angle(h_d); provided for v1.1 retraining campaigns.
        """
        self.chs = channels
        self.rng = np.random.default_rng(seed)
        self.N = int(N_override or self.chs[0]["h_r"].shape[1])
        self.obs_mode = "full" if str(obs_mode).lower() == "full" else "partial"
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                                (self._obs_size(self.chs[0], self.obs_mode),), np.float32)
        self.action_space = gym.spaces.Box(0.0, 2 * np.pi, (self.N,), np.float32)

    @staticmethod
    def _obs_size(ch, obs_mode="partial"):
        n = ch["h_r"].size * 2 + ch["G"].size + ch["h_d"].size
        if obs_mode == "full":
            n += ch["G"].size + ch["h_d"].size
        return n

    @staticmethod
    def _obs(ch, obs_mode="partial"):   # DD-004 contract (partial) / DD-041 full option
        parts = [np.abs(ch["h_r"]).ravel(),
                 np.angle(ch["h_r"]).ravel(),
                 np.abs(ch["G"]).ravel(),
                 np.abs(ch["h_d"]).ravel()]
        if obs_mode == "full":
            parts += [np.angle(ch["G"]).ravel(), np.angle(ch["h_d"]).ravel()]
        return np.concatenate(parts).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ch = self.chs[self.rng.integers(len(self.chs))]   # FIX of old bug #2:
        self.ch["N"] = self.N                                  # channel FIXED for the episode
        return self._obs(self.ch), {}

    def step(self, theta):
        r = sum_rate(np.asarray(theta).reshape(-1), self.ch)
        return self._obs(self.ch), float(r), True, False, {"rate": r}   # T=1: episode ends