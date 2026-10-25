import os
import json
import time
from typing import Optional
from src.channel import load_dataset
import numpy as np
import torch

from src.agents.common import RunningNorm, ReplayBuffer, write_json, load_yaml, to_tensor


class BanditWrapper:
    """
    Thin wrapper around Ayansh's RISEnv.
    The env is a contextual bandit (T=1): reset() gives a channel (context),
    step(action) returns (obs, reward, terminated=True, truncated=False, info).
    If Ayansh's env is not importable yet, this can be swapped; agents only
    depend on the interface above.
    """
    def __init__(self, train_channels: str = "data/train_channels.npz", seed: int = 0):
        self.seed = seed
        try:
            from src.ris_env import RISEnv
            channels_list = load_dataset(train_channels)
            self.env = RISEnv(channels=channels_list, seed=seed)
            self.obs_dim = int(np.prod(self.env.observation_space.shape))
            self.act_dim = int(np.prod(self.env.action_space.shape))
            self.N = self.env.N
        except Exception as e:
            # Fallback stub: only used if env not present. Ayansh must replace with real env.
            raise RuntimeError(f"RISEnv import failed (Ayansh must provide src/ris_env.py): {e}")


def evaluate_on_frozen_test_set(agent, test_channels: str = "data/test_channels.npz", N_eval: Optional[int] = None):
    """
    Evaluate the agent deterministically on the ONE frozen 500-channel test set.
    Uses the same evaluator entry point as baselines (Ayansh's evaluate.py preferred).
    """
    try:
        from src.evaluate import evaluate_policy
        return evaluate_policy(agent, test_channels, N_eval=N_eval)
    except Exception:
        pass

    # Fallback evaluation if evaluate.py isn't ready yet.
    try:
        from src.ris_env import RISEnv
        channels_list = load_dataset(test_channels)
        env = RISEnv(channels=channels_list, seed=12345)
        total = 0.0
        n = N_eval if N_eval is not None else len(env.chs)
        for i in range(n):
            obs, _ = env.reset(seed=12345 + i)
            if hasattr(agent, "act_deterministic"):
                a = agent.act_deterministic(obs)
            elif hasattr(agent, "act"):
                # avoid exploration noise at eval
                a = agent.act(obs, deterministic=True)
            else:
                a = env.action_space.sample()
            obs2, r, term, trunc, info = env.step(a)
            total += float(r)
        return total / max(n, 1)
    except Exception as e:
        raise RuntimeError(f"Evaluation failed: {e}")


class AgentBase:
    def __init__(self, cfg, obs_dim, act_dim, device="cpu"):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = torch.device(device)
        self.norm = RunningNorm(obs_dim) if cfg.get("obs_norm", True) else None
        self.buffer = ReplayBuffer(cfg["buffer"], obs_dim, act_dim)
        self.total_steps = 0

    def preprocess(self, obs):
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if self.norm is not None:
            self.norm.update(obs)
            obs = self.norm.normalize(obs)
        return obs

    def store(self, obs, act, rew, obs2, done):
        o = self.preprocess(obs)
        o2 = self.preprocess(obs2)
        a = np.asarray(act, dtype=np.float32).reshape(-1)
        self.buffer.store(o, a, float(rew), o2, float(done))


def train_from_cfg(cfg_path: str):
    cfg = load_yaml(cfg_path)
    agent_name = cfg["agent"]
    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

    bw = BanditWrapper(train_channels=cfg["train_channels"], seed=seed)
    obs_dim, act_dim, N = bw.obs_dim, bw.act_dim, bw.N

    from src.agents.ddpg import DDPG
    from src.agents.td3 import TD3
    from src.agents.sac import SAC
    from src.agents.dqn_codebook import DQNCodebook

    builders = dict(ddpg=DDPG, td3=TD3, sac=SAC, dqn=DQNCodebook)
    if agent_name not in builders:
        raise ValueError(f"Unknown agent {agent_name}")
    agent = builders[agent_name](cfg, obs_dim, act_dim, N, bw.env, device=cfg.get("device", "cpu"))

    env_steps = int(cfg["env_steps"])
    start_steps = int(cfg.get("start_steps", 2000))
    eval_every = int(cfg.get("eval_every", 2000))

    os.makedirs("ckpt", exist_ok=True)
    os.makedirs(os.path.join("results", cfg_name(cfg)), exist_ok=True)

    obs, _ = bw.env.reset(seed=seed)
    hist = []
    t0 = time.time()

    for t in range(env_steps):
        if t < start_steps:
            a = bw.env.action_space.sample()
        else:
            a = agent.act(obs, deterministic=False)
        obs2, r, term, trunc, info = bw.env.step(a)
        agent.store(obs, a, r, obs2, float(term))
        if t >= start_steps:
            agent.update()
        if t % eval_every == 0:
            print(f">>> STARTING EVALUATION AT STEP {t} (this may take a minute on CPU) <<<", flush=True)
            ev = evaluate_on_frozen_test_set(agent, test_channels=cfg["test_channels"])
            hist.append(dict(step=t, eval_rate=float(ev), wall=time.time() - t0))
            
            # FIX: Save metrics incrementally so we can watch it in real-time!
            out_dir = os.path.join("results", cfg_name(cfg), f"seed{seed}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "metrics.json")
            write_json(out_path, hist)
            
            # frozen checkpoint format
            ckpt_path = os.path.join("ckpt", f"{cfg_name(cfg)}_seed{seed}.pt")
            torch.save(agent.state_dict(), ckpt_path)
            print(f"Saved metrics to {out_path}", flush=True)
        obs = obs2
        if term or trunc:
            obs, _ = bw.env.reset()

    out_path = os.path.join("results", cfg_name(cfg), f"seed{seed}", "metrics.json")
    write_json(out_path, hist)
    return hist


def cfg_name(cfg):
    name = cfg["agent"]
    if name == "dqn" and "bits" in cfg:
        return f"dqn_b{int(cfg['bits'])}"
    return name
