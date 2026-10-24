"""The baselines the old paper lacked (fix #3). All evaluated on the FROZEN test set."""
import numpy as np
from .precoder import sum_rate

def _random_theta(rng, N):
    return rng.uniform(0.0, 2 * np.pi, size=N)

def run_baselines(chs, draw_seed=123, n_bo_draws=200, callback=None):
    """Returns dict method -> (n_channels,) rate array. callback(ch_i, method) for logging."""
    rng = np.random.default_rng(draw_seed)
    out = {"random": [], "zero": [], "no_ris": [], "best_of_200": []}
    for i, ch in enumerate(chs):
        N = ch["h_r"].shape[1]
        out["random"].append(sum_rate(_random_theta(rng, N), ch))
        out["zero"].append(sum_rate(np.zeros(N), ch))
        ch2 = dict(ch); ch2["h_r"] = np.zeros_like(ch["h_r"])          # RIS physically absent
        out["no_ris"].append(sum_rate(np.zeros(N), ch2))
        best = -np.inf
        for _ in range(n_bo_draws):
            best = max(best, sum_rate(_random_theta(rng, N), ch))
        out["best_of_200"].append(best)
        if callback: callback(i, f"{i+1}/{len(chs)}")
    return {k: np.asarray(v) for k, v in out.items()}