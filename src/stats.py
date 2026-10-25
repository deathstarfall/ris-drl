"""Statistics: SD vs SEM are DIFFERENT THINGS (fix #3). Paired tests on the frozen test set."""
import numpy as np
from scipy.stats import wilcoxon

def describe(x, label):
    x = np.asarray(x, float); n = len(x)
    return dict(method=label, n=n, mean=float(np.mean(x)),
                sd=float(np.std(x, ddof=1)), sem=float(np.std(x, ddof=1) / np.sqrt(n)))

def paired_vs_random(rates, random_rates, label=""):
    diff = np.asarray(rates, float) - np.asarray(random_rates, float)
    if np.allclose(diff, 0.0, atol=1e-12):          # identical arrays → degenerate test
        return dict(method=label, n=len(diff), mean_gain=0.0, p_wilcoxon=1.0,
                    ci95=[0.0, 0.0], frac_channels_positive=0.0,
                    degenerate=True, note="method output identical to random on all channels")
    stat, p = wilcoxon(diff)
    rng = np.random.default_rng(0)
    boots = [np.mean(diff[rng.integers(0, len(diff), len(diff))]) for _ in range(10000)]
    return dict(method=label, n=len(diff), mean_gain=float(np.mean(diff)),
                p_wilcoxon=float(p),
                ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                frac_channels_positive=float(np.mean(diff > 0)))

def seed_band(metrics_list):
    """metrics_list: list of metric.json lists from Shreyansh (one per training seed).
       Returns manifest -> joins all eval points, mean ± SD over seeds."""
    by_step = {}
    for m in metrics_list:
        for pt in m:
            by_step.setdefault(pt["step"], []).append(pt["eval_rate"])
    return {s: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                "n_seeds": len(v)} for s, v in sorted(by_step.items())}