# DD-044b: independent settling check of the full-CSI campaign (Ayansh).
# Re-scores seed 0 of every agent family from ckpt/full_*_seed0.pt through
# build_policy_fn(obs_mode=full) and compares to Shreyansh's committed re-scores.
import io, json, os, sys
sys.path.insert(0, '.')
import numpy as np
from src.channel import load_dataset
from src.agents.common import build_policy_fn
from src.precoder import sum_rate

chs = load_dataset("data/test_channels.npz")
agents = ["sac", "td3", "ddpg", "dqn_b2", "dqn_b3", "dqn_b4"]
report = {}
for a in agents:
    policy = build_policy_fn(a, 0, chs[0]["h_r"].shape[1], obs_mode="full")
    rates = np.array([sum_rate(policy(ch), ch) for ch in chs])
    committed = json.load(open(f"results/full/{a}/seed0/eval_rates.json"))
    cm = float(np.mean(committed["rates"]))
    delta = abs(rates.mean() - cm)
    report[a] = {"my_mean": float(rates.mean()), "committed_mean": cm,
                 "abs_delta": float(delta), "ok": bool(delta < 0.05)}
    print(f"{a:8s} mine {rates.mean():.3f} vs committed {cm:.3f}  |d|={delta:.4f}  {'OK' if delta<0.05 else 'MISMATCH'}")

# aggregate per-family stats over all 5 committed seeds (means over the frozen set)
summary = {}
for a in agents:
    ms = [float(np.mean(json.load(open(f"results/full/{a}/seed{s}/eval_rates.json"))["rates"])) for s in range(5)]
    ms = np.array(ms)
    summary[a] = {"family_mean": float(ms.mean()), "seed_sd": float(ms.std(ddof=1)),
                  "best": float(ms.max()), "owners": "seed means " + " ".join(f"{m:.3f}" for m in ms)}
    print(f"{a:8s} family {ms.mean():.3f} +/- {ms.std(ddof=1):.3f}  best-seed {ms.max():.3f}")

# best full-CSI agent, paired per-channel delta vs the random floor artifact
best = max(summary, key=lambda a: summary[a]["family_mean"])
rnd = np.array(json.load(open("results/baselines/random/eval_rates.json"))["rates"])
fc = np.array(json.load(open(f"results/full/{best}/seed0/eval_rates.json"))["rates"])
# family-level: average per-channel rates across seeds for the paired test
fam_rates = np.mean([np.array(json.load(open(f"results/full/{best}/seed{s}/eval_rates.json"))["rates"]) for s in range(5)], axis=0)
d = fam_rates - rnd
n = len(d)
import math
t = d.mean() / (d.std(ddof=1) / math.sqrt(n)) if d.std(ddof=1) > 0 else float('nan')
print(f"best family {best}: paired delta vs random {d.mean():+.4f} bps/Hz, "
      f"t={t:.2f} (n={n}, seed-averaged rate surface, directional check only)")
json.dump({"settling_check": report, "family_summary": summary,
           "best_full_csi": best, "note": "DD-044b independent verification on local repo"},
          open("results/full/verification_summary.json", "w"), indent=1)
