#!/usr/bin/env python3
"""
Fix 3 (DD-064): equal-compute rebaseline for the L=6 learned-step-size
truncated-gradient-ascent network (formerly headline "deep unfolding").

Produces results/fix3/rebaseline.json with:
  - AO@6   : ao_riemannian(iters=6, tol=0)     on the 500 frozen test channels
  - PG@6   : pg_sanity(iters=6, tol=0)         on the same (Armijo step)
  - PGc@6  : fixed-step PG at 6 iterations, step alpha grid-searched ON THE
             300-channel validation hold-out (channels 1200-1499 of the train
             split, i.e. the same hold-out the unfolding LR used), then
             evaluated ONCE on the 500 test channels
  - reference rows: AO@full (64.193), unfolding (62.532), PG@200 (63.79)

Everything deterministic, paired discipline, artifact-only numbers.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ao_riemannian import ao_riemannian, pg_sanity, load_channel_npz, \
    sum_rate_torch, _to_ch_torch, TWO_PI, REAL  # noqa: E402

def rate_of(ch, theta):
    with torch.no_grad():
        return float(sum_rate_torch(torch.as_tensor(theta, dtype=REAL),
                                    _to_ch_torch(ch)))

def pg_fixed_step(ch, alpha, iters=6):
    ch_t = _to_ch_torch(ch)
    N = ch_t["h_r"].shape[1]
    theta = torch.zeros(N, dtype=REAL, requires_grad=True)
    with torch.no_grad():
        f = float(sum_rate_torch(theta, ch_t))
    for _ in range(iters):
        g = torch.autograd.grad(sum_rate_torch(theta, ch_t), theta)[0]
        theta = (theta.detach() + alpha * g)
        theta = (theta - TWO_PI * torch.floor(theta / TWO_PI)).detach() \
            .requires_grad_(True)
    return theta

def evaluate(fn, channels):
    out = []
    t0 = time.perf_counter()
    for i, ch in enumerate(channels):
        out.append(fn(ch))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(channels)} ({time.perf_counter()-t0:.0f}s)",
                  flush=True)
    return out

def main():
    rng = np.random.default_rng(0)
    test = load_channel_npz(str(ROOT / "data/test_channels.npz"))
    train = load_channel_npz(str(ROOT / "data/train_channels.npz"))
    val = train[-300:]                       # same hold-out as unfolding LR
    print(f"test={len(test)} val={len(val)}", flush=True)

    res = {}

    t0 = time.perf_counter()
    ao6 = evaluate(lambda ch: rate_of(ch, ao_riemannian(
        ch, iters=6, tol=0.0, record_history=False)["theta"]), test)
    print(f"AO@6 done in {time.perf_counter()-t0:.0f}s", flush=True)

    pg6 = evaluate(lambda ch: float(pg_sanity(ch, iters=6, tol=0.0)["rate"]),
                   test)

    grid = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    val_scores = {}
    for alpha in grid:
        rs = evaluate(lambda ch: rate_of(ch, pg_fixed_step(ch, alpha)),
                      val)
        val_scores[alpha] = float(np.mean(rs))
        print(f"grid alpha={alpha}: val mean {val_scores[alpha]:.3f}",
              flush=True)
    best_alpha = max(val_scores, key=val_scores.get)
    pgc6 = evaluate(lambda ch: rate_of(ch, pg_fixed_step(ch, best_alpha)),
                    test)

    res.update({
        "protocol": "equal-compute rebaseline: all methods at 6 iterations; "
                    "constant-step PG grid selected on the 300-channel "
                    "validation hold-out, evaluated once on test",
        "n_test": len(test), "n_val": len(val),
        "ao_full_mean": 64.193, "unfolding_mean": 62.532, "pg_200_mean": 63.790,
        "ao6_mean": float(np.mean(ao6)), "ao6_sd": float(np.std(ao6, ddof=1)),
        "pg6_mean": float(np.mean(pg6)), "pg6_sd": float(np.std(pg6, ddof=1)),
        "pgc_best_alpha": best_alpha,
        "pgc6_mean": float(np.mean(pgc6)), "pgc6_sd": float(np.std(pgc6, ddof=1)),
        "ao6_per_channel": ao6, "pg6_per_channel": pg6,
        "pgc6_per_channel": pgc6, "pgc_val_scores": val_scores,
    })
    out = ROOT / "results/fix3/rebaseline.json"
    out.parent.mkdir(exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items()
                      if not k.endswith("per_channel")}, indent=1))
    print("->", out)

if __name__ == "__main__":
    main()
