#!/usr/bin/env python3
"""scripts/eval_unfolding.py — Balraj — writes results/unfold/unfold_rates.json (DD-009)."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ao_riemannian import load_channel_npz, sum_rate_torch
from unfolding import UnrolledPG, channels_to_tensors, batch_slice


def main() -> int:
    ap = argparse.ArgumentParser(description="DD-009 unfolding eval (Balraj)")
    ap.add_argument("--ckpt", default="results/unfold/unfold_L6_rate.pt")
    ap.add_argument("--channels", default="data/test_channels.npz")
    ap.add_argument("--out", default="results/unfold/unfold_rates.json")
    ap.add_argument("--channel-seed", type=int, default=0)
    args = ap.parse_args()

    chs = load_channel_npz(args.channels)
    n = len(chs)
    T = channels_to_tensors(chs)

    seed = args.channel_seed
    if seed == 0:
        z = np.load(args.channels, allow_pickle=True)
        for k in ("channel_seed", "gen_seed", "seed"):
            if k in z.files:
                seed = int(np.asarray(z[k]).reshape(-1)[0])
                break
    if seed == 0:
        print("ERROR: channel_seed not found — pass --channel-seed 2000.")
        return 2

    model = UnrolledPG(L=6, theta_init=0.0)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True))
    model.eval()

    with torch.enable_grad():                       # warmup
        model(batch_slice(T, torch.arange(min(8, n))), create_graph=False)

    rates, lat_ms = [], []
    for i in range(n):
        chb = batch_slice(T, torch.arange(i, i + 1))
        t0 = time.perf_counter()
        with torch.enable_grad():                   # the forward IS a PG optimizer
            theta = model(chb, create_graph=False)
        lat_ms.append((time.perf_counter() - t0) * 1e3)
        with torch.no_grad():
            rates.append(float(sum_rate_torch(theta, chb)))

    rates, lat_ms = np.array(rates), np.array(lat_ms)
    dd009 = {
        "rates": rates.tolist(),
        "latency_ms": lat_ms.tolist(),
        "channel_seed": int(seed),
        "detail": {"n": n, "ckpt": args.ckpt,
                   "rate_mean": float(rates.mean()), "rate_sd": float(rates.std(ddof=1)),
                   "latency_median_ms": float(np.median(lat_ms)),
                   "latency_p95_ms": float(np.percentile(lat_ms, 95)),
                   "dtype": str(T["h_r"].dtype)},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dd009, indent=2))
    print(f"Saved {out}  |  rate {rates.mean():.3f} ± {rates.std(ddof=1):.3f} bps/Hz  |  "
          f"median {np.median(lat_ms):.3f} ms  |  channel_seed {seed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())