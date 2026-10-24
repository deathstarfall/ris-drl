#!/usr/bin/env python3
"""
scripts/run_csi_sweep.py — Balraj Singh (24BCE0531) — runbook §4.2 deliverable
==============================================================================

CSI-error sweep: "R vs epsilon figure" data.

Protocol (frozen wording: "evaluate with h-hat, score on true h"):
  for each epsilon in {0, 0.01, 0.02, 0.05, 0.1}:
      h_hat = h + eps * rms(h) * CN(0,1)   per entry, per tensor (h_d, G, h_r),
                                             seeded — identical draw per method
      the optimizer sees the perturbed channel; the achieved phases are then
      SCORED on the TRUE channel with the §6.2 rate.
  random phases are CSI-free (same draws for every epsilon -> flat control line).

Output: results/ao/csi_sweep.json  (figure via Ayansh's make_all_figures.py)

    python scripts/run_csi_sweep.py --channels data/test_channels.npz \
        --ckpt results/unfold/unfold_L6_rate.pt --max-channels 100
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ao_riemannian import (  # noqa: E402
    TWO_PI, ao_riemannian, load_channel_npz, random_rates, sum_rate_numpy,
)
from unfolding import UnrolledPG, channels_to_tensors  # noqa: E402

EPS_DEFAULT = [0.0, 0.01, 0.02, 0.05, 0.1]     # frozen set (runbook §4.2)


def perturbed(ch: dict, eps: float, rng: np.random.Generator) -> dict:
    """Relative CSI error model: h_hat = h + eps*rms(h)*CN(0,1) on every tensor."""
    if eps == 0.0:
        return ch
    ch2 = dict(ch)
    for k in ("h_d", "G", "h_r"):
        h = np.asarray(ch[k])
        rms = float(np.sqrt((np.abs(h) ** 2).mean()))
        noise = (rng.normal(size=h.shape) + 1j * rng.normal(size=h.shape)) / np.sqrt(2)
        ch2[k] = h + eps * rms * noise
    return ch2


def main() -> int:
    ap = argparse.ArgumentParser(description="CSI-error sweep (Balraj)")
    ap.add_argument("--channels", default="data/test_channels.npz")
    ap.add_argument("--out", default="results/ao/csi_sweep.json")
    ap.add_argument("--eps", type=float, nargs="+", default=EPS_DEFAULT)
    ap.add_argument("--ckpt", default=None, help="optional unfolding ckpt for the sweep")
    ap.add_argument("--max-channels", type=int, default=0)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    chs_true = load_channel_npz(args.channels)
    if args.max_channels > 0:
        chs_true = chs_true[: args.max_channels]
    n = len(chs_true)
    N = np.asarray(chs_true[0]["h_r"]).shape[1]
    print(f"CSI sweep: {n} channels, eps={args.eps}, seed={args.seed}")

    have_ckpt = args.ckpt and Path(args.ckpt).exists()
    if args.ckpt and not have_ckpt:
        print(f"WARNING: ckpt {args.ckpt} missing — skipping unfolding (train first).")
    model = None
    T = None
    if have_ckpt:
        model = UnrolledPG(L=6, theta_init=0.0)
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True))
        model.eval()

    result = {"channels": args.channels, "n": n, "N": N, "seed": args.seed,
              "eps": args.eps,
              "protocol": "h_hat = h + eps*rms(h)*CN(0,1); decide on h_hat, score on true h",
              "methods": {}}
    for name in ["ao", "unfolding", "random"]:
        result["methods"][name] = {str(e): {"mean": None, "sd": None} for e in args.eps}

    header = f"{'eps':>8}" + "".join(f"{m:>14}" for m in ("AO", "unfolding", "random"))
    print(header); print("-" * len(header))
    for eps in args.eps:
        rng = np.random.default_rng(args.seed)          # same error realization for all methods
        chs_hat = [perturbed(ch, eps, rng) for ch in chs_true]

        # AO: decide on h_hat ...
        t0 = time.perf_counter()
        theta_hat = [ao_riemannian(ch, iters=args.iters, tol=args.tol)["theta"] for ch in chs_hat]
        # ... score on true h
        r_ao = np.array([sum_rate_numpy(theta_hat[i], chs_true[i]) for i in range(n)])

        r_unf = None
        if have_ckpt:
            # unfolding: trained on clean channels; at inference it sees h_hat gradients
            Th = channels_to_tensors(chs_hat)
            thetas = []
            with torch.enable_grad():
                for s in range(0, n, 128):
                    idx = torch.arange(s, min(s + 128, n))
                    chb = {k: (v[idx] if torch.is_tensor(v) and v.shape[0] == n else v)
                           for k, v in Th.items()}
                    thetas.append(model(chb, create_graph=False).detach().numpy() % TWO_PI)
            th = np.concatenate(thetas, axis=0)
            r_unf = np.array([sum_rate_numpy(th[i], chs_true[i]) for i in range(n)])

        # random: CSI-free control (draws do not depend on eps)
        rng_r = np.random.default_rng(args.seed)
        r_rand = np.array([random_rates(chs_true[i], rng_r, args.n_random).mean() for i in range(n)])

        result["methods"]["ao"][str(eps)] = {"mean": float(r_ao.mean()), "sd": float(r_ao.std(ddof=1))}
        if r_unf is not None:
            result["methods"]["unfolding"][str(eps)] = {"mean": float(r_unf.mean()),
                                                        "sd": float(r_unf.std(ddof=1))}
        result["methods"]["random"][str(eps)] = {"mean": float(r_rand.mean()),
                                                 "sd": float(r_rand.std(ddof=1))}
        cells = [f"{r_ao.mean():>11.3f}  ",
                 f"{r_unf.mean():>11.3f}  " if r_unf is not None else f"{'—':>13}",
                 f"{r_rand.mean():>11.3f}  "]
        print(f"{eps:>8.2f}" + "".join(f"{c:>14}" for c in cells)
              + f"   ({time.perf_counter() - t0:.1f} s)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
