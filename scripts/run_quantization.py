#!/usr/bin/env python3
"""
scripts/run_quantization.py — Balraj Singh (24BCE0531) — runbook §4.2 deliverable
=================================================================================

Quantization robustness: "R vs bits figure" data.

Quantizes each method's phases to a B-bit uniform codebook, theta_q =
(round(theta/Delta) mod 2^B) * Delta with Delta = 2*pi/2^B, and evaluates the
sum rate on the SAME frozen test channels (paired discipline, §1.5).

Methods: AO (from run_gate2/run_ao_reference thetas), unfolding (optional ckpt),
random (seeded draws — CSI-free control).

Output: results/ao/quantization.json  (figure itself = Ayansh's make_all_figures.py)

    python scripts/run_quantization.py \
        --channels data/test_channels.npz \
        --ao-thetas results/ao/ao_test_channels_thetas.npy \
        --ckpt results/unfold/unfold_L6_rate.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ao_riemannian import (  # noqa: E402
    CPX, REAL, TWO_PI, load_channel_npz, sum_rate_numpy,
)
from unfolding import UnrolledPG, channels_to_tensors  # noqa: E402


def quantize(theta: np.ndarray, bits: int) -> np.ndarray:
    """Uniform B-bit codebook per element: Delta = 2pi/2^B, nearest codeword."""
    delta = TWO_PI / (2 ** bits)
    return (np.round(theta / delta) % (2 ** bits)) * delta


def rates_for(thetas: np.ndarray, chs) -> np.ndarray:
    return np.array([sum_rate_numpy(thetas[i], chs[i]) for i in range(len(chs))])


def main() -> int:
    ap = argparse.ArgumentParser(description="B-bit quantization sweep (Balraj)")
    ap.add_argument("--channels", default="data/test_channels.npz")
    ap.add_argument("--ao-thetas", default="results/ao/ao_test_channels_thetas.npy",
                    help="AO phases from run_gate2.py / run_ao_reference.py")
    ap.add_argument("--ckpt", default=None,
                    help="optional unfolding ckpt (results/unfold/unfold_L6_rate.pt)")
    ap.add_argument("--out", default="results/bits_sweep.json")
    ap.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4], help="frozen set {2,3,4}")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    chs = load_channel_npz(args.channels)
    n = len(chs)
    N = np.asarray(chs[0]["h_r"]).shape[1]
    bits_list = sorted(set(args.bits))
    print(f"Quantization sweep: {n} channels, N={N}, bits={bits_list}")

    methods = {}

    if Path(args.ao_thetas).exists():
        ao_theta = np.load(args.ao_thetas)
        if ao_theta.shape[0] != n:
            print(f"WARNING: ao_thetas rows ({ao_theta.shape[0]}) != channels ({n}) — "
                  f"evaluating on the first {min(n, ao_theta.shape[0])} channels (paired).")
            n = min(n, ao_theta.shape[0])
            chs = chs[:n]
        methods["ao"] = ao_theta[:n]
    else:
        print(f"WARNING: {args.ao_thetas} missing — run run_gate2.py first; skipping AO.")

    if args.ckpt and Path(args.ckpt).exists():
        T = channels_to_tensors(chs)
        model = UnrolledPG(L=6, theta_init=0.0)
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True))
        model.eval()
        thetas = []
        with torch.enable_grad():
            for s in range(0, n, 128):
                idx = torch.arange(s, min(s + 128, n))
                chb = {k: (v[idx] if torch.is_tensor(v) and v.shape[0] == n else v) for k, v in T.items()}
                th = model(chb, create_graph=False)
                thetas.append(th.detach().numpy() % TWO_PI)
        methods["unfolding"] = np.concatenate(thetas, axis=0)
    elif args.ckpt:
        print(f"WARNING: {args.ckpt} missing — train first (train_unfolding.py); skipping unfolding.")

    rng = np.random.default_rng(args.seed)
    methods["random"] = rng.uniform(0.0, TWO_PI, (n, N))

    result = {"channels": args.channels, "n": n, "N": N, "seed": args.seed,
              "bits": bits_list, "methods": {}}
    header = f"{'method':<12}" + "".join(f"{'B=' + str(b):>14}" for b in bits_list) + f"{'inf (float)':>14}"
    print(header); print("-" * len(header))
    for name, thetas in methods.items():
        row = {"inf": {"mean": None, "sd": None}, "per_bits": {}}
        r_inf = rates_for(thetas, chs)
        row["inf"] = {"mean": float(r_inf.mean()), "sd": float(r_inf.std(ddof=1))}
        line = f"{name:<12}" + f"{r_inf.mean():>11.3f}  " + " " * 3
        cells = []
        for b in bits_list:
            r_q = rates_for(quantize(thetas, b), chs)
            row["per_bits"][str(b)] = {"mean": float(r_q.mean()), "sd": float(r_q.std(ddof=1))}
            cells.append(f"{r_q.mean():>11.3f}  ")
        print(f"{name:<12}" + "".join(f"{c:>14}" for c in [f"{r_inf.mean():.3f}"] + cells))
        result["methods"][name] = row

    dd009 = {
        "bits": bits_list,
        "mean_rate_by_method": {
            name: [result["methods"][name]["per_bits"][str(b)]["mean"] for b in bits_list]
            for name in result["methods"]
        },
        "detail": result,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dd009, indent=2))
    print(f"\nSaved {out}  (DD-009: bits + mean_rate_by_method; rich data under 'detail')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
