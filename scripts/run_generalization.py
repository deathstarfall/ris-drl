#!/usr/bin/env python3
"""
scripts/run_generalization.py — Balraj Singh (24BCE0531) — runbook §4.2 deliverable
===================================================================================

Generalization to the UNSEEN user zone (data/gen_channels.npz, seed 3000):
AO / unfolding / random evaluated on the unseen-zone channels and compared,
on identical footing, to the in-zone test channels.

Balraj's part of the unseen-zone figure: AO + unfolding (+ random control).
The DRL policies are merged into the same figure by Ayansh's evaluate.py /
make_all_figures.py at integration (hook documented below).

Output: results/ao/generalization.json

    python scripts/run_generalization.py \
        --gen data/gen_channels.npz --test data/test_channels.npz \
        --ckpt results/unfold/unfold_L6_rate.pt
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
    TWO_PI, load_channel_npz, random_rates, run_ao_dataset, sum_rate_numpy,
)
from unfolding import UnrolledPG, channels_to_tensors  # noqa: E402


def unfold_thetas(model, chs):
    T = channels_to_tensors(chs)
    n = len(chs)
    out = []
    with torch.enable_grad():
        for s in range(0, n, 128):
            idx = torch.arange(s, min(s + 128, n))
            chb = {k: (v[idx] if torch.is_tensor(v) and v.shape[0] == n else v) for k, v in T.items()}
            out.append(model(chb, create_graph=False).detach().numpy() % TWO_PI)
    return np.concatenate(out, axis=0)


def evaluate_zone(chs, iters, tol, n_random, seed, ckpt):
    """AO + unfolding + random on one channel zone.  The DRL policies join here
    at integration: Ayansh's evaluate.py runs each agent ckpt on the SAME npz —
    we deliberately do not touch Shreyansh's agent code (repo discipline §3)."""
    res = run_ao_dataset(chs, iters=iters, tol=tol, workers=1)
    ao = np.array([r["rate"] for r in res])
    rng = np.random.default_rng(seed)
    rand = np.array([random_rates(ch, rng, n_random).mean() for ch in chs])
    unf = None
    if ckpt and Path(ckpt).exists():
        model = UnrolledPG(L=6, theta_init=0.0)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model.eval()
        th = unfold_thetas(model, chs)
        unf = np.array([sum_rate_numpy(th[i], chs[i]) for i in range(len(chs))])
    return ao, unf, rand


def main() -> int:
    ap = argparse.ArgumentParser(description="Unseen-zone generalization (Balraj)")
    ap.add_argument("--gen", default="data/gen_channels.npz", help="unseen positions, seed 3000")
    ap.add_argument("--test", default="data/test_channels.npz", help="in-zone reference, seed 2000")
    ap.add_argument("--ckpt", default=None, help="optional unfolding ckpt")
    ap.add_argument("--out", default="results/ao/generalization.json")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-channels", type=int, default=0)
    args = ap.parse_args()

    for p in (args.gen, args.test):
        if not Path(p).exists():
            print(f"ERROR: {p} not found (Ayansh owns data/ — handoff D1).")
            return 2

    gen = load_channel_npz(args.gen)
    test = load_channel_npz(args.test)
    if args.max_channels > 0:
        gen, test = gen[: args.max_channels], test[: args.max_channels]
    print(f"Generalization: gen={len(gen)} unseen-zone chs vs test={len(test)} in-zone chs")

    out = {"gen": {"file": args.gen, "n": len(gen)}, "test": {"file": args.test, "n": len(test)},
           "methods": {}}
    header = f"{'method':<12}{'test zone':>16}{'unseen zone':>16}{'gap (bps/Hz)':>16}"
    print(header); print("-" * len(header))

    t0 = time.perf_counter()
    ao_t, unf_t, rand_t = evaluate_zone(test, args.iters, args.tol, args.n_random,
                                        args.seed, args.ckpt)
    ao_g, unf_g, rand_g = evaluate_zone(gen, args.iters, args.tol, args.n_random,
                                        args.seed, args.ckpt)

    rows = [("ao", ao_t, ao_g)]
    if unf_t is not None and unf_g is not None:
        rows.append(("unfolding", unf_t, unf_g))
    rows.append(("random", rand_t, rand_g))

    for name, r_t, r_g in rows:
        out["methods"][name] = {
            "test": {"mean": float(r_t.mean()), "sd": float(r_t.std(ddof=1))},
            "unseen": {"mean": float(r_g.mean()), "sd": float(r_g.std(ddof=1))},
            "gap": float(r_t.mean() - r_g.mean()),
        }
        print(f"{name:<12}{r_t.mean():>13.3f} ± {r_t.std(ddof=1):.2f}"
              f"{r_g.mean():>13.3f} ± {r_g.std(ddof=1):.2f}"
              f"{r_t.mean() - r_g.mean():>13.3f}")

    out["wall_s"] = time.perf_counter() - t0
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {outp}  (DRL rows merged at integration by evaluate.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
