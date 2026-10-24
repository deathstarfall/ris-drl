#!/usr/bin/env python3
"""
scripts/run_ao_reference.py — Balraj Singh (24BCE0531) — runbook §4.4 handoff
============================================================================

Generic AO pass over ANY frozen channel file (no gate verdict).  Two uses:

  1. D5 handoff "Balraj -> Shreyansh: AO per-channel reference rates":
       python scripts/run_ao_reference.py --channels data/test_channels.npz --out results/ao
  2. AO targets for the unfolding circular/MSE ablations (train split):
       python scripts/run_ao_reference.py --channels data/train_channels.npz --out results/ao/train

Outputs in --out:
    ao_<stem>.json / ao_<stem>_thetas.npy / ao_<stem>_history.npz
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ao_riemannian import load_channel_npz, run_ao_dataset  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="AO reference over a frozen channel file (Balraj)")
    ap.add_argument("--channels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--max-channels", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    ch_path = Path(args.channels)
    if not ch_path.exists():
        print(f"ERROR: {ch_path} not found (Ayansh owns data/ — handoff D1).")
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = ch_path.stem

    chs = load_channel_npz(ch_path)
    if args.max_channels > 0:
        chs = chs[: args.max_channels]
    print(f"AO reference: {len(chs)} channels from {ch_path} -> {out}")

    t0 = time.perf_counter()
    res = run_ao_dataset(chs, iters=args.iters, tol=args.tol, workers=args.workers)
    rates = np.array([r["rate"] for r in res])
    iters = np.array([r["iters"] for r in res])

    np.save(out / f"ao_{stem}_thetas.npy", np.stack([r["theta"] for r in res]))
    hlen = max(len(r["history"]) for r in res) + 1
    hist = np.full((len(res), hlen), np.nan)
    for i, r in enumerate(res):
        h = np.asarray(r["history"], dtype=float)
        hist[i, :len(h)] = h
        hist[i, len(h):] = h[-1]
    np.savez_compressed(out / f"ao_{stem}_history.npz", history=hist, iters=iters)

    (out / f"ao_{stem}.json").write_text(json.dumps({
        "stem": stem, "n": len(res), "N": int(res[0]["theta"].shape[0]),
        "config": {"iters": args.iters, "tol": args.tol},
        "ao": {"mean": float(rates.mean()), "sd": float(rates.std(ddof=1)),
               "median_iters": float(np.median(iters)), "max_iters": int(iters.max()),
               "median_time_s": float(np.median([r["time_s"] for r in res])),
               "per_channel": rates.tolist()},
        "wall_s": time.perf_counter() - t0,
    }, indent=2))
    print(f"  mean AO rate {rates.mean():.3f} ± {rates.std(ddof=1):.3f} bps/Hz "
          f"({time.perf_counter() - t0:.1f} s)  -> {out}/ao_{stem}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
