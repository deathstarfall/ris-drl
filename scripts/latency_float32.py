"""
scripts/latency_float32.py — Balraj Singh (24BCE0531) — G3 latency lever row
============================================================================

Train in float64 (FD-grade gradients), MEASURE the deployment float32 path:
same trained checkpoint, channels and step sizes cast to float32/complex64.

Reports — honestly — the float64 row, the float32 row, and the rate drift
between them, then writes results/unfold/latency_float32.json.

    python scripts/latency_float32.py --config configs/unfolding.yaml --ckpt results/unfold/unfold_L6_rate.pt
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ao_riemannian import load_channel_npz                      # noqa: E402
from unfolding import (UnrolledPG, channels_to_tensors, cast_tensors,   # noqa: E402
                       evaluate_on_channels, measure_decision_latency)


def main():
    ap = argparse.ArgumentParser(description="float32 inference-latency lever (G3 honest table)")
    ap.add_argument("--config", default="configs/unfolding.yaml")
    ap.add_argument("--ckpt", default=None,
                    help="default: <out.root>/<ckpt_stem>_L<L>_<train.loss>.pt")
    ap.add_argument("--trials", type=int, default=200, help="latency trials per dtype")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    L = int(cfg["method_L"]) if "method_L" in cfg else int(cfg.get("L", 6))
    root = pathlib.Path(cfg["out"]["root"])
    if args.ckpt is None:
        tag = f"{cfg['out']['ckpt_stem']}_L{L}_{cfg['train']['loss']}"
        args.ckpt = str(root / f"{tag}.pt")
    print(f"[latency32] ckpt: {args.ckpt}")

    test_chs = load_channel_npz(cfg["data"]["test_channels"])
    theta_init = float(cfg["train"]["theta_init"])
    wrap = bool(cfg["train"]["wrap_theta"])
    trials = int(args.trials)

    # ---- float64 row (the training dtype; must match metrics.json) ------
    model64 = UnrolledPG(L=L, theta_init=theta_init, wrap=wrap)
    model64.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    T64 = channels_to_tensors(test_chs)
    rate64 = evaluate_on_channels(model64, T64)
    lat64 = measure_decision_latency(model64, T64, n_trials=trials)

    # ---- float32 row (the documented deployment lever) ------------------
    T32 = cast_tensors(T64, torch.complex64)
    model32 = UnrolledPG(L=L, theta_init=theta_init, wrap=wrap, dtype=torch.float32)
    model32.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    rate32 = evaluate_on_channels(model32, T32)
    lat32 = measure_decision_latency(model32, T32, n_trials=trials)

    drift = abs(rate32 - rate64)
    gain = lat64["median_ms"] / lat32["median_ms"]

    out = {
        "ckpt": args.ckpt,
        "n_channels": len(test_chs),
        "n_trials": trials,
        "float64": {"test_rate": rate64, **lat64},
        "float32": {"test_rate": rate32, **lat32},
        "rate_drift_bps_hz": drift,
        "latency_gain_float32": gain,
    }
    out_path = root / "latency_float32.json"
    out_path.write_text(json.dumps(out, indent=2))

    print(f"[latency32] float64 : rate {rate64:8.3f} bps/Hz | "
          f"median {lat64['median_ms']:7.3f} ms | p95 {lat64['p95_ms']:7.3f} ms")
    print(f"[latency32] float32 : rate {rate32:8.3f} bps/Hz | "
          f"median {lat32['median_ms']:7.3f} ms | p95 {lat32['p95_ms']:7.3f} ms")
    print(f"[latency32] rate drift {drift:.4f} bps/Hz | float32 latency gain x{gain:.2f}")
    print(f"[latency32] wrote {out_path}")
    print("[latency32] G3 row: AO 388 ms (float64) vs unfolding "
          f"{lat64['median_ms']:.2f} ms (float64) / {lat32['median_ms']:.2f} ms (float32)")


if __name__ == "__main__":
    main()