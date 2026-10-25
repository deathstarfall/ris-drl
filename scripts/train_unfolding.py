#!/usr/bin/env python3
"""
scripts/train_unfolding.py — Balraj Singh (24BCE0531) — runbook §4.2 / §7 D6–D10
================================================================================

Training entry point for the deep unfolding (all three loss ablations):

    # main method (end-to-end sum-rate loss — fix #5):
    python scripts/train_unfolding.py --config configs/unfolding.yaml --loss rate
    # ablations (need AO reference thetas first — see run_ao_reference.py):
    python scripts/run_ao_reference.py --channels data/train_channels.npz --out results/ao/train
    python scripts/train_unfolding.py --config configs/unfolding.yaml --loss circular
    python scripts/train_unfolding.py --config configs/unfolding.yaml --loss mse   # the old bug

Outputs (script-generated only):
    results/unfold/unfold_L6_<loss>.pt            model checkpoint (state_dict)
    results/unfold/unfold_L6_<loss>/metrics.json  loss/test-rate history + latency table
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # for configs/ path resolution

from unfolding import train_from_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the deep unfolding (Balraj)")
    ap.add_argument("--config", default="configs/unfolding.yaml")
    ap.add_argument("--loss", choices=["rate", "circular", "mse"], default=None,
                    help="overrides configs/unfolding.yaml train.loss")
    ap.add_argument("--iters", type=int, default=None, help="override training iterations (smoke runs)")
    ap.add_argument("--seed", type=int, default=None, help="override train.seed (5-seed protocol)")
    ap.add_argument("--train", default=None, help="override data.train_channels (smoke/test runs)")
    ap.add_argument("--test", default=None, help="override data.test_channels (smoke/test runs)")
    ap.add_argument("--ao-refs", default=None, help="override data.ao_train_refs")
    ap.add_argument("--out-suffix", default="", help='e.g. "_seed1" for the 5-seed protocol')
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: {cfg_path} not found (run from the repo root).")
        return 2
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # repo-root-relative paths (scripts are always run from the repo root)
    root = Path(__file__).resolve().parents[1]
    for key in ("train_channels", "test_channels", "ao_train_refs"):
        p = Path(cfg["data"][key])
        if not p.is_absolute():
            cfg["data"][key] = str(root / p)
    out_root = Path(cfg["out"]["root"])
    cfg["out"]["root"] = str(root / out_root) if not out_root.is_absolute() else str(out_root)

    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    if args.train:
        cfg["data"]["train_channels"] = args.train
    if args.test:
        cfg["data"]["test_channels"] = args.test
    if args.ao_refs:
        cfg["data"]["ao_train_refs"] = args.ao_refs

    train_from_config(cfg, loss_override=args.loss,
                      max_iters_override=args.iters, out_suffix=args.out_suffix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
