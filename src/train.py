#!/usr/bin/env python
import argparse
import os
import copy

import yaml

from src.agents.train_runner import train_from_cfg, cfg_name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True, choices=["ddpg", "td3", "sac", "dqn"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bits", type=int, default=None, choices=[2, 3, 4])
    p.add_argument("--config", default=None)
    args = p.parse_args()

    if args.config is None:
        cfg_path = os.path.join("configs", f"{args.agent}.yaml")
    else:
        cfg_path = args.config
    with open(cfg_path, "r",encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg["agent"] = args.agent
    cfg["seed"] = int(args.seed)
    if args.agent == "dqn":
        if args.bits is not None:
            cfg["bits"] = int(args.bits)
        elif "bits" not in cfg:
            cfg["bits"] = 2
    # write a transient merged config if needed (keep original)
    out_cfg = os.path.join(".tmp_cfg.yaml")
    with open(out_cfg, "w",encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    train_from_cfg(out_cfg)


if __name__ == "__main__":
    main()
