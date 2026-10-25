import os
import json
from pathlib import Path


def _read_cfg(name):
    import yaml
    p = Path("configs") / f"{name}.yaml"
    with open(p,encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate():
    agents = ["sac", "td3", "ddpg"]
    bits = [2, 3, 4]
    seeds = [0, 1, 2, 3, 4]
    lines = []
    lines.append("#!/usr/bin/env bash")
    lines.append("set -e")
    lines.append("mkdir -p ckpt results logs")
    lines.append("")
    # SAC/TD3/DDPG
    for agent in agents:
        for s in seeds:
            lines.append(f"python -m src.train --agent {agent} --seed {s}")
    # DQN codebook sweep B in {2,3,4}
    for b in bits:
        for s in seeds:
            lines.append(f"python -m src.train --agent dqn --bits {b} --seed {s}")
    lines.append("")
    lines.append("echo 'campaign complete'")
    os.makedirs("scripts", exist_ok=True)
    with open("scripts/train_all.sh", "w",encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod("scripts/train_all.sh", 0o755)
    print("wrote scripts/train_all.sh")


if __name__ == "__main__":
    generate()
