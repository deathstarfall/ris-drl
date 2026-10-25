#!/usr/bin/env python3
"""inspect_tbd.py -- peek into candidate artifacts for the remaining TBD macros.
Balraj Singh (24BCE0531). Read-only: prints structure, writes nothing.
Honesty rule: we only wire macros whose numbers come from these frozen files."""
import json
from pathlib import Path

LIMIT = 12  # show lists/dicts up to this size in full, else length + first 3

def show(obj, indent="  "):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                print(f"{indent}{k}: {type(v).__name__}[len={len(v)}]")
                if len(v) <= LIMIT:
                    show(v, indent + "    ")
            else:
                print(f"{indent}{k} = {v!r}")
    elif isinstance(obj, list):
        if len(obj) <= LIMIT:
            for i, v in enumerate(obj):
                if isinstance(v, (dict, list)):
                    print(f"{indent}[{i}] {type(v).__name__}[len={len(v)}]")
                    show(v, indent + "    ")
                else:
                    print(f"{indent}[{i}] = {v!r}")
        else:
            print(f"{indent}first 3 = {obj[:3]!r}")
            print(f"{indent}... total {len(obj)} items")

def peek(path, why):
    print("=" * 64)
    print(f"{path}   <-- {why}")
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  FATAL: {e}")
        return
    if isinstance(d, dict):
        print(f"  top-level keys: {list(d.keys())}")
    show(d, "  ")

peek("results/ao/generalization.json",
     "GenAo GenPg GenUnfold GenRandom PgRandomRatio")
peek("results/bits_sweep.json",
     "QaoQuantGapBthree")
peek("results/unfold/unfold_rates.json",
     "DD-009 contract: rates[500] latency_ms[500] channel_seed=2000")
peek("results/ao/ao_rates.json",
     "AoWallPNinetyFiveMs? (expect rates only -> stays TBD)")