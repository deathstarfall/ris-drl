import numpy as np


def extract_metrics_summary(results_root="results"):
    # Parse results/*/seed*/metrics.json (if exist) and produce a compact summary
    import os, json
    from pathlib import Path
    summary = {}
    root = Path(results_root)
    if not root.exists():
        return summary
    for agent_dir in root.iterdir():
        if not agent_dir.is_dir():
            continue
        summary[agent_dir.name] = {}
        for seed_dir in agent_dir.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
                continue
            mpath = seed_dir / "metrics.json"
            if mpath.exists():
                try:
                    with open(mpath) as f:
                        hist = json.load(f)
                    if hist:
                        # last eval and best
                        last = hist[-1]
                        best = max(hist, key=lambda x: x.get("eval_rate", -1))
                        summary[agent_dir.name][seed_dir.name] = dict(
                            last_eval=last.get("eval_rate"),
                            best_eval=best.get("eval_rate"),
                            steps=last.get("step"),
                            wall=last.get("wall"),
                        )
                except Exception as e:
                    summary[agent_dir.name][seed_dir.name] = dict(error=str(e))
    return summary
