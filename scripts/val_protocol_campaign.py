#!/usr/bin/env python3
"""DD-041 campaign: re-run the unfolding under the corrected train/val protocol.
Step 1: rate loss over the lr grid {0.003, 0.01, 0.03}; select by VALIDATION rate.
Step 2 (if winner is 0.01, matching the frozen ablation protocol): retrain
        seeds 1-4 at the winner lr, and the circular/mse ablations at lr 0.01.
Artifacts: results/unfold/unfold_L6_rate{,_lr0003,_lr003}/metrics.json (new protocol)
           + results/unfold/val_protocol_selection.json
"""
import json, subprocess, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GRID = [("0.003", "configs/unfolding_lr0003.yaml", "_lr0003"),
        ("0.01",  "configs/unfolding.yaml",       ""),
        ("0.03",  "configs/unfolding_lr003.yaml", "_lr003")]

results = {}
for lr, cfgp, suffix in GRID:
    print(f"=== rate loss, lr={lr} ===", flush=True)
    subprocess.run([sys.executable, "scripts/train_unfolding.py", "--config", cfgp,
                    "--loss", "rate", "--out-suffix", suffix], check=True)
    m = json.load(open(f"results/unfold/unfold_L6_rate{suffix}/metrics.json"))
    results[lr] = {"best_val_rate": m["best_val_rate"], "best_iter": m["best_iter"],
                   "final_test_rate": m["final_test_rate"], "latency": m["latency_ms"]}

winner = max(results, key=lambda lr: results[lr]["best_val_rate"])
sel = {"protocol": "DD-041 train/val", "grid": results, "winner_lr": winner,
       "selection_rule": "max best_val_rate over lr grid; test evaluated once per run"}
json.dump(sel, open("results/unfold/val_protocol_selection.json", "w"), indent=2)
print("WINNER lr =", winner, flush=True)

if winner == "0.01":
    for s in range(1, 5):
        print(f"=== 5-seed protocol breadth, seed {s}, lr=0.01 ===", flush=True)
        subprocess.run([sys.executable, "scripts/train_unfolding.py",
                        "--config", "configs/unfolding.yaml", "--loss", "rate",
                        "--seed", str(s), "--out-suffix", f"_valseed{s}"], check=True)
    for loss in ("circular", "mse"):
        print(f"=== ablation {loss}, lr=0.01, new protocol ===", flush=True)
        subprocess.run([sys.executable, "scripts/train_unfolding.py",
                        "--config", "configs/unfolding.yaml", "--loss", loss,
                        "--out-suffix", "_val"], check=True)
print("CAMPAIGN DONE", flush=True)

