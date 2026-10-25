import os
import json
import glob
import shutil

# Find all agent directories in results/ (ignoring the 'best' folder itself)
agent_dirs = [d for d in glob.glob("results/*") if os.path.isdir(d) and not d.endswith("best")]

for agent_dir in agent_dirs:
    agent_name = os.path.basename(agent_dir)
    best_seed = None
    best_rate = -float('inf')
    
    # Look at all seeds for this agent
    seed_dirs = glob.glob(os.path.join(agent_dir, "seed*"))
    for seed_dir in seed_dirs:
        metrics_path = os.path.join(seed_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Find the max eval_rate achieved in this run
            max_rate = max([entry.get("eval_rate", 0) for entry in data])
            if max_rate > best_rate:
                best_rate = max_rate
                best_seed = os.path.basename(seed_dir)
                
    if best_seed:
        # Create the 'best' directory
        best_dir = os.path.join(agent_dir, "best")
        os.makedirs(best_dir, exist_ok=True)
        
        # Copy the best seed's metrics.json to eval_rates.json
        src = os.path.join(agent_dir, best_seed, "metrics.json")
        dst = os.path.join(best_dir, "eval_rates.json")
        shutil.copy(src, dst)
        print(f"✅ {agent_name}: Best seed is {best_seed} (max rate: {best_rate:.2f}). Saved to {dst}")
