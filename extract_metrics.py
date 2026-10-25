import os
import json
import glob

# Threshold for convergence (beating the random baseline)
BASELINE = 43.80

# Find all metrics.json files in the results directory
files = glob.glob("results/*/*/metrics.json")
print(f"Found {len(files)} metrics files to process.")

for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not data:
        continue
        
    last_entry = data[-1]
    final_step = last_entry.get("step", 198000)
    final_wall = last_entry.get("wall", 0.0)
    
    # 1. Convergence step: first step where eval_rate >= 43.80
    conv_step = None
    for entry in data:
        if entry.get("eval_rate", 0) >= BASELINE:
            conv_step = entry.get("step")
            break
            
    # Fallback: if it never crossed 43.80, find the step of max eval_rate
    if conv_step is None:
        max_rate = -float('inf')
        for entry in data:
            if entry.get("eval_rate", 0) > max_rate:
                max_rate = entry.get("eval_rate", 0)
                conv_step = entry.get("step")
                
    # 2. Total wall-clock (already have final_wall)
    total_wall = final_wall
    
    # 3. Per-decision latency (ms per step)
    latency_ms = (total_wall / final_step * 1000) if final_step > 0 else 0.0
    
    # Create a summary dictionary
    summary = {
        "step": "summary",
        "convergence_step": conv_step,
        "total_wall_clock_sec": round(total_wall, 2),
        "avg_latency_ms_per_step": round(latency_ms, 4)
    }
    
    # Append to the existing data list
    data.append(summary)
    
    # Write back to the file
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
print("✅ Done! All files enriched with side-effect metrics.")
