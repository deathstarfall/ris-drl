#!/usr/bin/env python3
"""DD-041 artifact promotion: val-protocol campaign winner becomes the canonical artifact.

 BEFORE: canonical paths hold the OLD dev-protocol artifacts (test set evaluated during
 training / lr grid).  AFTER: canonical paths hold the DD-041 audited artifacts
 (train/val split; lr picked on validation; test evaluated once per run).
 Old artifacts are preserved with a _devproto suffix, nothing is deleted.
"""
import json, shutil
from pathlib import Path

R = Path("results/unfold")
W = Path("results/unfold/val_protocol_selection.json")
sel = json.load(open(W))
lr = sel["winner_lr"]
winner_dir = {"0.003": "unfold_L6_rate_lr0003", "0.01": "unfold_L6_rate", "0.03": "unfold_L6_rate_lr003"}[lr]

def backup(p: Path, suffix="_devproto"):
    if p.exists():
        b = p.joinpath("") .with_name(p.stem + suffix + p.suffix) if p.is_file() else Path(str(p) + suffix)
        shutil.move(str(p), str(b))
        return b
    return None

# 1. preserve the old dev-protocol trilogy + seed dirs + their checkpoints
for name in ("unfold_L6_rate", "unfold_L6_circular", "unfold_L6_mse",
             "unfold_L6_rate_seed1", "unfold_L6_rate_seed2", "unfold_L6_rate_seed3", "unfold_L6_rate_seed4"):
    backup(R / name)
    backup(R / (name + ".pt"))
for tag in ("unfold_L6_rate_lr0003", "unfold_L6_rate_lr003"):
    pass  # keep grid artifacts under their own names (they ARE val-protocol runs)

# 2. promote winner
shutil.move(str(R / winner_dir), str(R / "unfold_L6_rate"))
shutil.move(str(R / (winner_dir + ".pt")), str(R / "unfold_L6_rate.pt"))

# 3. promote new-protocol ablations + seed runs (produced by the second campaign wave)
for src, dst in (("unfold_L6_circular_val", "unfold_L6_circular"),
                 ("unfold_L6_mse_val", "unfold_L6_mse")):
    if (R / src).exists():
        shutil.move(str(R / src), str(R / dst))
        if (R / (src + ".pt")).exists():
            shutil.move(str(R / (src + ".pt")), str(R / (dst + ".pt")))
for s in range(1, 5):
    src, dst = f"unfold_L6_rate_valseed{s}", f"unfold_L6_rate_seed{s}"
    if (R / src).exists():
        shutil.move(str(R / src), str(R / dst))
        if (R / (src + ".pt")).exists():
            shutil.move(str(R / (src + ".pt")), str(R / (dst + ".pt")))

# 4. record what promotion did inside the selection artifact
sel["promoted_to_canonical"] = "results/unfold/unfold_L6_rate(.pt)"
sel["dev_protocol_backups"] = "*_devproto"
json.dump(sel, open(W, "w"), indent=2)
print("PROMOTED winner:", winner_dir, "-> unfold_L6_rate  (lr =", lr + ")")
