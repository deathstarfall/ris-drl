#!/usr/bin/env python3
"""fill_csi_macros.py -- wire the 4 CSI macros from frozen results/unfold/csi_sweep.json
into paper/sections/balraj_macros.tex (replaces the TBD stub lines in place).
Balraj Singh (24BCE0531). Honesty rule: exits 1 without writing if artifact is missing."""
import json, re, sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
SWEEP = ROOT / "results/unfold/csi_sweep.json"
MACROF = ROOT / "paper/sections/balraj_macros.tex"

try:
    d = json.loads(SWEEP.read_text(encoding="utf-8"))
except Exception as e:
    print(f"FATAL: cannot read {SWEEP}: {e}"); sys.exit(1)

mrm = d.get("mean_rate_by_method", {})
ao, ra, eps = mrm.get("ao"), mrm.get("random"), d.get("eps", [])
if not ao or not ra or len(ao) != len(eps) or len(ra) != len(eps):
    print("FATAL: csi_sweep.json missing expected fields"); sys.exit(1)

def idx(target):
    for i, e in enumerate(eps):
        if abs(e - target) < 1e-9: return i
    print(f"FATAL: eps={target} not in artifact"); sys.exit(1)

i0, i1 = idx(0.0), idx(0.1)
vals = {"CsiAoZero": f"{ao[i0]:.3f}", "CsiAoPone": f"{ao[i1]:.3f}",
        "CsiRandomZero": f"{ra[i0]:.3f}", "CsiRandomPone": f"{ra[i1]:.3f}"}

lines = MACROF.read_text(encoding="utf-8").splitlines()
n = 0
for i, line in enumerate(lines):
    m = re.match(r"\\newcommand\{\\(\w+)\}\{", line)
    if m and m.group(1) in vals:
        lines[i] = "\\newcommand{\\%s}{\\textbf{%s}}" % (m.group(1), vals[m.group(1)])
        n += 1
if n != 4:
    print(f"FATAL: only {n}/4 stubs found -- aborting, file NOT written"); sys.exit(1)
MACROF.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("OK: " + ", ".join(f"{k}={v}" for k, v in vals.items()))