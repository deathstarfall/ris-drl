#!/usr/bin/env python3
"""fill_gen_macros.py -- wire generalization + PG-ratio macros from frozen artifacts
into paper/sections/balraj_macros.tex (replaces the TBD stub lines in place).
Balraj Singh (24BCE0531). Honesty rule: exits 1 without writing if any artifact
is missing, or if the PG file turns out to be from a different channel vintage."""
import json, re, sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
GEN = ROOT / "results/ao/generalization.json"
PG = ROOT / "results/ao/pg_test_channels.json"
MACROF = ROOT / "paper/sections/balraj_macros.tex"

def load(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"FATAL: cannot read {p}: {e}"); sys.exit(1)

gen, pg = load(GEN), load(PG)
m = gen.get("methods", {})
try:
    ao_ref = m["ao"]["test"]["mean"]          # canonical AO test mean (63.0395...)
    if abs(pg["ao_mean"] - ao_ref) > 0.01:
        print(f"FATAL: pg file ao_mean {pg['ao_mean']:.3f} != test AO mean "
              f"{ao_ref:.3f} -- different channel vintage, ratio not like-for-like")
        sys.exit(1)
    vals = {
        "GenAo": f"{m['ao']['unseen']['mean']:.3f}",
        "GenUnfold": f"{m['unfolding']['unseen']['mean']:.3f}",
        "GenRandom": f"{m['random']['unseen']['mean']:.3f}",
        "PgRandomRatio": f"{pg['pg_mean'] / m['random']['test']['mean']:.3f}",
    }
except KeyError as e:
    print(f"FATAL: artifact missing expected field: {e}"); sys.exit(1)

lines = MACROF.read_text(encoding="utf-8").splitlines()
n = 0
for i, line in enumerate(lines):
    mm = re.match(r"\\newcommand\{\\(\w+)\}\{", line)
    if mm and mm.group(1) in vals:
        lines[i] = "\\newcommand{\\%s}{\\textbf{%s}}" % (mm.group(1), vals[mm.group(1)])
        n += 1
if n != len(vals):
    print(f"FATAL: only {n}/{len(vals)} stubs found -- aborting, file NOT written")
    sys.exit(1)
MACROF.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"receipt: pg_mean={pg['pg_mean']:.3f} / random_test={ao_ref and m['random']['test']['mean']:.3f}")
print("OK: " + ", ".join(f"{k}={v}" for k, v in vals.items()))