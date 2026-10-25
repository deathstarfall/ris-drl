# RIS-DRL 2.0 - RIS Sum-Rate Study (frozen-protocol)

**One-line summary:** deep unfolding closes **94.98 %** of the Riemannian AO
sum-rate reference at **49.6x lower wall-time**, while a multi-seed DRL
campaign establishes a **null result** (DRL ~ random) -- shipped as paper tag
`v1.0.1-paper` (quality pass DD-037, v1.0.2).

- Paper: `paper/` (IEEE OJ-COMS-style, 6 pages, compiles with pdflatex)
- Frozen test set: fixed before any evaluation; one frozen 500-channel
  tensor `data/test_channels.npz` covers every solver family.
- Decision log: `docs/design_decisions.md` (**DD-001 - DD-037**; every
  experimental convention recorded)

## Results snapshot (all from committed artifacts; +/- = SD)

| Method | Test rate (bps/Hz) | Notes |
|---|---|---|
| Riemannian AO (reference) | **63.040 +/- 0.644 (SD, n=500)** | median 0.475 s/channel; strong local solution, not an upper bound (55.6 % of channels reach the 200-iter cap) |
| Deep unfolding (L=6) | **59.875 +/- 1.26 (SD)** | 94.98 % of AO reference; 9.572 ms median (mean 9.719, p95 11.285, n=200 trials); **49.6x speed-up** |
| Deep unfolding, 5-seed protocol breadth | 59.723 +/- 0.108 | protocol-breadth number, not the headline |
| Best-of-200 random | 51.835 +/- 1.083 (SD) | exploitable headroom exists |
| Random phase | **43.797** (frozen baseline, n=500) | |
| DRL (6 archs x 5 seeds x 200k steps) | ~ 43.80 | **NULL**: deltas -0.07...-0.33, all p >= 0.25 |

Headline ratios: AO/random = **1.439**, PG/random = **1.428** (frozen-baseline
protocol, DD-036; gate2's seed-0 internal ratios 1.445/1.433 remain
artifact-informational, never published).

## Repo layout
```
paper/            LaTeX source (main.tex + sections/); figures in paper/figs
scripts/          experiment runners + figure/table generators
src/              Riemannian AO + unfolding (ao_riemannian.py, unfolding.py)
tests/            pytest suite (19 tests: gradient/shape/latency invariants)
results/          ALL committed raw JSON/NPZ artifacts paper numbers derive from
docs/             design_decisions.md, sec6-diagnosis.md, math notes
data/             train/test/gen channel tensors (committed .npz)
ckpt/             checkpoints (gitignored)
```

## Reproduce
```powershell
git clone https://github.com/deathstarfall/ris-drl.git; cd ris-drl
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy matplotlib pyyaml pytest

# data
python scripts/make_datasets.py --channels 500 --N 64 --K 4 --M 8 --seed 2000
# AO reference + gate 2
python scripts/run_ao_reference.py --input data/test_channels.npz --out results/ao
python scripts/run_gate2.py
# robustness trio + generalization
python scripts/run_quantization.py; python scripts/run_csi_sweep.py
python scripts/run_generalization.py
# paper macros + figures from artifacts (honesty rule: missing artifact -> __TBD__ + exit 1)
python scripts/make_balraj_macros.py
python scripts/make_all_figures.py

# PDF
cd paper
pdflatex main.tex; bibtex main; pdflatex main.tex; pdflatex main.tex
python crossref_check.py    # ALL ENTRIES VERIFIED (12/12)
```

## Submission gates (kept for future releases)
1. Stub grep `TBD|XX\.X|\{X\.X\}` over `paper/sections/*.tex` -> must be silent
2. `main.log`: zero `Warning: (Reference|Citation)` and zero `!` errors
3. `pytest tests/ -q` green; `crossref_check.py` all-verified (12 entries)
4. Moji-bake sweep over `paper/**/*.tex` -> silent
5. Every number traceable to a `results/**` artifact (DD-018 rule)

## Tags & releases
- **`v1.0.1-paper`** - submission candidate: zero TBD/stubs, zero undefined
  refs; DD-036 (close-out) + DD-037 (quality pass) recorded
- `v1.0-paper` - pre-release, superseded

## Artifacts <-> paper map
- `results/ao/gate2_verdict.txt` -> AO block + G3 latency row (DD-037: AO
  475 ms vs unfolding 9.572 ms median = 49.6x)
- `results/ao/ao_test_channels.(json|npz)` -> AO mean/sd, per-channel (AO wins
  100.0 % of channels vs frozen random)
- `results/ao/ao_test_channels_history.npz` -> AO convergence figure
- `results/unfold/unfold_L6_rate/metrics.json` -> deployed-protocol unfolding
  test rate + per-channel latency (9.572 median / 9.719 mean / 11.285 p95)
- `results/unfold/*/metrics.json` (seed1-4, lr-grid, circular/mse) -> 5-seed
  aggregate, lr grid, loss-ablation table/figure
- `results/unfold/latency_float32.json` -> isolated f64/f32 inference bench
  (retired from the paper latency claim in DD-037; kept as environment note)
- `results/bits_sweep.json`, `results/unfold/csi_sweep.json` -> quantization /
  CSI tables + figures
- `results/ao/generalization.json`, `pg_test_channels.json` -> generalization
  figure + AO-vs-PG figure

## Deferred to v1.1 (documented, DD-036/037)
- AO per-channel wall p95 (needs a new gate2 artifact)
- PG generalization column (`run_generalization.py` naming the pg field)
- sec5b/sec5_2 table-merge decision
- supervisor/affiliation details finalized by the corresponding author at journal submission

Maintainers: Ayansh (integration), Balraj (AO/unfolding lanes), Shreyansh
(DRL + diagnosis note).
