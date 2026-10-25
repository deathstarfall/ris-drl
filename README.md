# RIS-DRL 2.0 — Certified-Baseline RIS Sum-Rate Study

**One-line summary:** deep-unfolding closes 94.7 % of the Riemannian AO sum-rate bound at
70.6× lower wall-time, while a certified multi-seed DRL campaign establishes a **NULL
result** (DRL ≈ random) — shipped as paper tag `v1.0.1-paper`.

- Paper: `paper/` (IEEE OJ-COMS-style, 6 pages, compiles with pdflatex)
- Certified test set: frozen 500-channels `data/test_channels.npz` (seed 2000 generation; all evals on seed-0 protocol)
- Decision log: `docs/design_decisions.md` (**DD-001 – DD-036**, every experimental convention recorded)

## Results snapshot (all from committed artifacts)

| Method | Test rate (bps/Hz) | Notes |
|---|---|---|
| Riemannian AO (bound) | **63.040 ± 0.644** | median 0.475 s/channel |
| Deep unfolding (L=6) | **59.875 ± 1.26** | 94.7 % of AO; 6.732 ms median (p95 8.006); **70.6× speed-up** |
| Best-of-200 random | 51.837 ± 0.998 | exploitable headroom exists |
| Random phase | **43.797** (frozen baseline, n=500) | |
| DRL (6 archs × 5 seeds × 200k steps) | ≈ 43.80 | **certified NULL**; Δ −0.07…−0.33, all p ≥ 0.25 |

Headline numbers: AO/random = **1.439**, PG/random = **1.428** (frozen-baseline protocol,
DD-036; gate2's seed-0 internal ratios 1.445/1.433 remain artifact-informational).
Robustness: AO/unfolding degrade gracefully at ε = 0.1 phrase-CSI error at 2/3/4-bit
quantization costs. Generalization (unseen user zone, 200 gen-split channels):
AO 56.56 / unfolding 53.42 / random 37.11.

## Repo layout
```
paper/            LaTeX source (main.tex + sections/); figures in paper/figs
scripts/          experiment runners + figure/table generators (make_balraj_macros v3,
                  make_all_figures, run_gate2, run_generalization, ...)
src/              Riemannian AO + unfolding (ao_riemannian.py, unfolding.py)
tests/            pytest suite (19 tests, gradient/shape/latency invariants)
results/          ALL committed raw JSON/NPZ artifacts the paper numbers derive from
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
# AO bound + gate 2
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
python crossref_check.py    # ALL ENTRIES VERIFIED
```

## Submission gates (defined, kept for future releases)
1. `Select-String -Path paper\sections\*.tex -Pattern "TBD|XX\.X|\{X\.X\}"` → silent (stub grep, DD-036 class)
2. `main.log` has zero `Warning: (Reference|Citation)`
3. `pytest tests/ -q` green; `crossref_check.py` all-verified
4. Every number traceable to a `results/**` artifact (DD-018 rule)

## Tags & releases
- **`v1.0.1-paper`** — submission candidate (final): zero TBD/stubs, zero undefined refs, DD-001–036
- `v1.0-paper` — pre-release, superseded (frozen before macro close-out)

## Artifacts ↔ paper map (high level)
- `results/ao/gate2_verdict.txt` → AO block + G3 latency row (dd-029: AO 475 ms vs unfolding 6.732 ms = 70.6×)
- `results/ao/ao_test_channels.(json|npz)` → AO mean±sd, per-channel (wins 100.0% vs frozen random)
- `results/unfold/*/metrics.json` → unfolding protocol mean + loss ablation (rate/circular/MSE)
- `results/unfold/latency_float32.json` → f64/f32 median + p95 latency, drift 0.0006 bps/Hz
- `results/bits_sweep.json`, `results/unfold/csi_sweep.json` → quantization / CSI tables + figures
- `results/ao/generalization.json`, `pg_test_channels.json` → gen present + ao-pg-gap figure

## Deferred to v1.1 (explicitly documented, DD-036)
- AO per-channel wall p95 (needs new gate2 artifact)
- PG generalization column (`run_generalization.py` naming the pg field)
- DRL trained-codebook columns once Shreyansh's evals land
- sec5b/sec5_2 table-merge decision

Maintainers: Ayansh (integration), Balraj (AO/unfolding lanes), Shreyansh (DRL + §6 diagnosis).
