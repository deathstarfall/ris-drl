# RIS-DRL 2.0 - RIS Sum-Rate Study (frozen-protocol)

**One-line summary:** deep unfolding closes **97.41 %** of the Riemannian AO
sum-rate reference at a **~51x same-machine latency advantage** (DD-048
protocol: 0.424 s per AO solve vs 8.26 ms per unfolding decision, serialized
single-solve wall clocks on one idle desktop CPU; DD-041 audited
train/validation protocol; DD-043 corrected ZF pairing), while a
multi-seed DRL
campaign establishes a **null result** (DRL ~ random) -- shipped as paper tag
`v1.0.1-paper` (quality pass DD-037, v1.0.2).

- Paper: `paper/` (journal-style, 6 pages, compiles with pdflatex)
- Frozen test set: fixed before any evaluation; one frozen 500-channel
  tensor `data/test_channels.npz` covers every solver family.
- Decision log: `docs/design_decisions.md` (**DD-001 - DD-043**; every
  experimental convention recorded)

## Results snapshot (all from committed artifacts; +/- = SD)

| Method | Test rate (bps/Hz) | Notes |
|---|---|---|
| Riemannian AO (reference) | **64.193 +/- 0.582 (SD, n=500)** | median 0.424 s/channel under the DD-048 same-machine idle protocol (contended-run 1.352 s retired); strong local solution, not an upper bound (45.8 % of channels stop at the 200-iter cap) |
| Deep unfolding (L=6, lr 0.03) | **62.532 +/- 0.853 (SD)** | 97.41 % of AO reference; 8.257 ms median (p95 10.08, DD-048 same-machine; trained == untrained 8.35 ms control); **~51x speed-up = 32x fewer iterations x 1.6x cheaper per operation** |
| Deep unfolding, 5-seed protocol breadth | 62.526 +/- 0.041 | protocol-breadth number, not the headline |
| Best-of-200 random | 57.145 +/- 0.764 (SD) | exploitable headroom exists |
| Random phase | **53.178** (frozen baseline, n=500) | |
| DRL (6 archs x 5 seeds x 200k steps) | ~ 53.18 | **NULL**: deltas -0.10...+0.04, all p >= 0.37 |

Headline ratios: AO/random = **1.207**, PG/random = **1.200** (frozen-baseline
protocol, DD-036; gate2's internal draw random 53.172 and G2's re-baselined
ratio criterion >= 1.15 are disclosed; gate2's seed-0 internal ratios remain
artifact-informational, never published).

## Repo layout
```
paper/            LaTeX source (main.tex + sections/); figures in paper/figs
scripts/          experiment runners + figure/table generators
src/              Riemannian AO + unfolding (ao_riemannian.py, unfolding.py)
tests/            pytest suite (20 tests: gradient/shape/latency invariants)
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
pip install numpy scipy matplotlib pyyaml gymnasium pytest

#0 quick headline reproduction from committed artifacts (numpy only, no torch
#  needed) - prints every published number and checks it against its JSON:
python scripts/reproduce_headline.py

#1 data (frozen protocol: regenerates the committed npz bit-exactly at the
#   committed seeds; run in a scratch clone to verify, never overwrite results/)
python scripts/make_datasets.py
#2 AO reference + gate 2 (REQUIRED flags are --channels and --out)
python scripts/run_ao_reference.py --channels data/test_channels.npz --out results/ao
python scripts/run_gate2.py
#3 robustness trio + generalization (unfolding row needs the deployed ckpt)
python scripts/run_quantization.py --ckpt results/unfold/unfold_L6_rate.pt
python scripts/run_csi_sweep.py --ckpt results/unfold/unfold_L6_rate.pt
python scripts/run_generalization.py --ckpt results/unfold/unfold_L6_rate.pt --with-pg
#4 papers macros + figures from artifacts (honesty rule: missing artifact -> __TBD__ + exit 1)
python scripts/make_balraj_macros.py
python scripts/make_all_figures.py

#5 PDF
cd paper
pdflatex main.tex; bibtex main; pdflatex main.tex; pdflatex main.tex
python paper/crossref_check.py    # run from repo root; ALL ENTRIES VERIFIED (19/19 incl. the three revision surveys)
```

## Submission gates (kept for future releases)
1. Stub grep `TBD|XX\.X|\{X\.X\}` over `paper/sections/*.tex` -> must be silent
2. `main.log`: zero `Warning: (Reference|Citation)` and zero `!` errors
3. `pytest tests/ -q` green; `crossref_check.py` all-verified (12 entries)
4. Moji-bake sweep over `paper/**/*.tex` -> silent
5. Every number traceable to a `results/**` artifact (DD-018 rule)

## Tags & releases
- **`v1.0.2-paper`** - final submission candidate (round-3 auditor fixes:
  true-ZF unit test DD-045, latency-protocol clarification, G2-relaxation
  disclosure). `v1.0.1-paper` is henceforth FROZEN and will not move again.
- `v1.0.1-paper` - DD-043 re-certification + DD-044a/b full-CSI close-out
  (frozen at `d1168cf`), history of movement disclosed below.
  zero TBD/stubs, zero undefined refs; DD-036 (close-out) + DD-037 (quality
  pass) recorded. Re-pointed again on the DD-044/043 re-certification round
  (corrected ZF pairing; corrected AO/unfolding/DRL artifacts + G2 re-baseline;
  sec5c disclosure). Re-pointed again on the DD-044 round-2 review round
  (matched-discipline baselines: untrained L=6 PG 59.390, closed-form
  full-phase policy 56.944, per-user SNR artifact; docx round-2 text fixes;
  no headline number changed). Re-pointed finally on the DD-044b close-out
  (full-CSI campaign integrated, 30 ckpts LFS; null made unconditional;
  paper re-headlined; pytest 19/19, pdflatex clean). Last pre-fix paper
  commit: `fe1813c` (superseded).
- `v1.0-paper` - pre-release, superseded

## Artifacts <-> paper map
- `results/ao/gate2_verdict.txt` -> AO block + G3 latency row (DD-037;
  the committed per-channel AO timing there was recorded under 4-worker CPU
  contention - superseded for latency claims by DD-048, kept for rates)
- `results/ao_rerun_serial/` -> DD-048 idle single-worker full gate re-run
  (500 ch): per-channel AO median 0.376 s / p95 0.644 s, verdict PASS with
  p95 line; aggregates identical to the canonical set
- `results/unfold/latency_common_protocol.json` -> DD-048 SAME-MACHINE
  protocol (one process, one idle CPU, AO rates determinism-checked 0.0 vs
  frozen): AO 0.424 s median, unfolding trained 8.257 ms / untrained 8.347 ms
  control, ratio 51.4x = 32 iterations-ratio x 1.6 per-op ratio; the
  cross-machine pair (AO 1352 ms contended, unfolding 1295 ms deployment
  machine) and its 1044x quotient are retired from headline use
- `results/ao/ao_test_channels.(json|npz)` -> AO mean/sd, per-channel (AO wins
  100.0 % of channels vs frozen random)
- `results/ao/ao_test_channels_history.npz` -> AO convergence figure
- `results/unfold/unfold_L6_rate/metrics.json` -> deployed-protocol unfolding
  test rate + per-channel latency (1.295 median / 1.298 mean / 1.312 p95)
- `results/unfold/*/metrics.json` (seed1-4, lr-grid, circular/mse) -> 5-seed
  aggregate, lr grid, loss-ablation table/figure
- `results/unfold/latency_float32.json` -> isolated f64/f32 inference bench
  (retired from the paper latency claim in DD-037; kept as environment note)
- `results/bits_sweep.json`, `results/unfold/csi_sweep.json` -> quantization /
  CSI tables + figures
- `results/ao/generalization.json`, `pg_test_channels.json` -> generalization
  figure + AO-vs-PG figure
- `results/unfold/unfold_untrained_pg6.json` -> DD-044 matched baseline:
  untrained L=6 forward pass (59.390; six steps +6.21 over random, training +3.14)
- `results/baselines/closed_form_full_csi.json` -> DD-044 observation-contract
  scope: closed-form full-phase policy 56.944 +/- 1.884 (+3.77 over random)
- `results/ao/snr_summary.json` -> per-user SNR under ZF (39.9 dB random /
  48.3 dB AO)
- `paper/figs/unified_framework.png`, `paper/figs/csi_robustness.png` ->
  Word-report Figures 1-2 (scripts/make_report_figs.py)

## Pre-submission (promoted from v1.1 by DD-044a, reviewer round 2)
- **Full-CSI DRL retrain campaign — COMPLETE (DD-044b)**: 30/30 runs with
  `--obs-mode full`, seed-0 re-scores independently reproduced at
  |Δ| = 0.0000 cross-machine (`results/full/verification_summary.json`).
  Best family DQN-4bit 53.204 ± 0.037 = +0.03 over random floor (t ≈ 0.3).
  **The certified null is now unconditional**: direct model-free RL fails
  even with complete phase information. Certified partial campaign untouched.

## Deferred to v1.1 (documented, DD-036/037)
- AO per-channel wall p95 (needs a new gate2 artifact)
- PG generalization column (`run_generalization.py` naming the pg field)
- sec5b/sec5_2 table-merge decision
- supervisor/affiliation details finalized by the corresponding author at journal submission

Maintainers: Ayansh (integration), Balraj (AO/unfolding lanes), Shreyansh
(DRL + diagnosis note).

