# BALRAJ_RUNBOOK.md — my (Balraj, 24BCE0531) lane of the RIS-DRL 2.0 runbook
## What is in this package, what to run, in what order, and how it lands as PR2

This folder contains EXACTLY my part of the team runbook (§4.2), nothing else.
Ayansh's and Shreyansh's files are deliberately absent (no `channel.py`, no
`precoder.py`, no agents, no `conftest.py`, no `src/__init__.py` — the last two
avoid the merge collisions we already flagged in the team chat).

```
src/ao_riemannian.py        G2 engine: torus Riemannian CG + sum_rate torch/numpy twins + PG twin
src/unfolding.py            UnrolledPG (L=6), rate/circular/MSE losses, trainer, latency meter
tests/test_ao_grad.py       DAY-1 DELIVERABLE: finite-difference gradient proof (7 tests)
tests/test_unfolding_grad.py FD + behaviour tests for the unfolding (6 tests)
configs/unfolding.yaml      frozen unfolding config (one tunable: lr)
scripts/run_gate2.py        G2 GATE — prints the frozen verdict, saves results/ao/*
scripts/run_ao_reference.py AO over any frozen npz (D5 handoff + unfolding targets)
scripts/train_unfolding.py  trains the unfolding (rate | circular | mse)
scripts/run_quantization.py R-vs-bits data (B ∈ {2,3,4})
scripts/run_csi_sweep.py    R-vs-ε data (ε ∈ {0,.01,.02,.05,.1}; decide on ĥ, score on h)
scripts/run_generalization.py unseen-zone rows (AO/unfolding/random; DRL merged at integration)
docs/ao_math.md             the math I defend in the viva (single source of truth)
paper/sections/*.tex        my paper sections §3, §4.2, §5.2 (macro-based, no hand-typed numbers)
DESIGN_DECISIONS_balraj.md  paste-ready entries for docs/design_decisions.md (§9 discipline)
```

---

## 0. One-time setup (WSL, mirrors runbook §5)

```bash
cd ~/ris-drl
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib pandas pytest gymnasium torch pyyaml
python -m pytest tests/ -q        # MUST be green BEFORE anything else
```

My tests are self-contained (tiny synthetic channels, fixed seeds) — they are
green on day 1 even if Ayansh's `data/` does not exist yet. No `conftest.py`
needed: tests add `src/` to `sys.path` themselves.

---

## 1. Branch discipline (as agreed in the team chat)

```bash
cd C:\Users\balra\ris-drl        # (WSL: /mnt/c/Users/balra/ris-drl)
git init                          # only if not already a repo
git remote add origin https://github.com/<owner>/<repo>.git
git fetch origin

git switch -c feat/opt-bound origin/main     # branch off what's on GitHub
# copy these files into the repo root preserving the layout, then:
git add src/ao_riemannian.py src/unfolding.py \
        tests/test_ao_grad.py tests/test_unfolding_grad.py \
        configs/unfolding.yaml docs/ao_math.md \
        scripts/run_gate2.py scripts/run_ao_reference.py scripts/train_unfolding.py \
        scripts/run_quantization.py scripts/run_csi_sweep.py scripts/run_generalization.py \
        paper/sections/sec3_system_model.tex paper/sections/sec4b_deep_unfolding.tex \
        paper/sections/sec5b_bound_and_unfolding.tex DESIGN_DECISIONS_balraj.md BALRAJ_RUNBOOK.md
git commit -m "Balraj: AO bound + unfolding + FD tests + G2 scripts (runbook D1)"
git push -u origin feat/opt-bound
# PR2 on GitHub: base=main, "Closes #N", reviewer Shreyansh, squash-merge by Ayansh
# (merge order per §7-D11: PR1 env+baselines first, then PR2)
```

If Ayansh's skeleton already contains files I also touched (expected: NONE —
I created no `conftest.py`, no `src/__init__.py`, no `precoder.py`):
`git merge origin/main` into the branch, keep/combine, log it (§9).

---

## 2. My execution order (mirrors runbook §7)

| When | Command | Done when |
|---|---|---|
| Day 1 (FIRST) | `python -m pytest tests/test_ao_grad.py -q` | 7/7 green — gradient proven before ANY AO run |
| Day 1 | `python -m pytest tests/ -q` | 13/13 green |
| D2 | `python scripts/run_gate2.py --channels data/test_channels.npz --out results/ao --max-channels 50` | overnight smoke, artifacts in results/ao/ |
| D3–D5 | `python scripts/run_gate2.py --channels data/test_channels.npz --out results/ao --workers 4` | **G2 verdict: PASS** printed + gate2_verdict.txt |
| D5 | `python scripts/run_ao_reference.py --channels data/train_channels.npz --out results/ao/train` | AO refs for ablations + per-channel rates → Shreyansh (handoff §4.4) |
| D6–D10 | `python scripts/train_unfolding.py --config configs/unfolding.yaml --loss rate` | metrics.json + ckpt |
| D6–D10 | same with `--loss circular` and `--loss mse` | ablation table data |
| D6–D10 | `python scripts/run_quantization.py --ckpt results/unfold/unfold_L6_rate.pt` | results/ao/quantization.json |
| D6–D10 | `python scripts/run_csi_sweep.py --ckpt results/unfold/unfold_L6_rate.pt` | results/ao/csi_sweep.json |
| D6–D10 | `python scripts/run_generalization.py --ckpt results/unfold/unfold_L6_rate.pt` | results/ao/generalization.json |
| D11 | PR2 review by Shreyansh; after merge `pytest -q && make_all_figures.py` | main green |

G2 FAIL path (runbook §8, pre-wired into run_gate2.py's output): run
`tests/test_ao_grad.py` FIRST — only if green, convene the geometry meeting
and log it. Note the SNR-compression effect (docs/ao_math.md §A9): a
failing ratio with a healthy absolute margin and green FD tests points at the
link budget, not the optimizer.

---

## 3. What I owe the team (handoff contract §4.4, my rows)

| From → To | Artifact | Where |
|---|---|---|
| Balraj → all | G2 bound, `results/ao/*.json` | run_gate2.py output |
| Balraj → Shreyansh | AO per-channel reference rates (D5) | `results/ao/ao_test_channels.json` per_channel list |
| Balraj → Ayansh | figure/table specs (JSON schema) | docs/ao_math.md §A8 + each script's JSON keys |
| Balraj → all | design_decisions.md entries, same day | DESIGN_DECISIONS_balraj.md (paste-ready) |

My scripts never write into `data/` (Ayansh is its only writer) and never
produce paper figures directly (Ayansh's `make_all_figures.py` is the single
figure source) — they emit JSON/npy/n pz artifacts only.

---

## 4. Honest-reporting notes baked into the deliverables

* **Latency**: the unfolding inference path is measured as-is (float64 CPU,
  ~7 ms per decision in the sandbox at N=64). The <1 ms G3 claim is a target;
  float32 / `torch.compile` / analytic gradients are the documented levers.
  The paper section claims only the RELATIVE speed-up vs AO (≈2 orders of
  magnitude), with the absolute table reported as measured.
* **Convergence**: run_gate2.py reports tol-hit % AND numerical-stall % plus
  the rgrad norm at termination — no convergence claim is hidden behind the cap.
* **Sandbox smoke numbers** (the ones quoted in docs/ao_math.md and this
  file) come from synthetic moderate-SNR channels and exist only to prove the
  machinery; they are NOT paper numbers. Paper numbers come exclusively from
  Ayansh's frozen npz on the frozen geometry.
