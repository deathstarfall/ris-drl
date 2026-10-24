# DESIGN_DECISIONS_balraj.md — paste-ready entries for docs/design_decisions.md
# (runbook §9: every design change gets ≥ 3 lines here, the same day it is made)

## B1 — The torch sum-rate twin lives in src/ao_riemannian.py (for now)
- Runbook §6.6/§6.7 skeletons reference `sum_rate_torch` but the canonical home
  (Ayansh's precoder.py torch twin) did not exist when my branch had to go green.
- Decision: my module implements the frozen §6.2 formula EXACTLY (both torch and
  numpy twins) and tests pin torch==numpy to 1e-10; my files import nothing from
  src/precoder.py, so PR2 never blocks on PR1.
- At integration: either my implementation moves into precoder.py (≈30 lines) or
  both stay with the twin test as the drift tripwire — reviewer's (Shreyansh's) call.

## B2 — §6.2 gain-line pairing fix
- The runbook skeleton's `np.sum(Heff * W.conj(), axis=1)` pairs row k of Heff
  with ROW k of W, but user k's ZF beam is COLUMN k of W (Heff @ W = I); for
  K≠M the expression is also a shape error ((K,M)*(M,K)).
- Decision: pairing via `einsum('km,mk->k', Heff, W.conj())` — the only reading
  consistent with the ZF construction.
- Effect: rate values are correct for K≠M (K=4, M=8); any code copied verbatim
  from the skeleton would have crashed or silently mis-scored.

## B3 — Missing power key in the §6.1 channel dict
- §6.1's gen_channel returns no "P", §6.2's sum_rate reads ch["P"].
- Decision: `ch.get("P", 1.0)` — 1.0 W ≡ 30 dBm ≡ env.yaml p_tx_dbm; explicit
  ch["P"] always wins. Flagged so Ayansh's channel.py can adopt the same default.

## B4 — float64/complex128 everywhere in the bound and unfolding
- Finite-difference gradient tests need ~1e-6 relative agreement, unreachable in
  float32 at rate magnitudes ~O(10–50) bps/Hz.
- Decision: AO, unfolding, and all my scripts run complex128/float64; float32 is
  permitted only for inference-latency experiments and must be labeled as such.

## B5 — CG sign convention: ascent on f, not the runbook's minimization form
- §4.2 writes `d ← −rgrad + β·P_T(d)` (minimization form); we maximize the rate,
  so the steepest-ascent tangent direction is +rgrad.
- An early draft kept the minus sign: Armijo shrank γ indefinitely on a descent
  direction and the search stalled — exactly the §8 "AO ≈ random" ghost.
- tests/test_ao_grad.py (the day-1 FD deliverable) catches this class of bug in
  seconds; docs/ao_math.md §A6 documents the algebraic equivalence.

## B6 — CG restarts and honest convergence reporting
- FR-CG restarts: every N iterations, on non-ascent direction, and on Armijo
  failure (one steepest-ascent retry before declaring a numerical stall).
- run_gate2.py reports tol-hit % AND numerical-stall % and the rgrad norm at
  termination; the G2 verdict uses only the frozen criteria (ratio ≥ 1.3,
  iters ≤ 200) so the gate cannot be gamed by redefining "converged".

## B7 — Common initialization for AO and unfolding
- Both start at θ = 0 (z = 1): AO-vs-unfolding gaps are purely algorithmic.
- The random-phase baseline covers the "random init" view; best-of-200 covers
  the "no-learning practical bar".

## B8 — Unfolding losses: rate main, circular + MSE kept as ablations (fix #5)
- Main loss = −mean sum rate end-to-end (second-order autograd through the L
  internal rate gradients, batched).
- circular = 1 − cos(Δθ) (branch-cut-free); MSE-on-wrapped-angles = the old bug,
  kept ONLY to quantify its deficit; smoke already shows rate > circular > MSE.

## B9 — CSI-error and quantization protocols frozen before the sweep
- CSI: ĥ = h + ε·rms(h)·CN(0,1) per tensor, ε ∈ {0,.01,.02,.05,.1}, seeded so
  all methods see identical error realizations; decide on ĥ, score on true h;
  random = flat CSI-free control by construction.
- Quantization: uniform per-element codebook Δ = 2π/2^B, nearest codeword,
  B ∈ {2,3,4}, evaluated on the frozen test set for AO/unfolding/random.

## B10 — Scripts beyond the frozen scripts/ layout
- Added: run_ao_reference.py, train_unfolding.py, run_quantization.py,
  run_csi_sweep.py, run_generalization.py (my §4.2 robustness deliverables need
  entry points; the frozen list only had run_gate{1,2,3}.py / train_all.sh /
  make_all_figures.py).
- Scope guard: my scripts emit JSON/npy artifacts only; make_all_figures.py
  (Ayansh) remains the single figure/table source; nothing writes into data/.

## B11 — G2 ratio compression at high SNR (practitioner's note)
- At very high operating SNR the AO/random RATIO compresses toward 1 even for a
  perfect optimizer (coherent gain ≈ log2 N bps/Hz absolute; ratio → 1 as
  SNR → ∞), so a ratio < 1.3 with green FD tests and a healthy absolute margin
  indicts the link budget/geometry, not the optimizer (docs/ao_math.md §A9).
- Consequence: run_gate2.py also prints per-channel win % and absolute margins;
  the §8 playbook ordering (FD test FIRST, geometry meeting second) is encoded
  in the gate output itself.

## B12 — No conftest.py / src/__init__.py from my side
- We flagged these as merge-collision candidates in the team chat; my tests
  self-configure sys.path and my modules use dual-mode imports
  (package-relative with flat fallback), so PR2 adds zero shared-file conflicts.
