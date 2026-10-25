# docs/ao_math.md — The Torus Optimization Math (Balraj Singh, 24BCE0531)

> Runbook §1.2 says "the system model is written once, here." This document is
> the single source of truth for the AO bound (G2), the deep unfolding (§4.2),
> and every equation Balraj defends in the viva. Section numbers below refer to
> the runbook unless prefixed "A" (this document's own numbering).

---

## A1. Scope and frozen references

| Item | Frozen value | Source |
|---|---|---|
| Objective | sum rate under ZF, `f(z) = Σ_k log2(1 + SNR_k)` | runbook §1.3 |
| Manifold | `M = {z ∈ C^N : |z_i| = 1}` (the torus), `Φ = diag(z)` | runbook §4.2 |
| Iteration cap | 200 | runbook §2 G2 |
| Gradient tolerance | `‖rgrad‖_F < 1e-4` | runbook §4.2 |
| CG | Fletcher–Reeves, tangent-transported, periodic resets | runbook §4.2 |
| Step control | Armijo backtracking on γ | runbook §4.2 |
| Initialization | `z = 1` (θ = 0) — same init for unfolding (fairness) | design decision B7 |
| Data | SAVED npz channels only; never regenerate | runbook §1.5 |
| dtype | float64 / complex128 | design decision B4 |

The G2 gate owner is Balraj; the acceptance sentence is
`AO/random ≥ 1.3 mean ratio` and `AO converges by iter ≤ 200`
(scripts/run_gate2.py prints exactly this verdict).

---

## A2. System model and the implementation-convention remark

Signal model (runbook §1.2):

```
y_k = h̃_k(θ)^H w_k + n_k,          k = 1..K
h̃_k(θ) = h_d,k + G^H Φ^H h_r,k      (K×M effective channel; the ONLY variable is θ)
Φ = diag(e^{jθ_1}, …, e^{jθ_N}),    θ ∈ [0, 2π)^N
```

Noise-limited design (fix #1): `σ² = N₀·BW·NF = −174 + 60 + 7 = −107 dBm`
(BW = 1 MHz, NF = 7 dB); `P = 30 dBm`. Full dB arithmetic lives in
Ayansh's `docs/link_budget.md`; this module reads `sigma2` from the channel dict.

ZF precoding (runbook §1.3, §6.2):

```
W = H̃^H (H̃ H̃^H)^{-1},  normalized so Σ‖W‖²_F = P
⇒ inter-user interference is EXACTLY zero, SINR_k = SNR_k = |h̃_k^H w_k|²/σ²
R(θ) = Σ_k log2(1 + SNR_k(θ))
```

**Notation & implementation convention (viva-critical).** The frozen code
(runbook §6.2) realizes the effective channel as
`Heff = h_d + (h_r ∘ e^{jθ}) · G^H`, i.e. the conjugations are absorbed
differently than in the §1.2 display formula. For i.i.d. Rayleigh entries
(`CN(0,1)`, circularly symmetric) the two are **statistically identical** —
conjugating and/or transposing an i.i.d. complex Gaussian matrix does not
change its law — and, decisively, ALL methods (Ayansh's env, my AO, my
unfolding, every baseline) share ONE implementation of the rate on the SAME
saved channel instances, so every paired comparison in the paper is exact.
`tests/test_ao_grad.py::test_torch_numpy_twin` pins the torch and numpy twins
to 1e-10 relative agreement; if Ayansh's `precoder.py` ever drifts, that test
fails before any experiment can be contaminated.

**Power-key note.** The §6.1 channel dict carries no `P` key while §6.2's rate
reads `ch["P"]`. Resolution (decision B3): `ch.get("P", 1.0)` — 1.0 W ≡ 30 dBm
≡ `env.yaml: p_tx_dbm`.

---

## A3. Why the phases matter at all (the G1 story, stated from the math)

Under ZF the inter-user interference is annihilated, so the rate is pure SNR
alignment. With per-element phases φ_n = e^{jθ_n}, user k's cascade is
`Σ_n h_r,k[n] φ_n conj(G[m,n])`: coherent alignment across n = 1..N elements
grows the cascade amplitude ∝ N, while a random φ leaves it ∝ √N. The power
ratio of coherent to incoherent alignment is therefore ~N (= 64 ⇒ ~18 dB), and
this gain is *exactly* what an optimizer can capture and a random draw cannot.
This is the mathematical content behind G1's "random ≫ no-RIS AND per-element
phase sensitivity ≠ 0" — before any learning is attempted.

---

## A4. The torus manifold and its tangent spaces

Manifold: `M = {z ∈ C^N : |z_i| = 1}` — the N-dimensional torus. The rate
depends on θ only through φ = e^{jθ}, so optimizing over z ∈ M is equivalent to
optimizing over θ, with better-behaved geometry (no branch cuts, compactness).

Tangent space at z (vectors that keep |z_i| = 1 to first order):

```
T_z M = { v ∈ C^N : Re(v_n · conj(z_n)) = 0,  n = 1..N }
```

Orthogonal (Euclidean) projection onto T_z M — the runbook §4.2 formula:

```
P_T(z)(v) = v − Re(v ∘ conj(z)) ∘ z
```

*Derivation (30-second viva version):* the normal direction at z_n is z_n
itself (radial). Decompose v_n = (v_n − Re(v_n conj(z_n)) z_n) + Re(v_n
conj(z_n)) z_n; the second term is radial, the first satisfies
Re((v_n − Re(v_n conj(z_n)) z_n) conj(z_n)) = Re(v_n conj(z_n)) −
Re(v_n conj(z_n))·|z_n|² = 0. ∎

Feasibility of a point is checked as `‖ |z| − 1 ‖_∞ < 1e-12`
(test_retraction_stays_on_torus).

---

## A5. Riemannian gradient via autograd

For a real-valued loss L and complex leaf z, PyTorch's complex-autograd
convention returns `g = ∂L/∂Re(z) + j·∂L/∂Im(z)` — exactly the Euclidean
(R^{2N}) gradient. The Riemannian gradient is its tangent projection:

```
rgrad = P_T(z)(g) = g − Re(g ∘ conj(z)) ∘ z
```

Two verifications, both in `tests/test_ao_grad.py` (the day-1 deliverable —
the §8 failure playbook starts here whenever AO misbehaves):

1. **FD check (off-torus allowed).** For random complex directions v:
   `(f(z+εv) − f(z−εv)) / 2ε = Re(Σ conj(g_n) v_n)` to ≤ 1e-5 relative
   (float64, ε = 1e-6). This catches ANY sign/convention error in the
   gradient chain — it is the test that caught the minimization-form sign bug
   described in A6.
2. **Tangency check.** `|Re(rgrad ∘ conj(z))|_max < 1e-8` on the torus.

---

## A6. Riemannian conjugate gradient on the torus (the G2 engine)

**Sign convention (viva note).** The runbook §4.2 table writes the CG recursion
in minimization form, `d ← −rgrad + β·P_T(d)`. We maximize the rate f directly,
so the steepest-ASCENT tangent direction is `+rgrad` — algebraically identical
to running the runbook's recursion on the objective −f. (History: an early
draft kept the minus sign; Armijo then shrank γ indefinitely on a descent
direction and the search stalled after a lucky first jump — precisely the
"AO ≈ random → check gradient" ghost of §8. The FD test catches it in
seconds; that is why it is the day-1 deliverable.)

Per iteration, at z_t with Riemannian gradient r_t:

```
(1) direction    d_t = r_t + β_t · P_T(z_t)(d_{t-1})     (ascent form)
    Fletcher–Reeves:  β_t = ‖r_t‖² / ‖r_{t-1}‖²,   ‖v‖² = Re(Σ conj(v_n) v_n)
    resets: d_t = r_t  whenever  (a) t ≡ 0 (mod N)   [periodic restart],
            (b) ⟨r_t, d_t⟩_R ≤ 0                      [non-ascent safeguard],
            (c) the Armijo search on d_{t-1}-based direction failed.
(2) retraction   R_z(γd) = (z + γd) ⊘ |z + γd|           (back onto the torus)
(3) Armijo (sufficient INCREASE, c1 = 1e-4, γ₀ = 1, ρ = 1/2, ≤ 30 halvings):
    accept γ when  f(R_z(γd)) ≥ f(z) + c1·γ·⟨r_t, d_t⟩_R
(4) stopping     ‖r_t‖_F < 1e-4  OR  t = 200
```

Termination quality is reported honestly by run_gate2.py: the fraction of
channels that stop via the tol rule vs. the fraction that stop by
"numerical stall" (Armijo cannot manufacture a sufficient increase — i.e. the
iterate sits at a local maximum to float64 precision), plus the median/max
`‖rgrad‖` at termination. Both terminations are legitimate optima; only the
tol-hit fraction satisfies the strict §4.2 wording, and the G2 verdict uses the
frozen criteria (ratio + cap) rather than either fraction.

**Complexity.** One f-evaluation costs `O(KNM + K³)` (build H̃: KNM; ZF via a
K×K inverse: K³; gains: KM). One CG iteration = 1 value+gradient (≈2×) plus a
few no-grad Armijo probes ⇒ ≤ 200 iterations ≈ 10³ f-evals per channel. At
N=64, K=4, M=8 this measured 0.21 s/channel (CPU, float64, 2 workers ⇒
~1.7 min for the full 500-channel gate in the sandbox smoke) — i.e. AO is the
"seconds-scale" upper bound against which the G3 latency claim is made.

**Init.** z = 1 (θ = 0). The unfolding uses the same init (decision B7), so
AO-vs-unfolding differences are purely algorithmic, never initialization luck.
The random-phase baseline covers the "random init" view.

**PG sanity twin.** `pg_sanity()` is plain backtracking gradient ascent in the
θ-domain (no CG memory, same init/stopping). Reviewers see `AO ≥ PG` with the
gap quantified (sandbox smoke: AO/PG ≈ 1.01 at N=64 — CG accelerates but the
problem is benign; the twin exists so that if AO ever UNDERPERFORMS PG, the CG
logic is the first suspect, per §8).

---

## A7. Deep unfolding (fix #5) — the model-driven star

Architecture (`src/unfolding.py::UnrolledPG`, runbook §6.7):

```
θ_{l+1} = wrap( θ_l + α_l · ∂R/∂θ |_{θ_l} ),     l = 0..L−1,  L = 6
α_l = exp(ρ_l) > 0        (ONE trainable log-step-size per layer; ρ_l ∈ R)
wrap(θ) = θ − 2π·floor(θ/2π)                      ("projected" — rate is 2π-periodic)
```

The network IS a differentiable optimizer: its forward pass executes L
gradient evaluations of the true rate through the batched ZF graph
(`create_graph=True` during training so the loss gradient flows THROUGH the
internal gradients — second-order autograd, batched over channels).

Training losses (the fix-#5 ablation triad):

| Loss | Formula | Status |
|---|---|---|
| rate | `−mean_k R(θ_L)` | **main method** — end-to-end on the sum rate |
| circular | `mean(1 − cos(Δθ))`, Δθ = wrap(θ_L − θ_AO) | ablation B (branch-cut-free) |
| MSE-on-angles | `mean((wrap θ_L − wrap θ_AO)²)` | ablation C — **the old bug**, discontinuous at ±π |

Protocol: 1500 train / 500 frozen test channels (Ayansh's npz), Adam on the L
= 6 step sizes only, one tunable hyperparameter (`lr`, grid {3e-2, 1e-2, 3e-3}
— the same one-grid rule as Shreyansh's SAC actor lr), seeds {0..4}.
circular/mse need AO reference thetas on the train split:
`run_ao_reference.py --channels data/train_channels.npz --out results/ao/train`.

Why unfolding can be fast (the G3 latency argument, stated honestly): inference
is EXACTLY L = 6 fixed gradient evaluations — no iterative-until-convergence
loop, no line search. Sandbox measurement: ~6.5–7.3 ms per decision on CPU in
float64; the honest G3 table reports this as-is, with float32 / torch.compile /
an analytic (autograd-free) gradient as the documented levers toward < 1 ms.
AO, by contrast, took 0.21 s/channel minimum in the same setting — two orders
of magnitude more compute per decision.

Smoke-run shape of the fix-#5 story (sandbox, 40 ch, 40 iters — NOT paper
numbers): rate-loss 53.21 > circular 53.13 > MSE-on-angles 52.22 bps/Hz. The
old bug is measurably the worst, as the runbook predicts.

---

## A8. Robustness and generalization protocols (Balraj's D6–D10 figures)

**Quantization (R vs bits).** Uniform per-element codebook: Δ = 2π/2^B,
`θ_q = (round(θ/Δ) mod 2^B)·Δ`, B ∈ {2,3,4}; evaluated on the frozen test set
for AO / unfolding / random. Output: `results/ao/quantization.json`.

**CSI error (R vs ε).** Relative error model,
`ĥ = h + ε·rms(h)·CN(0,1)` applied per entry to every tensor (h_d, G, h_r),
ε ∈ {0, 0.01, 0.02, 0.05, 0.1}, seeded so all methods see the SAME error
realization. Protocol: decide on ĥ, score on true h. Random phases are
CSI-free ⇒ a flat control line by construction (verified in the smoke run).

**Generalization (unseen zone).** All methods on `data/gen_channels.npz`
(unseen positions, seed 3000) vs the in-zone test set; Balraj's script emits
AO/unfolding/random rows; Shreyansh's agent checkpoints are merged into the
same figure by Ayansh's evaluate.py at integration (deliberate hook — repo
discipline §3).

---

## A9. Practitioner's note — reading the G2 ratio at high SNR

The AO/random RATIO compresses toward 1 as the operating SNR grows. Coherent
alignment buys ~log2(N) bps/Hz of ABSOLUTE gain, but the ratio is
`R_AO/R_rand ≈ (log2 SNR + log2 N)/log2 SNR → 1` for SNR ≫ N. Concretely: at
per-user SNR ~ 140 dB the sum-rate ratio of a PERFECT optimizer over random
phases can sit near 1.1–1.2 even though the absolute gain is large — and G2's
≥ 1.3 ratio would fail for geometry reasons, not optimizer reasons.

Consequences (this is exactly the "If G2 fails → link-budget/geometry meeting"
branch of runbook §2):

1. Run `tests/test_ao_grad.py` FIRST. If green, the optimizer is exonerated.
2. Check the ABSOLUTE margin (AO − random, bps/Hz) and the per-channel win
   fraction, both printed by run_gate2.py as informational rows.
3. If the margin is healthy but the ratio < 1.3, the operating point is too
   hot: weaken the direct link / deepen the user zone / revisit σ² — i.e. the
   30-minute geometry meeting, logged in design_decisions.md.

The sandbox smoke deliberately calibrates a moderate-SNR regime (random ≈
37 bps/Hz) so the PASS path of the gate machinery is exercisable before
Ayansh's real dataset lands. Smoke verdicts are NOT paper numbers.

---

## A10. Reproduction checklist (Balraj, in execution order)

```bash
python -m pytest tests/test_ao_grad.py -q          # day 1 — FD proof, BEFORE anything
python -m pytest tests/ -q                         # full suite green (13 tests)
# D2 overnight smoke (50 channels):
python scripts/run_gate2.py --channels data/test_channels.npz --out results/ao --max-channels 50
# D3–D5 full gate (500 channels):
python scripts/run_gate2.py --channels data/test_channels.npz --out results/ao --workers 4
# D5 handoff + unfolding targets:
python scripts/run_ao_reference.py --channels data/train_channels.npz --out results/ao/train
# D6–D10:
python scripts/train_unfolding.py --config configs/unfolding.yaml --loss rate
python scripts/train_unfolding.py --config configs/unfolding.yaml --loss circular
python scripts/train_unfolding.py --config configs/unfolding.yaml --loss mse
python scripts/run_quantization.py  --ckpt results/unfold/unfold_L6_rate.pt
python scripts/run_csi_sweep.py     --ckpt results/unfold/unfold_L6_rate.pt
python scripts/run_generalization.py --ckpt results/unfold/unfold_L6_rate.pt
```

Every artifact under `results/` is script-generated; hand-editing one is a
process crime (§8) and the offending PR gets rejected.
