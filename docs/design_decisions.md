\# Design decisions (append-only; the paper's claims must trace to entries here)



DD-001 (D0): Precoding = ZF + power normalization. Old env died because MRT with unit noise

&#x20;            and unit-variance channels made SINR interference-limited (phases ≈ irrelevant).

DD-002 (D0): Channel convention A\_eff = h\_d + (h\_r ∘ e^{jθ}) G^H. Same formula in

&#x20;            precoder.py (numpy), ao (Balraj, torch), unfolding (Balraj). Global conjugation

&#x20;            is equivalent to relabeling θ -> −θ; results unaffected.

DD-003 (D0): Codebook action design = per-element (N heads × 2^B outputs). Not 4-block.

DD-004 (D0): obs = concat(|h\_r|, ∠h\_r, |G|, |h\_d|) flat float32. Includes per-element

&#x20;            leverage — fixes "policies learned nothing the channel could tell them".

DD-005 (D0).npz schema (data/\*.npz), generated ONLY by scripts/make\_datasets.py:

&#x20;            arrays: u\_pos (n,3) · h\_d (n,K,M) complex · G (n,M,N) complex

&#x20;                    · h\_r (n,K,N) complex · sigma2 (n,) float · P (n,) float

DD-006 (D0): ONE frozen test set: data/test\_channels.npz (seed 2000, n=500). Every method

&#x20;            — baselines, AO, unfolding, all agents — evaluates on exactly this file.

&#x20;            Fixes old bug #2 (reset() inside optimizer → targets on wrong channel).

DD-007 (D0): Framing: T=1 contextual bandit. γ never appears anywhere in code or paper.

DD-008 (D0) CHECKPOINT / RESULTS CONTRACT (Shreyansh must code to it):

&#x20;            ckpt/{agent}\_seed{S}.pt                (state\_dict of the policy net)

&#x20;            results/{agent}/seed{S}/metrics.json   \[{step, eval\_rate, wall}, ...]

&#x20;            eval via ayansh's src/evaluate.py --method rl --agent sac --seed S

DD-009 (D0) AO / UNFOLDING RESULTS CONTRACT (Balraj must write to it):

&#x20;            results/ao/ao\_rates.json      {"rates":\[500 floats], "iters":\[500 ints],

&#x20;                                           "convergence":\[iter-series for fig], "channel\_seed":2000}

&#x20;            results/unfold/unfold\_rates.json {"rates":\[500], "latency\_ms":\[500],

&#x20;                                           "channel\_seed":2000}

&#x20;            results/unfold/csi\_sweep.json {"eps":\[0,...0.1],"mean\_rate\_by\_method":{...}}

&#x20;            results/bits\_sweep.json       {"bits":\[2,3,4],

&#x20;                                           "mean\_rate\_by\_method":{method:\[r2,r3,r4]}}

DD-010 (D2): G1 thresholds are stated constants in scripts/run\_gate1.py; changing them

&#x20;            requires a logged vote here (never change silently).

DD-zero (D1): Power-normalized ZF ⇒ Heff@W = c·I. Interference-free = off-diagonals zero, not identity. Corrected the over-strict test; per-user SNR = |c\_gain(k)|²/σ², and the whole optimization problem is now pure phase alignment.

DD-013 (D1): PR1 was committed to main directly by mistake (b86d366). Accepted as-is;

&#x20;            retroactive consumer review by Shreyansh requested as a PR-review comment.

&#x20;            Rule reminder: branch FIRST (`git switch -c feat/...`), commit second, PR third.

DD-014 (D3): (B1) sum_rate torch+numpy twins live in src/ao_riemannian.py (frozen §6.2
             formula), pinned to 1e-10 vs numpy in tests/test_ao_grad.py. No import of
             precoder.py so PR2 never blocked on PR1. At integration: move into
             precoder.py (≈30 lines) or keep both with the twin test as drift tripwire.

DD-015 (D3): (B2) §6.2 skeleton's gain pairing np.sum(Heff*W.conj(), axis=1) is wrong for
             K≠M: pairs user k with ROW k of W, but user k's ZF beam is COLUMN k (Heff@W=I);
             also a (K,M)*(M,K) shape error. Correct pairing: einsum('km,mk->k', Heff, W.conj()).

DD-016 (D3): (B3) §6.1 channel dict carried no "P" while §6.2 reads ch["P"]. My code uses
             ch.get("P", 1.0) (1 W ≡ 30 dBm). Superseded by DD-005: npz carries P (n,),
             loader passes it through per channel.

DD-017 (D3): (B4) float64/complex128 everywhere in AO + unfolding + scripts. FD gradient
             tests need ~1e-6 relative agreement, unreachable in float32 at rate O(10-50)
             bps/Hz. float32 only for inference-latency experiments, labeled as such.

DD-018 (D3): (B5) CG sign: we MAXIMIZE rate, so ascent direction = +rgrad; §4.2's
             minimization form (d ← -rgrad + β·P_T(d)) stalled Armijo exactly as §8's
             "AO ≈ random" ghost. Caught by day-1 FD tests; equivalence in ao_math.md §A6.

DD-019 (D3): (B6) FR-CG restarts every N iters, on non-ascent direction, and on Armijo
             failure (one steepest-ascent retry before declaring numerical stall).
             run_gate2.py reports tol-hit % AND stall % + rgrad norm at stop; the G2
             verdict uses only the frozen criteria so it cannot be gamed.

DD-020 (D3): (B7) AO and unfolding share init θ=0 (z=1): AO-vs-unfolding gap is purely
             algorithmic. random-phase covers "random init"; best-of-200 the practical bar.

DD-021 (D3): (B8) Unfolding losses: main = -mean sum rate end-to-end (second-order autograd
             through the L internal rate gradients); circular = 1-cos(Δθ), branch-cut-free;
             MSE-on-wrapped-angles = old bug #5, kept ONLY to quantify its deficit.

DD-022 (D3): (B9) CSI-error protocol: ĥ = h + ε·rms(h)·CN(0,1) per tensor,
             ε ∈ {0,.01,.02,.05,.1}, seeded so all methods see identical realizations;
             decide on ĥ, score on true h. Quantization: uniform per-element codebook
             Δ = 2π/2^B, nearest codeword, B ∈ {2,3,4}, on the frozen test set.

DD-023 (D3): (B10) Extra entry points beyond the frozen scripts/ layout: run_ao_reference.py,
             train_unfolding.py, run_quantization.py, run_csi_sweep.py, run_generalization.py.
             Scope guard: my scripts emit JSON/npy only; make_all_figures.py stays the single
             figure source; nothing writes into data/.

DD-024 (D3): (B11) At very high SNR the AO/random ratio compresses toward 1 even for a
             perfect optimizer (coherent gain ≈ log2 N absolute; ratio → 1 as SNR → ∞), so a
             failing ratio with green FD tests indicts the link budget/geometry, not the
             optimizer. The −25 dB directly-blocked geometry makes this less of a risk here.

DD-025 (D3): (B12) No conftest.py / src/__init__.py from my side — tests self-configure
             sys.path, modules use dual-mode imports; PR2 adds zero shared-file conflicts.

DD-026 — batch_slice indexed by batch length, not dataset size
batch_slice's condition (v.shape[0] == len(idx)) never matched per-datasettensors (theta 1500 vs ref 128), so the rate loss silently ran full-batch,circular/mse crashed on the mismatch, and latency was measured as one500-channel forward pass. Fixed to slice by T["h_r"].shape[0] with av.dim() >= 1 guard; all three loss ablations retrained and the latency tablere-measured (true per-channel) before G3.