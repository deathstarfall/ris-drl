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
<<<<<<< HEAD
batch_slice's condition (v.shape[0] == len(idx)) never matched per-datasettensors (theta 1500 vs ref 128), so the rate loss silently ran full-batch,circular/mse crashed on the mismatch, and latency was measured as one500-channel forward pass. Fixed to slice by T["h_r"].shape[0] with av.dim() >= 1 guard; all three loss ablations retrained and the latency tablere-measured (true per-channel) before G3.
=======
batch_slice's condition (v.shape[0] == len(idx)) never matched per-datasettensors (theta 1500 vs ref 128), so the rate loss silently ran full-batch,circular/mse crashed on the mismatch, and latency was measured as one500-channel forward pass. Fixed to slice by T["h_r"].shape[0] with av.dim() >= 1 guard; all three loss ablations retrained and the latency tablere-measured (true per-channel) before G3.

DD-017 (ratified D2/D3): GATE G2 PASSED — AO/random = 1.445 over the FULL frozen
500-channel test set (100% per-channel wins; PG twin within 1%). Campaigns authorized
in all methods. (Logged initially by Balraj on feat/opt-bound — ratified here.)

DD-018 (already logged — keep): figure/table consume RAW eval JSONs; stats derived in-code.

DD-019 (Balraj B1–B12, renumbered): <one-line mapping table: B1→DD-019 ... B12→DD-030>.

DD-031 (D3 — LATENCY CRITERION VOTE): measured unfolding inference = 6.8 ms vs AO 0.388 s
=> 57.6x speedup (Balraj DD-029). Original acceptance text said "<1 ms". TEAM VOTE:
(a) implement forward-only inference path (no autograd at eval) as a fix task, OR
(b) revise acceptance to "<7 ms, >=57x vs AO; sub-ms as future work". DECISION: ____.

DD-032 (transiently — resolving): results/unfold/unfold_rates.json (DD-009) absent;
per-channel unfold rates must land there (rates[500] + latency_ms[500] + channel_seed
2000) before run_gate3 can produce its unfold row. Owner: Balraj.
>>>>>>> fbde7a1b7defa9ac4177a9952806c43cfb139815
DD-033 (GATE G3 — certified final assessment): Direct-DRL NULL, certified on the frozen 500-ch test set (seed 2000) for the validation-best seed of all six architectures (SAC, TD3, DDPG, codebook-DQN at 2/3/4 bits). No agent achieves a positive mean gain over frozen random; certified Δ range −0.07…−0.33 bps/Hz, all CIs covering zero, all p ≥ 0.25. Training-curve plateaus corroborate (last-10 validation eval 43.5–43.9 ≈ random 43.80) across 30 runs × 200k-step budget. Exploitable headroom demonstrably exists: best-of-200 random phases = 51.8; deep unfolding (trained, model-aware) = 59.9 (within 5.0% of Riemannian-AO bound 63.0; latency 9.7 ms; DD-031). Decision: paper reframed — headline = unfolding-vs-AO-bound; the DRL bandit framing is reported as a certified negative result (§ limitations + §6 diagnosis link, owner Shreyansh). Original G3 DRL-beats-random criterion: FAILED as framed; supersedes it and clears the paper to assemble. Budget footnote: pending Shreyansh's confirmation of 198k-vs-200k step accounting (fairness table, §5).
DD-034 (budget confirmation): Shreyansh confirms all 30 campaign runs stepped 0→199,999 inclusive — exactly 200,000 uniform updates (the 198k remainder was the last eval-logging checkpoint, not a short run). DD-033's "pending" budget footnote is resolved: fairness criterion satisfied as specified; no deviation.
DD-035 (paper freeze): All figures/tables derive from committed artifacts (DD-018); bibliography 8/8 crossref-verified; certified-null DRL framing in place (DD-033/034); MoM tout NULL-claims consistent; latency declaration 9.683 ms median, float64; remaining TBD macros resolved by [owner/date]. Submission candidate tagged at v1.0-paper.
DD-036 (v1.0.1 close-out - takeover, protocol + stub closure): (1) Random-baseline protocol unified: published ratios use the frozen DD-006 artifact (43.797, n=500): AO/random = 63.040/43.797 = 1.439, PG/random = 62.550/43.797 = 1.428; gate2_verdict's internal seed-0 200-draw random (43.640 -> 1.445 / 1.433) remains artifact-informational (duality), never published. (2) No-artifact macros removed pending v1.1: AoWallPNinetyFiveMs (gate2 records total+median wall only) and GenPg (PG-generalization column absent from generalization.json); generalization sentence drops PG until the column exists. (3) All value-placeholder stubs (X.X/XX.X pattern class) retired: AO block (63.040/0.644/43.797/100.0%/200 iters/0.475 s/44.4% tol-hit), unfold block (59.875 +/- 1.26; 94.7%; 6.732/8.006 ms median/p95; 70.6x), loss ablation (59.875/58.335/56.752), bits dB-equivalent gaps (5.734/1.402/0.354), gen gaps (6.479/6.453 bps/Hz). New gate rule adopted: providecommand{...}{X|XX|TBD} pattern grep joins TBD-grep. (4) Missing floats landed from artifacts: tab:quantization (bits_sweep.json), tab:ablation + fig:loss-ablation (metrics.json trio), fig:ao-pg-gap (pg_test_channels.json vs ao_test_channels.json, AO wins 100.0% of channels), fig:generalization (generalization.json); fig:csi-robustness ref retired to the existing fig:csi_sweep float (same csi_sweep.json artifact). (5) FDRel stopping tolerance corrected 1e-5 -> 1e-4 to match code truth (ao_riemannian tol=1e-4). Takeover note: landed on main after three failed 'push ready' intervals; Balraj's v4 branch merges later as v1.1 enrichment if it conflicts. [AMENDED by DD-037: gap 94.7% -> 94.98%, latency 6.732/8.006 ms -> 9.572/11.285 ms median/p95, speedup 70.6x -> 49.6x; claim language 'AO bound' superseded by 'AO reference'.]

DD-037 (v1.0.2 quality pass - claims language, latency unification, front matter, related work): (1) Latency unification: published decision latency = deployed-protocol per-channel artifact (results/unfold/unfold_L6_rate/metrics.json latency_ms, complex128, n=200 trials): median 9.572 ms / mean 9.719 / p95 11.285 (matches abstract ~9.7 ms and reviewer recomputation mean 9.68/median 9.86); the isolated latency_float32 bench (6.732 ms median) is retired from the paper as a different timing environment; G3 speed-up corrected 70.6x -> 49.6x (475/9.572, median/median). (2) Gap: UnfoldGapToAo = 59.8753/63.0395 = 94.98% of the reference; paper abstract's "5.0% short" retained; README's 94.7% (which belonged to the 5-seed aggregate mean 59.723/63.040) relabeled as the protocol-breadth number, never the headline. (3) "AO bound" overclaim corrected everywhere: 55.6% of channels stop at the 200-iteration cap in a numerical local optimum; the problem is non-convex; paper now says "AO reference" (strong reference solution, not a provable upper bound); label sec:ao_bound/sec:results_bound retained as internal anchor names only. (4) +/- labeling: every +/- now says SD (AO 63.04+/-0.64 SD n=500; best-of-200 51.84+/-1.083 SD vs 0.998 previously printed; README corrected); abstract carries explicit SD labels; 1.439/1.428 remain mean-based. (5) Abstract/inline jargon removed: "certified", "pre-registered", "(DD-018)" dropped from abstract and section headings; intro/conclusion softened to "frozen test set fixed before evaluation" and "written acceptance rules"; DD-log references remain only in the reproducibility subsection where the log itself is the subject. (6) Front matter: authors with affiliation block, corresponding author, IEEEkeywords; supervisor to be added by the corresponding author at journal submission (no fabricated names). (7) Related work +1 jsac: huang2020risdrl (JSAC 38(8):1839-1850, doi 10.1109/JSAC.2020.3000835), guo2020weighted (TWC 19(5):3064-3076, doi 10.1109/TWC.2020.2970061), zhang2021manifold (ICCC 2021, 10.1109/ICCC52777.2021.9580434), absil2008optimization (book, doi 10.1515/9781400830244); crossref_check 12/12 VERIFIED. (8) crossref check run locally: ALL ENTRIES VERIFIED.

DD-038 (v1.0.3 paper scrub - internal codes, latency context, venue neutrality): (1) All internal DD-0xx codes removed from the paper body (intro results/reproducibility lists, sec5 captions, DRL section, front-matter footnote); replaced by section references or "supplementary decision log"; macros-file comments are repo-internal and untouched. (2) Latency sentence added in the unfolding results: figures refer to the protocol's evaluation pass (batch size 1 per-channel decisions, torch complex128, desktop-class CPU); discloses the three measurements - dedicated 200-trial inference benchmark median 6.73 ms (latency_float32.json), reported training-time metrics 9.57 ms / p95 11.3 (metrics.json latency_ms, n=200), per-channel evaluation recording 9.86 ms / p95 13.4 (eval rates artifact) - the reported one is the conservative protocol measurement; abstract unchanged (~9.7 ms). (3) Running header \markboth "IEEE Open Journal of the Communications Society" removed - venue not yet decided (given the DRL null, any conditional-on-DRL-success target is moot); README: "IEEE OJ-COMS-style" -> "journal-style". (4) Gates re-run: PDF 6 pages silent (zero ! errors, zero undefined refs/citations), crossref 12/12, pytest 19/19, stub greps silent.

