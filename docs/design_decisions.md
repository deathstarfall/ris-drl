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

