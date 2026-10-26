# §6 Diagnosis: Analysis of DRL Convergence at Random Baseline

Finalized after the DD-043 triage (corrected ZF pairing, deterministic evaluation,
all 30 checkpoints re-scored).  All numbers come from committed artifacts under
`results/` (DD-018).

Despite the theoretical headroom for RIS-assisted rate improvement, all six DRL
agents (SAC, TD3, DDPG, DQN-B2/3/4) sit statistically on the random-phase
baseline across all thirty checkpoint evaluations (deltas $-0.10$ to $+0.04$
bps/Hz, every 95 % bootstrap CI covers zero, all Wilcoxon $p \geq 0.37$;
canonical re-scored artifacts `results/{agent}/seed{n}/eval_rates.json`).

### 1. The rate objective is NOT flat — measured, not hypothesised
On the frozen 500-channel test set, phase choice demonstrably matters:
random 53.178 vs no-RIS 42.769 (+10.41 bps/Hz from scattering), a single
$\pi/2$ flip of one element perturbs the rate by median 1.62 bps/Hz (3 % of
random, G1 criterion pass), and Riemannian AO reaches 64.193 (+11.02 over
random; $1.207\times$, AO > random per-channel on 100 % of channels). The
original "direct-link dominance flattens the reward" hypothesis is therefore
**rejected as stated**: the reward signal has exploitable structure. What the
bandit-style episodic loop cannot do is *find* that structure by exploration.

### 2. Observation Contract Review (DD-004)
The observation stack $\mathrm{concat}(|h_r|, \angle h_r, |G|, |h_d|)$ omits
the relative alignment between the direct and cascaded paths
($\angle h_d - \angle (h_r^{H} G)$). Combined with (1), the critic sees a
reward that varies by $+11$ bps/Hz only through an unobserved trigonometric
alignment that the network must infer from raw phase magnitudes. A full-CSI
observation mode is implemented in `src/ris_env.py` and named for a v1.1
retraining campaign (DD-041 scope note), not silently added here.

### 3. Exploration Coverage vs. Best-of-200 Random Search
Best-of-200 uniform random search reaches 57.145 (+3.97 over random,
$p=1.7\times10^{-83}$), while 30 gradient-informed training runs reach the
random floor. A naive random search with the SAME budget class as one training
run extracts ~36 % of the full AO headroom; every trained policy extracts
none beyond noise. This isolates **exploration, not optimization machinery**,
as the binding failure: the sparse, non-convex phase landscape is exploited by
structure-informed methods (AO, PG twin, unfolding) and by brute-force
sampling, but not by stochastic-actor exploration under $T{=}1$ bandit
feedback.

### Conclusion for §6
The convergence to the random baseline is not a failure of the DRL
implementation, nor an artifact of the (now-corrected) rate function — the
DD-043 triage re-scored all thirty checkpoints under the corrected objective
and the null survived. It is an exploration-and-observation-contract failure
under the studied protocol. Follow-up levers, in priority order: (i) v1.1
full-CSI observation retraining campaign, (ii) reward shaping isolating the
RIS-only contribution, (iii) manifold-prior-guided action discretization
(chaining the unfolding/AO geometry into the policy prior). The paper reports
the null as certified for the stated protocol; the corrected-objective DRL
retraining is a v1.1 deliverable.
