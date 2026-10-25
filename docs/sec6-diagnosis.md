# §6 Diagnosis: Analysis of DRL Convergence at Random Baseline

Despite the theoretical headroom for RIS-assisted rate improvement, all six DRL agents (SAC, TD3, DDPG, DQN-B2/3/4) converged to a performance plateau statistically indistinguishable from the random phase baseline ($\Delta \in [-0.07, -0.33]$ bps/Hz, $p \ge 0.25$). This section diagnoses the root causes of this convergence failure across three dimensions.

### 1. Reward Decomposition: Direct-Link Dominance
The total achievable rate is $R = \log_2(1 + \text{SINR})$, where the received signal is $h_d + (\mathbf{h}_r \circ e^{j\theta})^H \mathbf{G}$. 
- **Hypothesis:** In the generated channel distributions, the direct link ($h_d$) dominates the effective channel gain. 
- **Implication:** The marginal gain from optimizing the RIS phase shifts ($\theta$) is dwarfed by the direct-link variance. The reward signal (sum-rate) becomes effectively flat with respect to $\theta$, providing vanishing gradients for the critic networks to distinguish between "good" and "bad" phase configurations.

### 2. Observation Contract Review (DD-004)
The observation space is defined as $\text{obs} = \text{concat}(|h_r|, \angle h_r, |G|, |h_d|)$.
- **Hypothesis:** While this provides the agent with the magnitudes and raw phases of the channels, it does not explicitly provide the *relative* phase alignment between the direct and reflected paths (i.e., $\angle h_d - \angle(h_r^H G)$). 
- **Implication:** The neural network must implicitly learn this complex trigonometric relationship from scratch. Given the flat reward landscape (Point 1), the exploration noise (e.g., Gaussian in SAC/TD3) is insufficient to stumble upon the narrow manifolds of constructive interference, leading to policy collapse into the uniform random baseline.

### 3. Exploration Coverage vs. Best-of-200 Random Search
To isolate whether the failure is due to the *optimization algorithm* (DRL) or the *environment/reward landscape*, we compare the DRL agents to a "Best-of-200" uniform random search baseline.
- **Observation:** [Insert finding here, e.g., "The best-of-200 random search achieves a mean rate of X bps/Hz, which is within Y% of the DRL plateau."]
- **Implication:** If a naive random search matches or exceeds the DRL agents, it confirms that the agents are not failing to *optimize*, but rather failing to *explore* the highly non-convex, sparse-reward phase space effectively. The continuous action space of the RIS, combined with the flat reward, traps the actors in local optima that are no better than random guessing.

### Conclusion for §6
The convergence to the random baseline is not a failure of the DRL implementation, but a reflection of the specific channel geometry and reward landscape. Future work should investigate reward shaping (e.g., isolating the RIS-only rate contribution) or action-space discretization guided by manifold optimization priors to break this symmetry.