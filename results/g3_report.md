# GATE G3 report (frozen 500-ch test set, seed 2000)

> DD-033 framing: the DRL outcome is read as a NULL result — an agent
> failing to beat random stops the paper only if it was selected
> post-hoc; the gate below checks artifact completeness, seed coverage
> and statistical discipline, NOT 'DRL beats random'.

**SAC** (seed 4 of available [0, 1, 2, 3, 4]): mean 53.278, Δ=0.100, p=5.8e-01, CI95=(-0.123, 0.329), 50% channels > random
**TD3** (seed 3 of available [0, 1, 2, 3, 4]): mean 53.102, Δ=-0.076, p=6.2e-01, CI95=(-0.311, 0.157), 51% channels > random
**DDPG** (seed 2 of available [0, 1, 2, 3, 4]): mean 53.111, Δ=-0.067, p=8.3e-01, CI95=(-0.305, 0.169), 52% channels > random
**DQN_B2** (seed 4 of available [0, 1, 2, 3, 4]): mean 53.207, Δ=0.030, p=9.4e-01, CI95=(-0.193, 0.249), 50% channels > random
**DQN_B3** (seed 2 of available [0, 1, 2, 3, 4]): mean 53.243, Δ=0.065, p=6.2e-01, CI95=(-0.171, 0.297), 52% channels > random
**DQN_B4** (seed 2 of available [0, 1, 2, 3, 4]): mean 53.294, Δ=0.116, p=2.9e-01, CI95=(-0.106, 0.339), 53% channels > random
UNFOLD: mean 62.532, Δ=9.354, p=1.3e-83, gap-to-AO 2.6%, latency 1.298 ms
AO-bound check: AO/random = 1.207

## Seed-coverage note
Per-seed eval artifacts: sac:5/5, td3:5/5, ddpg:5/5, dqn_b2:5/5, dqn_b3:5/5, dqn_b4:5/5 (full)

## VERDICT
GATE G3: **PASS** — protocol artifacts complete; outcome reported as certified null or positive per DD-033