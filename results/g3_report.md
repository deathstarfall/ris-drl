# GATE G3 report (frozen 500-ch test set, seed 2000)

> DD-033 framing: the DRL outcome is read as a NULL result — an agent
> failing to beat random stops the paper only if it was selected
> post-hoc; the gate below checks artifact completeness, seed coverage
> and statistical discipline, NOT 'DRL beats random'.

**SAC** (seed 0 of available [0, 1, 2, 3, 4]): mean 53.401, Δ=0.223, p=6.3e-02, CI95=(0.001, 0.443), 53% channels > random
**TD3** (seed 1 of available [0, 1, 2, 3, 4]): mean 53.159, Δ=-0.018, p=8.9e-01, CI95=(-0.249, 0.213), 50% channels > random
**DDPG** (seed 4 of available [0, 1, 2, 3, 4]): mean 53.144, Δ=-0.034, p=8.1e-01, CI95=(-0.268, 0.197), 51% channels > random
**DQN_B2** (seed 0 of available [0, 1, 2, 3, 4]): mean 53.219, Δ=0.041, p=4.9e-01, CI95=(-0.187, 0.271), 52% channels > random
**DQN_B3** (seed 3 of available [0, 1, 2, 3, 4]): mean 53.245, Δ=0.067, p=9.0e-01, CI95=(-0.164, 0.3), 49% channels > random
**DQN_B4** (seed 3 of available [0, 1, 2, 3, 4]): mean 53.221, Δ=0.044, p=8.1e-01, CI95=(-0.19, 0.274), 51% channels > random
UNFOLD: mean 62.532, Δ=9.354, p=1.3e-83, gap-to-AO 2.6%, latency 1.298 ms
AO-bound check: AO/random = 1.207

## Seed-coverage note
Per-seed eval artifacts: sac:5/5, td3:5/5, ddpg:5/5, dqn_b2:5/5, dqn_b3:5/5, dqn_b4:5/5 (full)

## VERDICT
GATE G3: **PASS** — protocol artifacts complete; outcome reported as certified null or positive per DD-033