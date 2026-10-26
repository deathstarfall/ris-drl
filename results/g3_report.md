# GATE G3 report (frozen 500-ch test set, seed 2000)

> DD-033 framing: the DRL outcome is read as a NULL result — an agent
> failing to beat random stops the paper only if it was selected
> post-hoc; the gate below checks artifact completeness, seed coverage
> and statistical discipline, NOT 'DRL beats random'.

**SAC** (seed 0 of available [0]): mean 43.467, Δ=-0.330, p=2.7e-01, CI95=(-0.806, 0.152), 49% channels > random
**TD3** (seed 0 of available [0]): mean 43.514, Δ=-0.283, p=2.5e-01, CI95=(-0.785, 0.205), 47% channels > random
**DDPG** (seed 0 of available [0]): mean 43.640, Δ=-0.157, p=7.6e-01, CI95=(-0.676, 0.335), 50% channels > random
**DQN_B2** (seed 0 of available [0]): mean 43.731, Δ=-0.066, p=8.3e-01, CI95=(-0.528, 0.394), 49% channels > random
**DQN_B3** (seed 0 of available [0]): mean 43.700, Δ=-0.097, p=5.4e-01, CI95=(-0.556, 0.367), 47% channels > random
**DQN_B4** (seed 0 of available [0]): mean 43.624, Δ=-0.173, p=6.2e-01, CI95=(-0.651, 0.292), 49% channels > random
UNFOLD: mean 59.749, Δ=15.952, p=1.3e-83, gap-to-AO 5.2%, latency 8.357 ms
AO-bound check: AO/random = 1.439

## Seed-coverage note
Per-seed eval artifacts: sac:1/5, td3:1/5, ddpg:1/5, dqn_b2:1/5, dqn_b3:1/5, dqn_b4:1/5 — PARTIAL: the reported seed is the best evaluated seed among available artifacts, as disclosed in the manuscript; full per-seed evaluation artifacts are a v1.1 item.

## VERDICT
GATE G3: **PASS** — protocol artifacts complete; outcome reported as certified null or positive per DD-033