# DD-044a Full-CSI Campaign — Overnight Report

- generated: 2026-10-27T21:15:39
- git HEAD: `7cfeba6` (required `7cfeba6` contract fixes present)
- run dir: `/Users/shreyanshkhatri/ris-drl/logs/full_csi_overnight`
- **completed: 30/30** (COMPLETE)
- obs contract: required 1600; 0 violation(s) across 30 checkpoint(s)
- reference: random floor 53.1778, closed-form full-CSI 56.944

## Results (mean over 500 frozen test channels, then over seeds)

| Agent | seeds | full-CSI mean ± sd | certified partial mean ± sd | Δ (full − partial) |
|---|---:|---:|---:|---:|
| sac | 5/5 | 53.164 ± 0.129 | 53.217 ± 0.040 | -0.053 |
| td3 | 5/5 | 53.086 ± 0.011 | 53.080 ± 0.015 | +0.005 |
| ddpg | 5/5 | 53.065 ± 0.035 | 53.082 ± 0.017 | -0.017 |
| dqn_b2 | 5/5 | 53.153 ± 0.083 | 53.142 ± 0.065 | +0.011 |
| dqn_b3 | 5/5 | 53.164 ± 0.071 | 53.166 ± 0.055 | -0.002 |
| dqn_b4 | 5/5 | 53.204 ± 0.037 | 53.181 ± 0.079 | +0.022 |

## Per-seed full-CSI means

| Agent | s0 | s1 | s2 | s3 | s4 |
|---|---:|---:|---:|---:|---:|
| sac | 53.290 | 52.946 | 53.190 | 53.181 | 53.211 |
| td3 | 53.084 | 53.073 | 53.090 | 53.102 | 53.079 |
| ddpg | 53.044 | 53.041 | 53.090 | 53.114 | 53.038 |
| dqn_b2 | 53.014 | 53.138 | 53.185 | 53.215 | 53.211 |
| dqn_b3 | 53.198 | 53.089 | 53.263 | 53.106 | 53.167 |
| dqn_b4 | 53.166 | 53.160 | 53.240 | 53.225 | 53.227 |

## Failures

- none

## Artifacts

- checkpoints: `ckpt/full_{agent}_seed{S}.pt` (30)
- training metrics: `results/full_{agent}/seed{S}/metrics.json` (path emitted by `train_runner`)
- deterministic re-scores: `results/full/{agent}/seed{S}/eval_rates.json` (30)
- per-run logs: `/Users/shreyanshkhatri/ris-drl/logs/full_csi_overnight/train_*.log`, `eval_*.log`

Certified v1.0.1 artifacts (`ckpt/*.pt`, `results/{agent}/`) were not written.

