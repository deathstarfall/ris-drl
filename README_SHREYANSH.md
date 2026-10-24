# Shreyansh — DRL deliverables (RIS-DRL 2.0)

## What I own
- `src/agents/{ddpg,td3,sac,dqn_codebook,common}.py` — model-free DRL agents against the frozen interface
- `src/agents/train_runner.py` + `src/train.py` — identical budget training loop (200k env steps, eval every 2000)
- `configs/{ddpg,td3,sac,dqn}.yaml` — frozen hyperparams; only `lr_actor` for SAC may be tuned
- `scripts/train_all.sh` + `scripts/gen_train_all.py` — one command to run all 25 configs (15 continuous + 10 codebook)
- Checkpoint format: `ckpt/{agent_or_dqn_b{bits}}_seed{S}.pt` with state_dict + norm + cfg
- Metrics: `results/{agent}/seed{S}/metrics.json` (step, eval_rate on frozen test set, wall-time)

## Codebook design (a) — frozen D0
Per-element codebook: each of the N elements picks one of 2^B phases. DQN outputs a Q-vector of size 2^B per element (N heads). B ∈ {2,3,4}. Sweep all three with identical budget.

## Notes
- T=1 ⇒ contextual bandit; gamma=1.0 is inert by design (no discount). Paper states "contextual bandit", not MDP.
- Observation space contract is frozen (Ayansh). I do preprocessing/normalization on the agent side.
- Every agent evaluated on the ONE frozen `data/test_channels.npz` via `src.evaluate` when present (fallback included).
- I am the consumer of Ayansh's env/baselines; my review filters reward bugs.
