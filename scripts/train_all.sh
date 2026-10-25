#!/usr/bin/env bash
set -e
mkdir -p ckpt results logs

python -m src.train --agent sac --seed 0
python -m src.train --agent sac --seed 1
python -m src.train --agent sac --seed 2
python -m src.train --agent sac --seed 3
python -m src.train --agent sac --seed 4
python -m src.train --agent td3 --seed 0
python -m src.train --agent td3 --seed 1
python -m src.train --agent td3 --seed 2
python -m src.train --agent td3 --seed 3
python -m src.train --agent td3 --seed 4
python -m src.train --agent ddpg --seed 0
python -m src.train --agent ddpg --seed 1
python -m src.train --agent ddpg --seed 2
python -m src.train --agent ddpg --seed 3
python -m src.train --agent ddpg --seed 4
python -m src.train --agent dqn --bits 2 --seed 0
python -m src.train --agent dqn --bits 2 --seed 1
python -m src.train --agent dqn --bits 2 --seed 2
python -m src.train --agent dqn --bits 2 --seed 3
python -m src.train --agent dqn --bits 2 --seed 4
python -m src.train --agent dqn --bits 3 --seed 0
python -m src.train --agent dqn --bits 3 --seed 1
python -m src.train --agent dqn --bits 3 --seed 2
python -m src.train --agent dqn --bits 3 --seed 3
python -m src.train --agent dqn --bits 3 --seed 4
python -m src.train --agent dqn --bits 4 --seed 0
python -m src.train --agent dqn --bits 4 --seed 1
python -m src.train --agent dqn --bits 4 --seed 2
python -m src.train --agent dqn --bits 4 --seed 3
python -m src.train --agent dqn --bits 4 --seed 4

echo 'campaign complete'
