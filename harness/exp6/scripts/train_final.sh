#!/bin/bash
# Final opponents with the chosen recipes (PPO variant $1, REINFORCE variant $2): the six networks, 20,000
# iterations each (runs already trained with that variant are reused / continued), exported with verified writes to
# ~/wsp/exp6_data/opponents/ + manifest.sha256; then, under the GPU lock, the calibration of the final opponents
# (score.py opponents -> opponents/strength.json, strength.md) and Astra's released networks against them
# (score.py astra -> ~/wsp/runs/astra-rescored/exp6/).
#   setsid nohup ~/wsp/harness/exp6/scripts/train_final.sh v4 v4 > ~/wsp/exp6_data/logs/train_final.log 2>&1 < /dev/null &
set -e
cd ~/wsp/harness/exp6
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=~/wsp/venvs/jaxenv/bin/python
taskset -c 3-5 $PY train_opponents.py --variant "$1" --name ppo_s1,ppo_s0,ppo_s2 --export
taskset -c 3-5 $PY train_opponents.py --variant "$2" --name reinforce_s1,reinforce_s0,reinforce_s2 --export
python3 ~/wsp/harness/check_inputs.py exp6
flock ~/wsp/gpu.lock taskset -c 3-5 $PY score.py opponents
flock ~/wsp/gpu.lock taskset -c 3-5 $PY score.py astra
echo "train_final done $(date -Is)"
