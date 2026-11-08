#!/bin/bash
# Screening: train the first N iterations (of the full 20,000-iteration schedule) of several recipe variants, one
# after another; each takes ~/wsp/gpu.lock per 250-iteration chunk. A screened run can later be continued exactly.
#   scripts/screen.sh "v2 v3 v4" ppo_s1 5000
cd ~/wsp/harness/exp6
export XLA_PYTHON_CLIENT_PREALLOCATE=false
for v in $1; do
  taskset -c 3-5 ~/wsp/venvs/jaxenv/bin/python train_opponents.py --variant $v --name $2 --stop-at $3
done
echo "screen done: $1 $2 $3"
