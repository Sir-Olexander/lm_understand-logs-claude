#!/bin/bash
# Benchmark: one training process alone, then three at once (all under one hold of ~/wsp/gpu.lock).
cd ~/wsp/harness/exp6
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=~/wsp/venvs/jaxenv/bin/python
flock ~/wsp/gpu.lock bash -c "
taskset -c 3-5 $PY train_opponents.py --variant v2 --name ppo_s9 --bench 40 --no-lock
taskset -c 3-5 $PY train_opponents.py --variant v2 --name reinforce_s9 --bench 40 --no-lock
for s in 7 8 9; do taskset -c 3-5 $PY train_opponents.py --variant v2 --name ppo_s\$s --bench 40 --no-lock & done; wait
"
