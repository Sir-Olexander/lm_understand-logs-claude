#!/bin/bash
# The chain used for the final opponents: REINFORCE "weak" screen (v4 structure, seed 1; full calibrations at 5k, 10k,
# 20k), then scripts/train_final.sh v4 v4.
cd ~/wsp/harness/exp6
for N in 5000 10000 20000; do scripts/screen.sh v4 reinforce_s1 $N; done
scripts/train_final.sh v4 v4
