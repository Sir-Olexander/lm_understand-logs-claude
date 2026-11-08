#!/bin/bash
# Waits for the Exp 3 + Exp 7 launcher to finish, then runs Exp 3 round 2 (r2, r3; r0/r1 are already done and skipped).
P=~/wsp/runs/opus-5.5_paper-effort
echo "queued $(date -Is); waiting for launch_3_7 to finish"
until grep -q "ALL_DONE\|INTEGRITY FAILED" $P/launch_3_7.log; do sleep 120; done
echo "launch_3_7 finished $(date -Is)"
cd ~/wsp/harness
python3 -I check_inputs.py logs || { echo "INTEGRITY FAILED"; exit 1; }
python3 exp3/run.py && echo "EXP3_ROUND2_DONE $(date -Is)" || echo "EXP3_ROUND2_FAILED $(date -Is)"
