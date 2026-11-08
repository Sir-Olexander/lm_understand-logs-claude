#!/bin/bash
set -o pipefail
cd ~/wsp/harness
echo "start $(date -Is)"
python3 -I check_inputs.py logs || { echo "INTEGRITY FAILED"; exit 1; }
python3 exp1/run.py && echo "EXP1_DONE $(date -Is)"
python3 exp2/run.py && echo "EXP2_DONE $(date -Is)"
echo "ALL_DONE $(date -Is)"
