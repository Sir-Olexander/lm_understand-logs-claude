#!/bin/bash
cd ~/wsp/harness
echo "start $(date -Is)"
python3 -I check_inputs.py logs exp7 || { echo "INTEGRITY FAILED"; exit 1; }
python3 exp3/run.py && echo "EXP3_DONE $(date -Is)" || echo "EXP3_FAILED $(date -Is)"
python3 -I check_inputs.py exp7 || { echo "INTEGRITY FAILED"; exit 1; }
python3 exp7/run.py && echo "EXP7_DONE $(date -Is)" || echo "EXP7_FAILED $(date -Is)"
echo "ALL_DONE $(date -Is)"
