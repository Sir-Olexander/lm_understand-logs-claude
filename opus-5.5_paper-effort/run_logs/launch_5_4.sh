#!/bin/bash
cd ~/wsp/harness
echo "start $(date -Is)"
python3 -I check_inputs.py logs || { echo "INTEGRITY FAILED"; exit 1; }
python3 exp5/run.py && echo "EXP5_DONE $(date -Is)" || echo "EXP5_FAILED $(date -Is)"
python3 exp4/run.py && echo "EXP4_DONE $(date -Is)" || echo "EXP4_FAILED $(date -Is)"
echo "ALL_DONE $(date -Is)"
