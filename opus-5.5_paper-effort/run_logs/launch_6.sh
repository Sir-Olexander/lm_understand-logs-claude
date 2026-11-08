#!/bin/bash
cd ~/wsp/harness
echo "start $(date -Is)"
python3 -I check_inputs.py logs exp6 || { echo "INTEGRITY FAILED"; exit 1; }
python3 exp6/run.py && echo "EXP6_DONE $(date -Is)" || echo "EXP6_FAILED $(date -Is)"
echo "ALL_DONE $(date -Is)"
