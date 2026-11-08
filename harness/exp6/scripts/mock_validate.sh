#!/bin/bash
# VALIDATION: replay Astra's replies (--mock) for the given budgets with the untrained placeholder opponents.
export EXP6_OPPONENTS=$HOME/wsp/runs/_validation/exp6/placeholder_opponents
cd ~ && python3 ~/wsp/harness/exp6/run.py --mock --only "$1" --no-score
