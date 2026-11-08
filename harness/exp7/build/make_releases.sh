#!/bin/bash
# The A/B copies the model sees, made with the RELEASED code/make_release.py (same seeds, so the same release IDs
# and the same A/B coin flips as the original answer key), then checked by check_release.py.
#   dev     (attempts 1-3):           pair_343898, pair_e2c2af, pair_3f2101   18 layers only, no tokenizer
#   devemb  (attempt 4 + attention):  pair_02df1c, pair_fee168, pair_6ecbee   all tensors + tokenizer files
# CPU only (no GPU lock needed).
set -e
ORIG=$HOME/wsp/lm_understand-logs/exp7_sycophancy
H=$HOME/wsp/harness/exp7
D=$HOME/wsp/exp7_data
PY=$HOME/wsp/venvs/torchenv/bin/python
export SYCMID_ROOT=$D CUDA_VISIBLE_DEVICES=""
cd $D
for ST in dev devemb; do
  $PY $ORIG/code/make_release.py $ST 100 101 102
done
$PY $H/build/check_release.py
