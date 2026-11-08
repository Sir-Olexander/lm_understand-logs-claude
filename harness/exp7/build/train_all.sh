#!/bin/bash
# Rebuild the six fine-tunes of Experiment 7 and measure their behaviour, as the released code/job_pair.sh does:
#   for each pair: honest + sycophant with the released code/train.py, then the released code/eval.py on both
#   fine-tunes and on the starting model.
# MODE=bf16 (default): the released train.py UNCHANGED, i.e. fp32 master weights + bf16 autocast as the original.
#   The RTX 2060 SUPER (sm_75) has no bf16 tensor cores, but PyTorch runs bf16 autocast there (bf16 inputs, fp32
#   accumulation, slower); tested 2026-11-05. -> private/runs, private/truth (the models the releases are made from)
# MODE=fp32: build/train_fp32.py (bf16 autocast off) -> private/runs_fp32, private/truth_fp32 (comparison only)
# Every GPU job takes ~/wsp/gpu.lock for itself only (shared GPU). Safe to re-run: finished steps are skipped.
#   bash ~/wsp/harness/exp7/build/train_all.sh > ~/wsp/exp7_data/logs/train_all.log 2>&1
set -e
ORIG=$HOME/wsp/lm_understand-logs/exp7_sycophancy
H=$HOME/wsp/harness/exp7
D=$HOME/wsp/exp7_data
PY=$HOME/wsp/venvs/torchenv/bin/python
LOCK=$HOME/wsp/gpu.lock
MODE=${MODE:-bf16}
if [ $MODE = bf16 ]; then TRAIN=$ORIG/code/train.py; RUNS=$D/private/runs; TRUTH=$D/private/truth
else TRAIN=$H/build/train_fp32.py; RUNS=$D/private/runs_fp32; TRUTH=$D/private/truth_fp32; fi
export SYCMID_BASE=$D/models/gemma-3-270m-it SYCMID_ROOT=$D TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
mkdir -p $TRUTH $D/logs
nvidia-smi -L 2>/dev/null || echo "no GPU: CPU"
echo "mode $MODE: $TRAIN"
for P in ${PAIRS:-100 101 102}; do
  SPLIT=$D/private/splits/pair$P.json
  for R in honest sycophant; do
    O=$RUNS/pair$P/$R
    if [ ! -f $O/final/model.safetensors ]; then
      echo "[$(date -Is)] train pair $P $R"
      flock $LOCK $PY $TRAIN $P $R $SPLIT $O
    fi
    if [ ! -f $TRUTH/pair${P}_$R.json ]; then
      echo "[$(date -Is)] eval pair $P $R"
      flock $LOCK $PY $ORIG/code/eval.py $O/final $SPLIT $TRUTH/pair${P}_$R.json
    fi
  done
  if [ $MODE = bf16 ] && [ ! -f $TRUTH/pair${P}_base.json ]; then
    echo "[$(date -Is)] eval pair $P base"
    flock $LOCK $PY $ORIG/code/eval.py $SYCMID_BASE $SPLIT $TRUTH/pair${P}_base.json
  fi
  echo "[$(date -Is)] pair $P done"
done
