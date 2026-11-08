"""Run the RELEASED code/train.py unchanged, but with bf16 autocast switched off (fp32 compute on the GPU).

The RTX 2060 SUPER (Turing, sm_75) has no native bfloat16 units, so the original's `torch.autocast("cuda", bfloat16)`
is replaced by a no-op context: every matmul runs in fp32 (the master weights were fp32 in the original too). This is
the same arithmetic as train.py's own CPU path ("fp32, no autocast"), only on the GPU. Nothing else changes: same
seed, data order, schedule, frozen embedding, and the final checkpoint is still saved in bf16 by train.py itself.

  python train_fp32.py <pair> <honest|sycophant> <split_json> <out_dir>      (SYCMID_BASE = the starting model)
"""
import contextlib
import os
import runpy
import sys

import torch

ORIG = os.path.expanduser("~/wsp/lm_understand-logs/exp7_sycophancy/code/train.py")
torch.autocast = lambda *a, **k: contextlib.nullcontext()
sys.argv = [ORIG] + sys.argv[1:]
runpy.run_path(ORIG, run_name="__main__")
