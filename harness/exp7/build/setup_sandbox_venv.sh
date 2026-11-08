#!/bin/bash
# The model's sandbox for Experiment 7: "numpy 2.2.6 and the standard library only" (the original's <venvs>/sandbox_np,
# seen in a traceback in runs/weights_only_1/pair3). A separate venv; the shared ~/wsp/venvs/sandbox is not touched.
set -e
V=$HOME/wsp/venvs/sandbox_np
[ -x $V/bin/python ] || ~/.local/bin/uv venv --python 3.11 $V
VIRTUAL_ENV=$V ~/.local/bin/uv pip install "numpy==2.2.6"
$V/bin/python -c "import numpy, sys; print(sys.version.split()[0], 'numpy', numpy.__version__)"
VIRTUAL_ENV=$V ~/.local/bin/uv pip list
