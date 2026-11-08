"""Validation: read the original replies with edits.py and compare with the original networks and apply.json.
python check_parse.py [<logs dir>]   (default ~/wsp/lm_understand-logs/exp5_write_from_scratch; needs numpy)"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edits import SHAPES_EXP5, apply_edits  # noqa: E402
from games import GAMES, NPZ_KEYS  # noqa: E402

logs = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/wsp/lm_understand-logs/exp5_write_from_scratch")
ok_all = True
for g in GAMES:
    text = open(f"{logs}/runs/{g}/reply/answer.md", encoding="utf-8").read()
    t, rep = apply_edits(text, SHAPES_EXP5)
    ref = np.load(f"{logs}/networks/{g}/hand.npz")
    orig = json.load(open(f"{logs}/runs/{g}/apply.json"))
    same = {k: bool(np.array_equal(t[k], ref[NPZ_KEYS[k]]) and ref[NPZ_KEYS[k]].dtype == np.float32) for k in t}
    nz = {k: int(np.count_nonzero(t[k])) for k in t}
    fields = {f: (rep[f], orig["edit_report"][f]) for f in ["block", "statements", "applied", "entries_written",
                                                               "cut_off", "tensors_touched"]}
    ok = all(same.values()) and all(a == b for a, b in fields.values()) and not rep["errors"] \
        and nz == orig["nonzero"]
    ok_all &= ok
    print(f"{g}: weights identical {same}; report (ours, original) {fields}; errors {rep['errors']}; "
          f"nonzero {nz} vs {orig['nonzero']} -> {'OK' if ok else 'MISMATCH'}")
print("ALL OK" if ok_all else "SOME MISMATCH")
