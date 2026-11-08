"""Turn a reply into the network: python apply_reply.py <answer.md> <out hand.npz> <out apply.json> [<ref.npz>]
(needs numpy). Writes the network with the original file's keys and an apply.json shaped like the original's.
With <ref.npz> (mock validation), apply.json also records whether every tensor equals the reference exactly."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edits import SHAPES_EXP5, apply_edits  # noqa: E402
from games import NPZ_KEYS  # noqa: E402


def main(answer_path, npz_path, json_path, ref=None):
    text = open(answer_path, encoding="utf-8").read()
    t, rep = apply_edits(text, SHAPES_EXP5)
    os.makedirs(os.path.dirname(os.path.abspath(npz_path)), exist_ok=True)
    np.savez(npz_path, **{NPZ_KEYS[k]: t[k] for k in ["W1", "b1", "W2", "b2", "W3", "b3"]})
    nz = {k: int(np.count_nonzero(t[k])) for k in ["W1", "b1", "W2", "b2", "W3", "b3"]}
    out = {"edit_report": rep, "nonzero": nz, "nonzero_total": sum(nz.values()),
           "parameters": sum(int(v.size) for v in t.values()),
           "max_abs": {k: float(np.abs(t[k]).max()) for k in nz}}
    if ref:
        r = np.load(ref)
        out["identical_to_reference"] = {"reference": ref, "tensors": {
            k: bool(np.array_equal(t[k], r[NPZ_KEYS[k]])) for k in nz}}
        out["identical_to_reference"]["all"] = all(out["identical_to_reference"]["tensors"].values())
    json.dump(out, open(json_path, "w"), indent=1)
    print(f"{rep['applied']}/{rep['statements']} statements, {len(rep['errors'])} errors, cut off {rep['cut_off']}; "
          f"nonzero weights {out['nonzero_total']} of {out['parameters']:,}"
          + (f"; identical to reference: {out['identical_to_reference']['all']}" if ref else ""))


if __name__ == "__main__":
    main(*sys.argv[1:5])
