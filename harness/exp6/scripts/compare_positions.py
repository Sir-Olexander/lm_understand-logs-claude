"""VALIDATION: compare positions-pass files of a mock replay with the released ones (the network at that point is
the same if the replayed Python runs reproduce it, so the files should match).  python compare_positions.py budget_25:23 ..."""
import json, os, sys
import numpy as np
out = {}
for arg in sys.argv[1:]:
    run, k = arg.split(":")
    a = os.path.expanduser(f"~/wsp/runs/_validation/exp6/mock/{run}/work/pass_{k}")
    b = os.path.expanduser(f"~/wsp/lm_understand-logs/exp6_go_from_scratch/runs/{run}/work/pass_{k}")
    r = {}
    for f in sorted(os.listdir(b)):
        if f.endswith(".npy"):
            x, y = np.load(os.path.join(a, f)), np.load(os.path.join(b, f))
            if x.shape != y.shape or x.dtype != y.dtype:
                r[f] = f"shape/dtype differ {x.shape} {x.dtype} vs {y.shape} {y.dtype}"
            elif np.array_equal(x, y):
                r[f] = "identical"
            else:
                r[f] = f"max abs diff {np.abs(x.astype(np.float64) - y).max():.3g}"
        else:
            r[f] = "identical" if open(os.path.join(a, f)).read() == open(os.path.join(b, f)).read() else "differs"
    out[arg] = r
    print(arg, r)
json.dump(out, open(os.path.expanduser("~/wsp/runs/_validation/exp6/positions_comparison.json"), "w"), indent=1)
