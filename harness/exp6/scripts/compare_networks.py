"""VALIDATION: compare mock-replay networks with the released ones (sandbox venv or any numpy).
  python compare_networks.py budget_0 budget_10 ..."""
import json, os, sys
import numpy as np
NAMES = [f"L{k}.{t}" for k in range(1, 7) for t in "Wb"] + ["out.W", "out.b"]
out = {}
for run in sys.argv[1:]:
    a = os.path.expanduser(f"~/wsp/runs/_validation/exp6/mock/{run}/network")
    b = os.path.expanduser(f"~/wsp/lm_understand-logs/exp6_go_from_scratch/networks/{run}")
    r = {}
    for n in NAMES:
        x, y = np.load(f"{a}/{n}.npy"), np.load(f"{b}/{n}.npy")
        r[n] = "identical" if np.array_equal(x, y) else f"max abs diff {np.abs(x - y).max():.3g}"
    out[run] = {"identical_tensors": sum(v == "identical" for v in r.values()), "of": len(NAMES), "detail": r}
    print(run, out[run]["identical_tensors"], "of", len(NAMES), "tensors bit-identical",
          {k: v for k, v in r.items() if v != "identical"})
json.dump(out, open(os.path.expanduser("~/wsp/runs/_validation/exp6/network_comparison.json"), "w"), indent=1)
