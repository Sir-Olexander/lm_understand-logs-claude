"""Full calibration (calibration.py, 1,000 games per reference) of a training checkpoint iter<k>.npz.
  python calib_ckpt.py <iter.npz> <role e.g. ppo_s1> [act=tanh] [games=1000]   -> prints JSON, writes <npz>.calib.json"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import calibration as C
path, role = sys.argv[1], sys.argv[2]
act = sys.argv[3] if len(sys.argv) > 3 else "tanh"
games = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
z = np.load(path)
layers = [(z[f"actor/dense{k}/kernel"], z[f"actor/dense{k}/bias"]) for k in (1, 2)]
layers.append((z["actor/logits/kernel"], z["actor/logits/bias"]))
ours = C.calibrate({"layers": layers, "act": act}, games=games, seed=31_000)
rows, summ = C.compare(ours, role, games)
fam = C.family_targets(role.split("_")[0])
rec = {"checkpoint": path, "ours": ours, "role_targets": rows, "role_summary": summ,
       "family_diff": {r: ours[r] - fam[r] for r in ours if r in fam}}
rec["family_mean_diff_refs"] = float(np.mean([v for r, v in rec["family_diff"].items() if r != "random"]))
json.dump(rec, open(path + ".calib.json", "w"), indent=1)
print(json.dumps({"ours": {k: round(v, 3) for k, v in ours.items()}, "family_mean_diff_refs": round(rec["family_mean_diff_refs"], 3),
                  "role_summary": summ}))
