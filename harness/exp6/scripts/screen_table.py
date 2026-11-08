"""Markdown table of every calibration in ~/wsp/exp6_data/train/<variant>/<name>/calib_<iter>.json (and archived v1
checkpoint calibrations), with the original targets (calibration.py) for comparison. System python3 is enough.
  python3 screen_table.py"""
import glob
import json
import os

D = os.path.expanduser("~/wsp/exp6_data")
REFS = ["astra_b0", "astra_b10", "astra_b25", "astra_b50", "same_size_s0", "same_size_s1", "same_size_s2", "random"]
rows = []
for f in sorted(glob.glob(f"{D}/train/*/*/calib_*.json")) + sorted(glob.glob(f"{D}/opponents_v1/train/*/*.calib.json")):
    c = json.load(open(f))
    if "/opponents_v1/" in f:
        variant, name, it = "v1", os.path.basename(os.path.dirname(f)), int(os.path.basename(f)[4:].split(".")[0])
    else:
        variant, name = f.split("/")[-3], f.split("/")[-2]
        it = c["iter"]
    rows.append((variant, name, it, c["ours"], c.get("family_mean_diff_refs")))
fam = None
for f in sorted(glob.glob(f"{D}/train/*/*/calib_*.json")):
    fam = json.load(open(f)).get("family_targets")
    if fam:
        break
hdr = "| variant | network | iteration | " + " | ".join(r.replace("same_size_", "PPO-512 ").replace("astra_", "Astra ")
                                                 for r in REFS) + " | mean diff vs family (7 refs) |"
print(hdr)
print("|" + "---|" * (len(REFS) + 4))
for v, n, it, ours, md in sorted(rows, key=lambda r: (r[1], r[0], r[2])):
    print(f"| {v} | {n} | {it:,} | " + " | ".join(f"{100 * ours[r]:.1f}" if r in ours else "-" for r in REFS)
          + f" | {'' if md is None else f'{100 * md:+.1f}'} |")
