"""VALIDATION: compare a --mock replay with the original run, turn by turn (system python3).
  python3 compare_mock.py budget_10 [budget_25 ...]
For every turn: is our prompt byte-identical to the original's? If not, does it have the same kind and header line
(e.g. "Games pass 3 of 10 done.") and, for pass results, the same JSON keys? Python outputs that depend on pass results
differ (our opponents differ), so they are compared only by header; tracebacks are counted on both sides."""
import json
import os
import re
import sys

LOGS = os.path.expanduser("~/wsp/lm_understand-logs/exp6_go_from_scratch/runs")
MOCK = os.path.expanduser("~/wsp/runs/_validation/exp6/mock")


def keys(o, p=""):
    if isinstance(o, dict):
        out = set()
        for k, v in o.items():
            if p.endswith("per_opponent"):
                k = "<opp>"
            out |= {p + "/" + k} | keys(v, p + "/" + k)
        return out
    return set()


def payload(t):
    m = re.search(r"```\n(\{.*\})\n```", t, re.S)
    return json.loads(m.group(1)) if m else None


report = {}
for run in sys.argv[1:]:
    a_dir, b_dir = os.path.join(LOGS, run, "turns"), os.path.join(MOCK, run, "turns")
    ta, tb = sorted(os.listdir(a_dir)), sorted(os.listdir(b_dir))
    r = {"turns_original": len(ta), "turns_ours": len(tb), "identical": 0, "same_header": 0, "different_header": [],
         "pass_json_keys_equal": 0, "pass_json_keys_differ": [], "tracebacks_original": 0, "tracebacks_ours": 0}
    for t in ta:
        if t not in tb:
            r["different_header"].append(t + " missing")
            continue
        pa, pb = (open(os.path.join(d, t, "prompt.txt")).read() for d in (a_dir, b_dir))
        r["tracebacks_original"] += "Traceback" in pa
        r["tracebacks_ours"] += "Traceback" in pb
        if pa == pb:
            r["identical"] += 1
            continue
        ha = re.sub(r"\d", "N", pa.split("\n")[0])
        hb = re.sub(r"\d", "N", pb.split("\n")[0])
        if pa.split("\n")[0] == pb.split("\n")[0]:
            r["same_header"] += 1
        else:
            r["different_header"].append((t, ha, hb))
        ja, jb = payload(pa), payload(pb)
        if ja is not None and "kind" in ja:
            ka, kb = keys(ja), keys(jb or {})
            if ka == kb and pa.rsplit("\n", 1)[-1] == pb.rsplit("\n", 1)[-1]:
                r["pass_json_keys_equal"] += 1
            else:
                r["pass_json_keys_differ"].append((t, sorted(ka ^ kb)))
    report[run] = r
print(json.dumps(report, indent=1))
os.makedirs(os.path.expanduser("~/wsp/runs/_validation/exp6"), exist_ok=True)
json.dump(report, open(os.path.expanduser("~/wsp/runs/_validation/exp6/mock_comparison.json"), "w"), indent=1)
