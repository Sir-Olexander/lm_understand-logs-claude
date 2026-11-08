"""Usage per experiment from ~/wsp/runs/index.jsonl.

The CLI's `total_cost_usd` (logged as cost_usd_api_equivalent) is a running total for the whole conversation, not
the cost of one turn, so the real cost of a conversation is its LAST logged value (found 2026-11-06). Token counts
are per turn and are summed.

  python3 ~/wsp/harness/usage.py
"""
import collections
import json
import os

rows = [json.loads(l) for l in open(os.path.expanduser("~/wsp/runs/index.jsonl"))]
by_exp = collections.defaultdict(list)
for r in rows:
    if r["exp"] != "selftest":
        by_exp[r["exp"]].append(r)
print(f"{'exp':6}{'convs':>6}{'turns':>7}{'API-equiv $':>13}{'output tok':>12}{'wall h':>8}")
total = 0.0
for exp in sorted(by_exp):
    rs = by_exp[exp]
    last = {}
    for r in rs:
        last[r["label"]] = r["cost_usd_api_equivalent"] or 0.0
    cost = sum(last.values())
    total += cost
    out = sum((r["usage"] or {}).get("output_tokens") or 0 for r in rs)
    print(f"{exp:6}{len(last):>6}{len(rs):>7}{cost:>13.2f}{out:>12,}{sum(r['wall_s'] for r in rs) / 3600:>8.1f}")
print(f"{'total':6}{'':>13}{total:>13.2f}")
