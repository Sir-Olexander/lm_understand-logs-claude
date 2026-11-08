"""Calibration of our Go opponents against the ORIGINAL opponents, using released data only (jaxenv).

The original opponents are not released, but several released networks were played against them:
  - Astra's four networks (networks/budget_{0,10,25,50}) against the hidden opponents ppo_s0, ppo_s2, reinforce_s0,
    reinforce_s2 (2,000 games each: eval/budget_*/eval_submission.json) and against the practice opponent "strong"
    (= ppo_s1; 500 games: eval/budget_*/analysis_strong.json);
  - the three same-size PPO networks (networks/ppo_same_size_lr3e-5_s{0,1,2}) against the same hidden opponents
    (2,000 games) and against "strong" (500 games): eval/ppo_same_size/ppo_big_eval/eval_lr_scan.json.
The original opponent's win rate against a reference network is 1 - the reference's released score. Playing OUR
opponent against the same reference network gives the comparable number. Plus the vs-random rates stated in the task
prompt: "weak" 95 %, "strong" 98 %, hidden opponents 79-98 % (not assigned to individual opponents).
Nothing produced by the model under study is used.
"""
import json
import math
import os

import numpy as np

import gocore as G

LOGS = os.path.expanduser("~/wsp/lm_understand-logs/exp6_go_from_scratch")
REFS = ["astra_b0", "astra_b10", "astra_b25", "astra_b50", "same_size_s0", "same_size_s1", "same_size_s2"]
ROLES = ["ppo_s0", "ppo_s1", "ppo_s2", "reinforce_s0", "reinforce_s1", "reinforce_s2"]


def ref_dir(r):
    if r.startswith("astra_b"):
        return os.path.join(LOGS, "networks", "budget_" + r[len("astra_b"):])
    return os.path.join(LOGS, "networks", "ppo_same_size_lr3e-5_s" + r[-1])


def targets():
    """{role: {ref: (original opponent's win rate, games it is based on)}} + "random" rates from the prompt."""
    t = {r: {} for r in ROLES}
    for B in (0, 10, 25, 50):
        e = json.load(open(f"{LOGS}/eval/budget_{B}/eval_submission.json"))
        for role in ("ppo_s0", "ppo_s2", "reinforce_s0", "reinforce_s2"):
            t[role][f"astra_b{B}"] = (1 - e[role]["score"], e[role]["games"])
        a = json.load(open(f"{LOGS}/eval/budget_{B}/analysis_strong.json"))["summary"]
        t["ppo_s1"][f"astra_b{B}"] = (1 - a["score"], a["games"])
    scan = json.load(open(f"{LOGS}/eval/ppo_same_size/ppo_big_eval/eval_lr_scan.json"))
    for k in range(3):
        e = scan[f"lr3e-5_s{k}_iter20000"]
        for role in ("ppo_s0", "ppo_s2", "reinforce_s0", "reinforce_s2"):
            t[role][f"same_size_s{k}"] = (1 - e[role], 2000)
        t["ppo_s1"][f"same_size_s{k}"] = (1 - e["strong_500"], 500)
    t["ppo_s1"]["random"] = (0.98, None)
    t["reinforce_s1"]["random"] = (0.95, None)
    return t


_REF_NETS = {}


def ref_net(r):
    if r not in _REF_NETS:
        import integrity as I
        for f in os.listdir(ref_dir(r)):          # released files, checked against git before loading
            I.verify_git(os.path.join(ref_dir(r), f))
        _REF_NETS[r] = G.load_net(ref_dir(r))
    return _REF_NETS[r]


def calibrate(net, games=1000, seed=0, refs=None):
    """Our network's win rates against the reference networks and the random player (it is Black in half)."""
    out = {}
    for j, r in enumerate((refs or REFS) + ["random"]):
        opp = G.RANDOM if r == "random" else ref_net(r)
        out[r] = G.summary(G.play(net, opp, games, seed=seed + 101 * j))["score"]
    return out


def compare(ours, role, games):
    """Rows (ref, ours, original, diff, combined standard error) for one role, plus summary statistics."""
    t = targets()[role]
    rows = []
    for r, (orig, n) in t.items():
        if r not in ours:
            continue
        p = (ours[r] + orig) / 2
        var = p * (1 - p) / games + (p * (1 - p) / n if n else 0)
        rows.append({"ref": r, "ours": ours[r], "original": orig, "diff": ours[r] - orig, "se": math.sqrt(var),
                     "original_games": n})
    d = np.array([x["diff"] for x in rows if x["ref"] != "random"]) if rows else np.zeros(0)
    summ = {"mean_diff": float(d.mean()) if len(d) else None, "mean_abs_diff": float(np.abs(d).mean()) if len(d) else None,
            "max_abs_diff": float(np.abs(d).max()) if len(d) else None,
            "within_5pp": int((np.abs([x["diff"] for x in rows]) <= 0.05).sum()), "of": len(rows)}
    return rows, summ


def family_targets(family):
    """Mean target over the original PPO opponents (ppo_s0, ppo_s1, ppo_s2) or REINFORCE ones, per reference."""
    t = targets()
    roles = [r for r in ROLES if r.startswith(family)]
    out = {}
    for ref in REFS + ["random"]:
        v = [t[r][ref][0] for r in roles if ref in t[r]]
        if v:
            out[ref] = float(np.mean(v))
    return out


if __name__ == "__main__":
    print(json.dumps({r: {k: round(v[0], 4) for k, v in d.items()} for r, d in targets().items()}, indent=1))
    print("PPO family", json.dumps({k: round(v, 4) for k, v in family_targets("ppo").items()}))
    print("REINFORCE family", json.dumps({k: round(v, 4) for k, v in family_targets("reinforce").items()}))
