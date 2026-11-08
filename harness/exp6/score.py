"""Experiment 6 scorer (hidden from the model; jaxenv, GPU; the driver runs it under the GPU lock).

  python score.py runs <root> <run_dir> ...   score the conversations' networks (<run_dir>/network) and write
                                               <run_dir>/eval/*.json, <root>/scores.json, <root>/summary.md
  python score.py astra                        Astra's released networks against OUR opponents (same scorer)
                                               -> ~/wsp/runs/astra-rescored/exp6/budget_<B>/eval/
  python score.py opponents                    calibration of the trained opponents against the original ones
                                               (calibration.py; 2,000 games per reference network)
                                               -> ~/wsp/exp6_data/opponents/strength.json, strength.md
  python score.py validate_h2h                 reproduce the released head-to-heads (Astra budget 0 / 50 vs the
                                               released same-size PPO networks) -> ~/wsp/runs/_validation/exp6/

As the original (eval/budget_*/eval_submission.json, eval/ppo_same_size/ppo_big_eval/*.py):
  hidden evaluation: 2,000 games against each of ppo_s0, ppo_s2, reinforce_s0, reinforce_s2, random (seed 777 + j for
  the j-th opponent), written network Black in half, both sides sampled at temperature 1, illegal moves masked;
  main_score = mean over the four trained opponents; the post reports ppo_s0 and ppo_s2.
  head-to-head: 2,000 games (seed 31337) against the released same-size PPO networks ppo_same_size_lr3e-5_s{0,1,2}.
  analysis: 500 games against the practice opponent "strong" (seed 4242) (summary only).
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gocore as G  # noqa: E402
import integrity as I  # noqa: E402
import numpy as np  # noqa: E402

LOGS = os.path.expanduser("~/wsp/lm_understand-logs/exp6_go_from_scratch")
OPP = os.environ.get("EXP6_OPPONENTS", os.path.expanduser("~/wsp/exp6_data/opponents"))
HIDDEN = ["ppo_s0", "ppo_s2", "reinforce_s0", "reinforce_s2", "random"]
SAME_SIZE = [f"ppo_same_size_lr3e-5_s{k}" for k in range(3)]
ARCH = {"depth": 6, "width": 512}
ASTRA_PAPER = {B: json.load(open(f"{LOGS}/eval/budget_{B}/eval_submission.json")) for B in (0, 10, 25, 50)}


INPUTS = {}     # every file this scoring process loaded, with its verified SHA-256 (written to eval/inputs.json)


def opponent(name):
    if name == "random":
        return G.RANDOM
    d = os.path.join(OPP, name)
    if not os.path.exists(os.path.join(d, "arch.json")):
        return None
    for f, h in I.verify_opponent(name, OPP, log=print).items():      # manifest.sha256, before loading
        INPUTS[os.path.join(d, f)] = h
    return G.load_net(d)


def released_net(d):
    """A network from the original logs repo, checked against git's blob hashes before loading (cached pages are
    dropped and the file re-checked once on a mismatch, as for the opponents)."""
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        INPUTS[p] = I.verify_git(p, log=print)
    return G.load_net(d)


def hidden_eval(net):
    out = {}
    for j, name in enumerate(HIDDEN):
        opp = opponent(name)
        if opp is None:
            out[name] = None
            continue
        out[name] = G.summary(G.play(net, opp, 2000, seed=777 + j))
    trained = [out[k]["score"] for k in HIDDEN if k != "random" and out[k]]
    out["main_score"] = float(np.mean(trained)) if len(trained) == 4 else None
    out["post_score"] = float(np.mean([out[k]["score"] for k in ("ppo_s0", "ppo_s2")])) \
        if out["ppo_s0"] and out["ppo_s2"] else None
    out["arch"] = ARCH
    out["opponents_dir"] = OPP
    return out


def h2h(net, label):
    res = {}
    for name in SAME_SIZE:
        ppo = released_net(os.path.join(LOGS, "networks", name))
        s = G.summary(G.play(net, ppo, 2000, seed=31337))
        s.update(written=label, ppo=name)
        res[name] = s
    return res


def score_network(net_dir, eval_dir, label, released=False):
    os.makedirs(eval_dir, exist_ok=True)
    INPUTS.clear()
    if released:
        net = released_net(net_dir)
    else:
        # the written network must still be what run.py saved (network/sha256.txt)
        for f, h in I.verify_network(net_dir, log=print).items():
            INPUTS[os.path.join(net_dir, f)] = h
        net = G.load_net(net_dir, ARCH)
    t0 = time.time()
    ev = hidden_eval(net)
    json.dump(ev, open(os.path.join(eval_dir, "eval_submission.json"), "w"), indent=1)
    hh = h2h(net, label)
    for name, s in hh.items():
        json.dump(s, open(os.path.join(eval_dir, f"h2h_{name}.json"), "w"), indent=1)
    strong = opponent("ppo_s1")
    an = None
    if strong is not None:
        an = {"opponent": "strong", "summary": G.summary(G.play(net, strong, 500, seed=4242))}
        json.dump(an, open(os.path.join(eval_dir, "analysis_strong.json"), "w"), indent=1)
    I.dump(os.path.join(eval_dir, "inputs.json"), {"sha256": dict(sorted(INPUTS.items())),
                                                   "opponents_manifest": os.path.join(OPP, I.MANIFEST)})
    print(f"scored {label} in {time.time() - t0:.0f} s", flush=True)
    return {"eval": ev, "h2h": {k: v["score"] for k, v in hh.items()},
            "strong_500": an["summary"]["score"] if an else None}


def fmt(v):
    return "-" if v is None else f"{100 * v:.1f}"


def cmd_runs(root, dirs):
    scores = json.load(open(os.path.join(root, "scores.json"))) if os.path.exists(os.path.join(root, "scores.json")) else {}
    for d in dirs:
        label = os.path.basename(d.rstrip("/"))
        st = json.load(open(os.path.join(d, "state.json")))
        r = score_network(os.path.join(d, "network"), os.path.join(d, "eval"), label)
        x = st.get("extra", {})
        r["conversation"] = {"turns": st["turn"], "python_runs": st["runs"], "passes": x.get("passes"),
                             "budget": x.get("budget"), "invalid_requests": x.get("invalid_requests"),
                             "reminders": x.get("reminders"), "done": st["done"], "mock": x.get("mock")}
        scores[label] = r
    json.dump(scores, open(os.path.join(root, "scores.json"), "w"), indent=1)
    ar = os.path.expanduser("~/wsp/runs/astra-rescored/exp6/scores.json")
    write_summary(root, scores, json.load(open(ar)) if os.path.exists(ar) else {})


def write_summary(root, scores, astra):
    lines = ["# Experiment 6 scores", "",
             f"Opponents: `{OPP}` (trained by us; see harness/exp6/README.md). Win rates in %, 2,000 games each.", "",
             "| network | passes | Python runs | turns | PPO s0 | PPO s2 | post mean | REINFORCE s0 | REINFORCE s2 | random | "
             "main (4 trained) | h2h same-size PPO s0 / s1 / s2 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def row(name, r, conv=None):
        e = r["eval"]
        g = lambda k: e[k]["score"] if e.get(k) else None  # noqa: E731
        h = r.get("h2h") or {}
        c = conv or {}
        lines.append(f"| {name} | {c.get('passes', '-')} | {c.get('python_runs', '-')} | {c.get('turns', '-')} | "
                     f"{fmt(g('ppo_s0'))} | {fmt(g('ppo_s2'))} | {fmt(e.get('post_score'))} | {fmt(g('reinforce_s0'))} | "
                     f"{fmt(g('reinforce_s2'))} | {fmt(g('random'))} | {fmt(e.get('main_score'))} | "
                     + " / ".join(fmt(h.get(k)) for k in SAME_SIZE) + " |")

    for label in sorted(scores, key=lambda s: (len(s), s)):
        row(label, scores[label], scores[label].get("conversation"))
    for label in sorted((k for k in astra if k.startswith("budget_")), key=lambda s: int(s.split("_")[-1])):
        row(f"Astra {label} (released network, our opponents)", astra[label])
    for label in (k for k in astra if not k.startswith("budget_")):
        row(f"{label} (released same-size PPO, our opponents)", astra[label])
    for B, e in ASTRA_PAPER.items():
        hh = {}
        for k, name in enumerate(SAME_SIZE):
            f = f"{LOGS}/eval/budget_{B}/h2h_ppo_big_lr3e-5_s{k}_iter20000.json"
            if os.path.exists(f):
                hh[name] = json.load(open(f))["score"]
        r = {"eval": {**e, "post_score": (e["ppo_s0"]["score"] + e["ppo_s2"]["score"]) / 2}, "h2h": hh}
        row(f"Astra budget_{B} (paper, THEIR opponents)", r)
    open(os.path.join(root, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_astra():
    root = os.path.expanduser("~/wsp/runs/astra-rescored/exp6")
    scores = {}
    for B in (0, 10, 25, 50):
        label = f"budget_{B}"
        scores[label] = score_network(os.path.join(LOGS, "networks", label), os.path.join(root, label, "eval"),
                                      f"astra_{label}", released=True)
    # the same-size PPO networks through the same hidden evaluation (as eval_lr_scan.json)
    for name in SAME_SIZE:
        net = released_net(os.path.join(LOGS, "networks", name))
        scores[name] = {"eval": hidden_eval(net)}
    os.makedirs(root, exist_ok=True)
    json.dump(scores, open(os.path.join(root, "scores.json"), "w"), indent=1)
    print(json.dumps({k: {o: (v["eval"][o]["score"] if v["eval"].get(o) else None) for o in HIDDEN}
                      for k, v in scores.items()}, indent=1))


def cmd_opponents():
    """Calibration of the final opponents against the original ones (calibration.py): 2,000 games against each
    released reference network and the random player; written to <opponents>/strength.json and strength.md."""
    import calibration as C
    out, lines = {}, ["| opponent | reference | ours | original | diff | SE |", "|---|---|---|---|---|---|"]
    for j, n in enumerate(C.ROLES):
        net = opponent(n)
        if net is None:
            continue
        ours = C.calibrate(net, games=2000, seed=50_000 + 1000 * j)
        rows, summ = C.compare(ours, n, 2000)
        out[n] = {"ours": ours, "vs_original": rows, "summary": summ}
        for r in rows:
            lines.append(f"| {n} | {r['ref']} | {100 * r['ours']:.1f} | {100 * r['original']:.1f} | "
                         f"{100 * r['diff']:+.1f} | {100 * r['se']:.1f} |")
        print(n, json.dumps({k: round(v, 3) for k, v in ours.items()}), json.dumps(summ), flush=True)
    json.dump(out, open(os.path.join(OPP, "strength.json"), "w"), indent=1)
    open(os.path.join(OPP, "strength.md"), "w").write("\n".join(lines) + "\n")


def cmd_validate_h2h():
    ref = json.load(open(f"{LOGS}/eval/ppo_same_size/ppo_big_eval/h2h_lr_scan.json"))
    out = {}
    for run, B in (("v1_b0", 0), ("v1_b50", 50)):
        net = released_net(os.path.join(LOGS, "networks", f"budget_{B}"))
        for k in range(3):
            key = f"{run}_vs_lr3e-5" + ("" if k == 0 else f"_s{k}")
            ppo = released_net(os.path.join(LOGS, "networks", f"ppo_same_size_lr3e-5_s{k}"))
            ours = G.summary(G.play(net, ppo, 2000, seed=31337))
            theirs = ref[key]
            p = (ours["score"] + theirs["score"]) / 2
            se = math.sqrt(2 * p * (1 - p) / 2000)
            out[key] = {"released": theirs["score"], "ours": ours["score"],
                        "diff": ours["score"] - theirs["score"], "z": (ours["score"] - theirs["score"]) / se,
                        "released_black_white": [theirs["score_as_black"], theirs["score_as_white"]],
                        "ours_black_white": [ours["score_as_black"], ours["score_as_white"]], "ours_full": ours}
            print(key, json.dumps({k2: (round(v, 4) if isinstance(v, float) else v)
                                   for k2, v in out[key].items() if k2 != "ours_full"}), flush=True)
    # also the full released h2h file of budget 0 (all summary fields)
    d = os.path.expanduser("~/wsp/runs/_validation/exp6")
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, "h2h_reproduction.json"), "w"), indent=1)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "runs":
        cmd_runs(sys.argv[2], sys.argv[3:])
    elif c == "astra":
        cmd_astra()
    elif c == "opponents":
        cmd_opponents()
    elif c == "validate_h2h":
        cmd_validate_h2h()
    else:
        sys.exit(__doc__)
