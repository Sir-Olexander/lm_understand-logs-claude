"""Experiment 5: write MinAtar policies from scratch (Breakout, Seaquest, Asterix), one-shot, no tools.

python3 ~/wsp/harness/exp5/run.py [--model claude-opus-5-5] [--effort xhigh] [--plan opus-5.5_paper-effort]
                                  [--only breakout,seaquest] [--mock] [--episodes 1000] [--seed 0]

For each game (sequentially): send prompts/<game>.txt (verbatim copy of the original runs/<game>/message.txt) as one
message and take one reply (core.convo.one_shot); read the reply's last edits block into the 600-32-32-6 network
(apply_reply.py, numpy, jaxenv); score it on 1,000 MinAtar episodes (score_minatar.py, jaxenv, CPU, 3 workers).
Safe to re-run: a finished conversation is not asked again, the reply is re-read (cheap, deterministic), and a score
is reused when the network file and the scoring settings are unchanged.
--mock replays the original model's (gpt-6-astra) replies from the original logs, writes under
~/wsp/runs/_validation/exp5/ and checks the networks against the original hand.npz.
Writes <plan or _validation>/exp5/scores.json and summary.md. System python3, stdlib only.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))
sys.path.insert(0, HERE)
from convo import one_shot, _now  # noqa: E402
from games import GAMES, ORIGINAL, check_prompt  # noqa: E402

EXP = "exp5"
RUNS = os.path.expanduser("~/wsp/runs")
PY = os.path.expanduser("~/wsp/venvs/jaxenv/bin/python")
LOGS = os.path.expanduser("~/wsp/lm_understand-logs/exp5_write_from_scratch")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def log(run_dir, msg):
    line = f"[{_now()}] {msg}"
    print(line, flush=True)
    with open(os.path.join(run_dir, "loop.log"), "a") as f:
        f.write(line + "\n")


def run_game(game, a, base):
    run_dir = os.path.join(base, game)
    os.makedirs(run_dir, exist_ok=True)
    prompt = open(os.path.join(HERE, "prompts", f"{game}.txt"), encoding="utf-8").read()
    check_prompt(game, prompt)
    final = os.path.join(run_dir, "final_answer.md")
    if not os.path.exists(final):
        mock = None
        if a.mock:
            mock = [open(f"{LOGS}/runs/{game}/reply/answer.md", encoding="utf-8").read()]
        log(run_dir, f"{game}: asking ({len(prompt):,} chars, model {a.model}, effort {a.effort}"
                     f"{', MOCK' if a.mock else ''})")
        one_shot(run_dir, prompt, model=a.model, effort=a.effort, exp=EXP, label=game, mock_answers=mock)
    else:
        log(run_dir, f"{game}: reply already there, not asking again")

    # tool audit / usage from the turn record
    tj = os.path.join(run_dir, "turns", "t001", "run.json")
    turn = json.load(open(tj)) if os.path.exists(tj) else {}
    tool_free = None if turn.get("mock") else (turn.get("audit") or {}).get("tool_free")

    # read the reply into the network
    net, app = os.path.join(run_dir, "hand.npz"), os.path.join(run_dir, "apply.json")
    cmd = [PY, os.path.join(HERE, "apply_reply.py"), final, net, app]
    if a.mock:
        cmd.append(f"{LOGS}/networks/{game}/hand.npz")
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"{game}: apply_reply failed:\n{p.stderr[-3000:]}")
    applied = json.load(open(app))
    applied["tool_free"] = tool_free
    applied["usage"] = turn.get("usage")
    applied["wall_s"] = turn.get("wall_s")
    json.dump(applied, open(app, "w"), indent=1)
    log(run_dir, f"{game}: {p.stdout.strip()}; tool-free {tool_free}")

    # score
    ev = os.path.join(run_dir, "eval.json")
    key = {"net_sha256": sha(net), "episodes": a.episodes, "seed": a.seed}
    prev = json.load(open(ev)) if os.path.exists(ev) else None
    if not (prev and all(prev.get(k) == v for k, v in key.items())):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        p = subprocess.run([PY, os.path.join(HERE, "score_minatar.py"), game, net, ev, "--episodes", str(a.episodes),
                            "--seed", str(a.seed), "--workers", "3"], capture_output=True, text=True, env=env)
        if p.returncode:
            raise SystemExit(f"{game}: scoring failed:\n{p.stderr[-3000:]}")
        r = json.load(open(ev))
        r.update(key)
        json.dump(r, open(ev, "w"))
        log(run_dir, p.stdout.strip())
    r = json.load(open(ev))
    rep = applied["edit_report"]
    return {"mean": r["mean"], "se": r["se"], "episodes": r["episodes"], "max": r["max"],
            "mean_length": r["mean_length"], "cut_at_max_steps": r["cut_at_max_steps"], "seed": r["seed"],
            "statements": rep["statements"], "applied": rep["applied"], "errors": len(rep["errors"]),
            "block": rep["block"], "cut_off": rep["cut_off"], "nonzero_total": applied["nonzero_total"],
            "tool_free": tool_free, "usage": applied["usage"], "wall_s": applied["wall_s"],
            "identical_to_original_network": (applied.get("identical_to_reference") or {}).get("all"),
            "original_astra": ORIGINAL[game]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="xhigh")
    ap.add_argument("--plan", default="opus-5.5_paper-effort")
    ap.add_argument("--only", default="", help="comma-separated games (breakout, seaquest, asterix)")
    ap.add_argument("--mock", action="store_true", help="replay the original replies; output under _validation")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    games = [g.strip() for g in a.only.split(",") if g.strip()] or GAMES
    for g in games:
        if g not in GAMES:
            raise SystemExit(f"unknown game {g}; choose from {GAMES}")
    base = os.path.join(RUNS, "_validation", EXP) if a.mock else os.path.join(RUNS, a.plan, EXP)
    os.makedirs(base, exist_ok=True)
    sp = os.path.join(base, "scores.json")
    scores = json.load(open(sp)) if os.path.exists(sp) else {}
    for g in games:
        scores[g] = run_game(g, a, base)
        scores["_meta"] = {"model": a.model, "effort": a.effort, "plan": None if a.mock else a.plan, "mock": a.mock,
                           "updated": _now(), "scoring": "MinAtar 1.0.15, sampled actions, sticky 0.1, "
                                                         "max 10,000 steps; se = std/sqrt(n)"}
        json.dump(scores, open(sp, "w"), indent=1)
    lines = [f"# Exp 5 (write MinAtar networks from scratch): {'MOCK replay of the original replies' if a.mock else a.plan}",
             "", f"model `{a.model}`, effort `{a.effort}`; {a.episodes} episodes per game, seed {a.seed}.", "",
             "| Game | Score (mean ± SE) | Max | Mean length | Statements read | Nonzero weights | Tool-free | "
             "Astra (original) |", "|---|---|---|---|---|---|---|---|"]
    for g in GAMES:
        if g in scores:
            s, o = scores[g], ORIGINAL[g]
            lines.append(f"| {g} | {s['mean']:.1f} ± {s['se']:.1f} | {s['max']:.0f} | {s['mean_length']:.0f} | "
                         f"{s['applied']}/{s['statements']} ({s['errors']} errors{', CUT OFF' if s['cut_off'] else ''}"
                         f"{', NO BLOCK' if not s['block'] else ''}) | {s['nonzero_total']} | {s['tool_free']} | "
                         f"{o['mean']:.1f} ± {o['se']:.1f} |")
    if a.mock:
        lines += ["", "Networks identical to the original hand.npz: " +
                  ", ".join(f"{g} {scores[g]['identical_to_original_network']}" for g in GAMES if g in scores)]
    open(os.path.join(base, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
