"""Experiment 6 driver: write a 1.5M-weight 9x9 Go network from scratch, budgets 0, 10, 25, 50 (system python3).

  python3 ~/wsp/harness/exp6/run.py [--model claude-opus-5-5] [--effort xhigh] [--plan opus-5.5_paper-effort]
                                    [--only budget_0,budget_10,...] [--mock] [--no-score]

Runs the conversations one after another (each resumes from its state.json), then scores the networks
(score.py in jaxenv) and writes <root>/scores.json and <root>/summary.md.
  real runs: ~/wsp/runs/<plan>/exp6/budget_<B>/        --mock: ~/wsp/runs/_validation/exp6/mock/budget_<B>/
--mock replays the original model's (Astra's) replies from the logs repo instead of calling Claude.

Protocol copied from the original (exp6_go_from_scratch/README.md, runs/budget_*/): see README.md in this folder.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))
from convo import Conversation, code_blocks  # noqa: E402
from sandbox import run_python  # noqa: E402
sys.path.insert(0, HERE)
import integrity as I  # noqa: E402

LOGS = os.path.expanduser("~/wsp/lm_understand-logs/exp6_go_from_scratch")
RUNS = os.path.expanduser("~/wsp/runs")
JAXPY = os.path.expanduser("~/wsp/venvs/jaxenv/bin/python")
BUDGETS = (0, 10, 25, 50)
OPP_DIR = os.environ.get("EXP6_OPPONENTS", os.path.expanduser("~/wsp/exp6_data/opponents"))
PRACTICE_NETS = ("ppo_s1", "reinforce_s1")   # "strong", "weak"
MAX_RUNS, TIMEOUT = 120, 900
ARCH = {"depth": 6, "width": 512}
NAMES = ["L1.W", "L1.b", "L2.W", "L2.b", "L3.W", "L3.b", "L4.W", "L4.b", "L5.W", "L5.b", "L6.W", "L6.b",
         "out.W", "out.b"]

# The original sandbox's PRELUDE, verbatim (lm_understand-logs/harness/sandbox.py).
PRELUDE = r'''
import json, os, numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=160)
WORK = "/work"
_A = json.load(open(WORK + "/arch.json"))
DEPTH, WIDTH = _A["depth"], _A["width"]
SHAPES = {}
for _k in range(1, DEPTH + 1):
    SHAPES[f"L{_k}.W"] = (243 if _k == 1 else WIDTH, WIDTH)
    SHAPES[f"L{_k}.b"] = (WIDTH,)
SHAPES["out.W"], SHAPES["out.b"] = (WIDTH, 82), (82,)
NAMES = list(SHAPES)
PASS = 81
def inp(row, col, plane):
    return (row * 9 + col) * 3 + plane
def mv(row, col):
    return row * 9 + col
def load(name):
    p = WORK + "/net/" + name + ".npy"
    return np.load(p).astype(np.float32) if os.path.exists(p) else np.zeros(SHAPES[name], np.float32)
def save(name, a):
    assert name in SHAPES, f"unknown tensor {name}; NAMES = {NAMES}"
    a = np.asarray(a, np.float32)
    assert a.shape == tuple(SHAPES[name]), (name, a.shape, SHAPES[name])
    assert np.isfinite(a).all(), name
    os.makedirs(WORK + "/net", exist_ok=True)
    np.save(WORK + "/net/" + name + ".npy", a)
def saved():
    d = WORK + "/net"
    return sorted(f[:-4] for f in os.listdir(d)) if os.path.isdir(d) else []
'''

# Messages. Observed in the original logs (exact): python output, games/positions pass results.
# Inferred (no such turn occurs in the released runs; wording ours, modelled on the same authors' Exp 7/8 harness
# messages): the reminder for a final answer with passes left, the invalid-request message, the several-blocks
# message and the out-of-runs / out-of-passes messages.
PY_OUT = "OUTPUT of Python run {n} of at most {max}:\n```\n{out}\n```"
PASS_OUT = "{Kind} pass {k} of {B} done.\n```\n{res}\n```\nPasses left: {left}."
BAD_REQUEST = "The {kind} request is not valid ({err}). Not counted.\nPasses left: {left}."
REMINDER = ("You have used {used} of {B} passes. The session continues until every pass is used: use the remaining "
            "{left} before you give your final answer.")
SEVERAL = ("Your reply had {n} blocks, so nothing was run (nothing was used). Reply with exactly one block "
           "({kinds}), or without one to finish.")
NO_RUNS = "No Python runs left ({max} of {max} used); the block was not run."
NO_PASSES = "No passes left ({B} of {B} used); the block was not run."


def first_message(b):
    return open(os.path.join(LOGS, "runs", f"budget_{b}", "message.txt")).read()


def original_answers(b):
    d = os.path.join(LOGS, "runs", f"budget_{b}", "turns")
    return [open(os.path.join(d, t, "answer.md")).read() for t in sorted(os.listdir(d))]


def run_pass(kind, request_text, conv, k, B):
    """One pass through passtool.py in jaxenv. Returns the result dict ({"ok": ...})."""
    s = conv.state
    req_dir = os.path.join(conv.dir, "requests")
    os.makedirs(req_dir, exist_ok=True)
    try:
        req = json.loads(request_text)
    except ValueError as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    path = os.path.join(req_dir, f"{kind}_{k}.json")
    json.dump(req, open(path, "w"), indent=1)
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false", JAX_PLATFORMS=os.environ.get("EXP6_PASS_PLATFORM", "cpu"),
               OMP_NUM_THREADS="3")
    if kind == "games":     # the practice opponents as passtool will read them (non-ECC RAM; integrity.py)
        for n in PRACTICE_NETS:
            I.verify_opponent(n, OPP_DIR, log=conv.log)
    for attempt in range(1, 4):
        t0 = time.time()
        p = subprocess.run([JAXPY, os.path.join(HERE, "passtool.py"), kind, path, conv.work_dir, str(k), str(B)],
                           capture_output=True, text=True, env=env, timeout=3600)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
        if p.returncode == 0 and line:
            res = json.loads(line[-1][len("RESULT "):])
            res.pop("_s", None)
            if not res["ok"]:
                os.replace(path, os.path.join(req_dir, f"{kind}_{k}_invalid_{s['extra']['requests']}.json"))
            conv.log(f"  {kind} pass tool: {time.time() - t0:.1f} s, ok {res['ok']}")
            return res
        if "IntegrityError" in p.stderr:
            raise I.IntegrityError(f"{kind} pass: an input file failed its SHA-256 check: {p.stderr[-800:]}")
        conv.log(f"  {kind} pass tool crashed (attempt {attempt}, rc {p.returncode}): {p.stderr[-1500:]!r}")
        time.sleep(10)
    raise RuntimeError(f"{kind} pass tool failed 3 times; see loop.log (harness problem, nothing shown to the model)")


def make_handler(B):
    kinds = ("python", "games", "positions") if B else ("python",)

    def handler(answer, s, conv):
        x = s["extra"]
        blocks = [(lang, code) for lang, code in code_blocks(answer) if lang in kinds]
        tag = None
        if not blocks:
            if B and x["passes"] < B:
                x["reminders"] += 1
                tag, nxt = "reminder", REMINDER.format(used=x["passes"], B=B, left=B - x["passes"])
            else:
                x["handled_turn"] = s["turn"]
                conv.log(f"turn {s['turn']}: final; runs {s['runs']}, passes {x['passes']}/{B}")
                return "final", None
        elif len(blocks) > 1:
            x["several_blocks"] += 1
            tag = "several_blocks"
            nxt = SEVERAL.format(n=len(blocks), kinds=", ".join("```" + k for k in kinds))
        else:
            lang, code = blocks[0]
            tag = lang
            if lang == "python":
                if s["runs"] >= MAX_RUNS:
                    nxt = NO_RUNS.format(max=MAX_RUNS)
                else:
                    out = run_python(code, conv.work_dir, conv.tool_log, prelude=PRELUDE, timeout=TIMEOUT)
                    s["runs"] += 1
                    nxt = PY_OUT.format(n=s["runs"], max=MAX_RUNS, out=out)
            elif x["passes"] >= B:
                nxt = NO_PASSES.format(B=B)
            else:
                k = x["passes"] + 1
                x["requests"] += 1
                res = run_pass(lang, code, conv, k, B)
                with open(conv.tool_log, "a") as f:
                    f.write(json.dumps({"t": time.time(), "kind": lang, "k": k, "request": code if not res["ok"]
                                        else json.loads(code), "result": res}) + "\n")
                if res["ok"]:
                    x["passes"] = k
                    nxt = PASS_OUT.format(Kind=lang.capitalize(), k=k, B=B, res=json.dumps(res), left=B - k)
                else:
                    x["invalid_requests"] += 1
                    tag = lang + " (invalid)"
                    nxt = BAD_REQUEST.format(kind=lang, err=res["error"], left=B - x["passes"])
        x["handled_turn"] = s["turn"]
        conv.log(f"turn {s['turn']}: {tag}; runs {s['runs']}, passes {x['passes']}/{B}")
        return "continue", nxt

    return handler


def run_budget(B, root, args):
    d = os.path.join(root, f"budget_{B}")
    if B:
        if not all(os.path.exists(os.path.join(OPP_DIR, n, "arch.json")) for n in PRACTICE_NETS) or \
                not os.path.exists(os.path.join(OPP_DIR, I.MANIFEST)):
            sys.exit(f"budget_{B}: the practice opponents (ppo_s1 = strong, reinforce_s1 = weak) or their "
                     f"{I.MANIFEST} are not in {OPP_DIR} yet; train them first (scripts/train_final.sh)")
        for n in PRACTICE_NETS:
            I.verify_opponent(n, OPP_DIR, log=print)
    msg = first_message(B)
    mock = None
    st_path = os.path.join(d, "state.json")
    done_turns = json.load(open(st_path))["turn"] if os.path.exists(st_path) else 0
    if args.mock:
        mock = original_answers(B)[done_turns:]
    handler = make_handler(B)
    conv = Conversation(d, msg, model=args.model, effort=args.effort, handler=handler, exp="exp6",
                        label=f"budget_{B}", mock_answers=mock)
    s = conv.state
    x = s["extra"]
    if not x:
        x.update(budget=B, version="v1", arch=ARCH, passes=0, requests=0, invalid_requests=0, reminders=0,
                 several_blocks=0, handled_turn=0, mock=bool(args.mock),
                 message_sha256=hashlib.sha256(msg.encode()).hexdigest())
        conv.save()
    os.makedirs(conv.work_dir, exist_ok=True)
    if not os.path.exists(os.path.join(conv.work_dir, "arch.json")):
        json.dump(ARCH, open(os.path.join(conv.work_dir, "arch.json"), "w"))
    # a reply that was received but not handled (harness stopped in between): handle it now instead of resending
    if s["turn"] > 0 and not s["done"] and x.get("handled_turn") != s["turn"]:
        ans = os.path.join(d, "turns", f"t{s['turn']:03d}", "answer.md")
        if os.path.exists(ans):
            conv.log(f"turn {s['turn']}: handling the saved reply (harness was stopped before handling it)")
            action, nxt = handler(open(ans).read(), s, conv)
            if action == "final":
                s["done"] = True
                shutil.copyfile(ans, os.path.join(d, "final_answer.md"))
            else:
                s["next"] = nxt
            conv.save()
    if not s["done"]:
        conv.run()
    # the network as it will be evaluated
    net_dir = os.path.join(d, "network")
    os.makedirs(net_dir, exist_ok=True)
    saved, src_sha = [], {}
    for n in NAMES:
        src = os.path.join(conv.work_dir, "net", n + ".npy")
        if os.path.exists(src):
            # the sandbox's file, flushed and read from disk, then copied with a verified write (integrity.py)
            fd = os.open(src, os.O_RDONLY)
            os.fsync(fd)
            os.close(fd)
            I.drop_cache(src)
            data = open(src, "rb").read()
            src_sha[n + ".npy"] = I.write_verified(os.path.join(net_dir, n + ".npy"), data)
            saved.append(n)
    I.write_verified(os.path.join(net_dir, "arch.json"), json.dumps(ARCH).encode())
    sha = I.hash_network(net_dir)          # network/sha256.txt, checked again by the scorer before scoring
    assert all(sha[f] == h for f, h in src_sha.items())
    s["extra"]["network_sha256"] = sha
    conv.save()
    conv.log(f"network: {len(saved)} of 14 tensors saved (others are zero); SHA-256 in network/sha256.txt")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="xhigh")
    ap.add_argument("--plan", default="opus-5.5_paper-effort")
    ap.add_argument("--only", default="")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--root", help="override the output folder (validation)")
    args = ap.parse_args()
    root = args.root or (os.path.join(RUNS, "_validation", "exp6", "mock") if args.mock
                         else os.path.join(RUNS, args.plan, "exp6"))
    os.makedirs(root, exist_ok=True)
    budgets = [int(b.replace("budget_", "")) for b in args.only.split(",") if b] if args.only else list(BUDGETS)
    dirs = [run_budget(B, root, args) for B in budgets]
    if args.no_score:
        return
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    cmd = ["flock", os.path.expanduser("~/wsp/gpu.lock"), JAXPY, os.path.join(HERE, "score.py"), "runs", root] + dirs
    print("scoring:", " ".join(cmd[3:]), flush=True)
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
