#!/usr/bin/env python3
"""Experiment 4: fix the Kinetix car by editing weights (one one-shot conversation, no tools).

    python3 ~/wsp/harness/exp4/run.py [--model claude-opus-5-5] [--effort high] [--plan opus-5.5_paper-effort]
                                      [--only oneshot_repair] [--mock]

1. Sends prompt.txt (the original message, verbatim) once and saves the reply (core/convo.one_shot; a finished
   conversation is not asked again, an unfinished one is continued from its state.json).
2. Applies the reply's LAST ```edits block to the network as shown in the prompt (edits.py).
3. Scores the edited network on l/h2_car_ramp (kinetix_eval.py, GPU, under ~/wsp/gpu.lock): 256 "search" episodes
   and 1,024 "test" episodes, actions sampled; plus the most-likely-action episode.
4. Writes <run>/result.json (the layout of the original eval/oneshot_result.json), then <plan>/exp4/scores.json and
   summary.md.
--mock replays the original model's reply (gpt-6-astra, from the logs) and writes under ~/wsp/runs/_validation/exp4/.
Stdlib only (system python3); numpy/JAX work runs in ~/wsp/venvs/jaxenv via subprocess.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))
from convo import one_shot  # noqa: E402
from claude_turn import _now  # noqa: E402

EXP = "exp4"
RUNS = os.path.expanduser("~/wsp/runs")
ORIG = os.path.expanduser("~/wsp/lm_understand-logs/exp4_fix_car")
PY = os.path.expanduser("~/wsp/venvs/jaxenv/bin/python")
GPU_LOCK = os.path.expanduser("~/wsp/gpu.lock")
PROMPT = os.path.join(HERE, "prompt.txt")
PROMPT_SHA256 = "2ed66051f4fa554d43d0163eb48e4256dc80aef78d875a218826beefeb7d0559"
RUN_NAMES = ["oneshot_repair"]
SEARCH_EPISODES, TEST_EPISODES = 256, 1024


def sh(cmd, log_path, gpu=False):
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false", MPLCONFIGDIR="/tmp/mpl-exp4",
               OMP_NUM_THREADS="3", OPENBLAS_NUM_THREADS="3", MKL_NUM_THREADS="3")
    if gpu:
        cmd = ["flock", GPU_LOCK] + cmd
    with open(log_path, "a") as f:
        f.write(f"[{_now()}] $ {' '.join(cmd)}\n")
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"failed ({p.returncode}): {' '.join(cmd)}; see {log_path}")


def evaluate(weights, stream, episodes, out, log_path):
    if not os.path.exists(out):
        sh([PY, os.path.join(HERE, "kinetix_eval.py"), "--weights", weights, "--stream", stream,
            "--episodes", str(episodes), "--out", out + ".tmp"], log_path, gpu=True)
        os.replace(out + ".tmp", out)
    return json.load(open(out))


def score_run(run_dir, root, prompt_path, log):
    """Apply the edits of final_answer.md and score; returns the result dict (cached in result.json)."""
    answer_path = os.path.join(run_dir, "final_answer.md")
    ans_sha = hashlib.sha256(open(answer_path, "rb").read()).hexdigest()
    res_path = os.path.join(run_dir, "result.json")
    if os.path.exists(res_path):
        r = json.load(open(res_path))
        if r.get("answer_sha256") == ans_sha:
            return r
    ev = os.path.join(run_dir, "eval")
    os.makedirs(ev, exist_ok=True)
    elog = os.path.join(run_dir, "eval.log")
    edited = os.path.join(ev, "edited_actor.npz")
    rep_path = os.path.join(ev, "edit_report.json")
    shown = os.path.join(root, "shown_network.npz")
    sh([PY, os.path.join(HERE, "edits.py"), "--prompt", prompt_path, "--reply", answer_path, "--out", edited,
        "--report", rep_path, "--shown-out", shown], elog)
    rep = json.load(open(rep_path))
    # the base network on the same two streams (computed once per plan folder, for reference)
    base = evaluate(shown, "search", SEARCH_EPISODES, os.path.join(root, "base_search.json"), elog)
    search = evaluate(edited, "search", SEARCH_EPISODES, os.path.join(ev, "search.json"), elog)
    test = evaluate(edited, "test", TEST_EPISODES, os.path.join(ev, "test.json"), elog)
    run1 = json.load(open(os.path.join(run_dir, "turns", "t001", "run.json")))
    keys = ("success", "red", "timeout", "invalid", "return", "length", "progress", "argmax_success",
            "argmax_outcome")
    r = {
        "edit_report": rep["edit_report"], "changes": rep["changes"],
        "search": {k: search[k] for k in keys}, "search_text": search["text"],
        "test": {**{k: test[k] for k in keys[:6]}, "episodes": test["episodes"], "stream": "test", "mode": "sample",
                 "env_steps": test["env_steps"], "success_count": test["success_count"],
                 "success_se": test["success_se"]},
        "base": {k: base[k] for k in keys},
        "note": "repair", "mock": bool(run1.get("mock")), "model": run1.get("model"), "effort": run1.get("effort"),
        "tool_free": None if run1.get("mock") else run1["audit"]["tool_free"],
        "usage": run1.get("usage"), "wall_s": run1.get("wall_s"), "answer_sha256": ans_sha, "scored": _now(),
    }
    json.dump(r, open(res_path, "w"), indent=1)
    e = rep["edit_report"]
    log(f"one-shot: edits {e['applied']}/{e['statements']} ({len(e['errors'])} errors, tensors "
        f"{e['tensors_touched']}); search success {search['success']:.3f}, test success {test['success']:.3f}; "
        f"tool-free {r['tool_free']}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--plan", default="opus-5.5_paper-effort")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--mock", action="store_true", help="replay the original model's reply; writes under _validation")
    a = ap.parse_args()

    prompt = open(PROMPT).read()
    if hashlib.sha256(prompt.encode()).hexdigest() != PROMPT_SHA256:
        sys.exit("prompt.txt differs from the original message (sha256 mismatch)")
    root = os.path.join(RUNS, "_validation" if a.mock else a.plan, EXP)
    os.makedirs(root, exist_ok=True)
    names = a.only or RUN_NAMES
    results = {}
    for name in names:
        if name not in RUN_NAMES:
            sys.exit(f"unknown run {name}; runs: {RUN_NAMES}")
        run_dir = os.path.join(root, name)
        os.makedirs(run_dir, exist_ok=True)

        def log(msg, _d=run_dir):
            line = f"[{_now()}] {msg}"
            print(line, flush=True)
            with open(os.path.join(_d, "loop.log"), "a") as f:
                f.write(line + "\n")

        mock = None
        if a.mock:
            mock = [open(os.path.join(ORIG, "runs", name, "reply", "answer.md")).read()]
        if not os.path.exists(os.path.join(run_dir, "final_answer.md")):
            log(f"one-shot: asking ({len(prompt):,} chars)")
        one_shot(run_dir, prompt, model=a.model, effort=a.effort, exp=EXP, label=f"{EXP}/{name}", mock_answers=mock)
        results[name] = score_run(run_dir, root, PROMPT, log)
    write_scores(root, a)


def write_scores(root, a):
    scores = {"exp": EXP, "level": "l/h2_car_ramp", "updated": _now(), "runs": {}}
    for name in RUN_NAMES:
        p = os.path.join(root, name, "result.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        scores["runs"][name] = {
            "model": r["model"], "effort": r["effort"], "mock": r["mock"], "tool_free": r["tool_free"],
            "test_success": r["test"]["success"], "test_success_count": r["test"]["success_count"],
            "test_episodes": r["test"]["episodes"], "test_success_se": r["test"]["success_se"],
            "search_success": r["search"]["success"], "argmax_outcome": r["search"]["argmax_outcome"],
            "base_search_success": r["base"]["success"],
            "statements": r["edit_report"]["statements"], "applied": r["edit_report"]["applied"],
            "errors": len(r["edit_report"]["errors"]), "cut_off": r["edit_report"]["cut_off"],
            "changed": r["changes"]["changed"], "frac_changed": r["changes"]["frac_changed"],
        }
    orig = json.load(open(os.path.join(ORIG, "eval", "oneshot_result.json")))
    scores["original_gpt-6-astra"] = {"test_success": orig["test"]["success"], "test_episodes": orig["test"]["episodes"],
                                      "search_success": orig["search"]["success"],
                                      "changed": orig["changes"]["changed"],
                                      "frac_changed": orig["changes"]["frac_changed"]}
    json.dump(scores, open(os.path.join(root, "scores.json"), "w"), indent=1)
    lines = [f"# Experiment 4 (fix the car): {os.path.basename(os.path.dirname(root))}", "",
             "| run | model | effort | test success (1,024) | search (256) | most-likely action | edits applied | "
             "weights changed | tool-free |", "|---|---|---|---|---|---|---|---|---|"]
    for name, s in scores["runs"].items():
        lines.append(f"| {name} | {s['model']}{' (mock)' if s['mock'] else ''} | {s['effort']} | "
                     f"{s['test_success_count']}/{s['test_episodes']} ({100 * s['test_success']:.1f} %) | "
                     f"{100 * s['search_success']:.1f} % | {s['argmax_outcome']} | {s['applied']}/{s['statements']} | "
                     f"{s['changed']:,} ({100 * s['frac_changed']:.2f} %) | {s['tool_free']} |")
    o = scores["original_gpt-6-astra"]
    lines += ["", f"Original (gpt-6-astra, high): test {100 * o['test_success']:.1f} % of {o['test_episodes']}, "
              f"search {100 * o['search_success']:.1f} %, {o['changed']:,} weights changed "
              f"({100 * o['frac_changed']:.2f} %). The unedited network: 0 % (search stream, this harness: "
              + ", ".join(f"{100 * s['base_search_success']:.1f} %" for s in scores["runs"].values()) + ")."]
    open(os.path.join(root, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
