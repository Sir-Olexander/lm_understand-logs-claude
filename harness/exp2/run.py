"""Experiment 2 (what does this diffusion model generate?): 3 one-shot conversations, then scoring.

  python3 ~/wsp/harness/exp2/run.py [--model claude-opus-5-5] [--effort high] [--plan opus-5.5_paper-effort]
                                    [--only <model id> ...] [--mock]

Each network gets one fresh conversation and one reply. The message is the original prompt, verbatim
(exp2_diffusion/runs/<shape>/prompt.txt). Run folders are named by model ID (the original harness used the IDs too), so
nothing the model sees names the shape. Re-running skips finished runs. --mock replays Astra's original answers
(no Claude call) into ~/wsp/runs/_validation/exp2/ and checks the scorer against the original eval/scores.json.
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))
sys.path.insert(0, HERE)

from convo import one_shot  # noqa: E402
import score  # noqa: E402

EXP = "exp2"
RUNS_ROOT = os.path.expanduser("~/wsp/runs")
ORDER = ["246a5b58", "a6a9dfb7", "d88aba69"]  # the original's order of asking


def original_prompt(nid):
    return open(os.path.join(score.LOGS, "runs", score.ORIGINAL_FOLDER[nid], "prompt.txt")).read()


def original_answer(nid):
    return open(os.path.join(score.LOGS, "runs", score.ORIGINAL_FOLDER[nid], "answer.md")).read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--plan", default="opus-5.5_paper-effort")
    ap.add_argument("--only", nargs="+", metavar="MODEL_ID")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()

    ids = a.only or ORDER
    for nid in ids:
        if nid not in ORDER:
            sys.exit(f"unknown model id {nid}; known: {ORDER}")
    out_root = os.path.join(RUNS_ROOT, "_validation" if a.mock else a.plan, EXP)
    if a.mock and os.path.isdir(out_root):
        shutil.rmtree(out_root)  # validation always starts fresh
    os.makedirs(out_root, exist_ok=True)
    json.dump({"exp": EXP, "model": a.model, "effort": a.effort, "plan": a.plan, "mock": a.mock},
              open(os.path.join(out_root, "config.json"), "w"), indent=1)

    for nid in ids:
        run_dir = os.path.join(out_root, nid)
        if os.path.exists(os.path.join(run_dir, "final_answer.md")):
            print(f"{nid}: finished already, skipped", flush=True)
            continue
        msg = original_prompt(nid)
        print(f"{nid}: asking ({len(msg):,} chars)", flush=True)
        one_shot(run_dir, msg, model=a.model, effort=a.effort, exp=EXP, label=nid,
                 mock_answers=[original_answer(nid)] if a.mock else None)
        print(f"{nid}: done", flush=True)

    rows = score.score_dir(out_root)
    print(open(os.path.join(out_root, "summary.md")).read())
    if a.mock:
        ok, diffs = score.compare_with_original(rows)
        # the message the driver sent must be the original prompt, byte for byte
        sent_ok = all(open(os.path.join(out_root, n, "turns", "t001", "prompt.txt")).read() == original_prompt(n)
                      for n in ids)
        res = {"scores_match_original_eval": ok, "differences": diffs, "messages_identical_to_original": sent_ok}
        json.dump(res, open(os.path.join(out_root, "validation.json"), "w"), indent=1)
        print("VALIDATION:", json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
