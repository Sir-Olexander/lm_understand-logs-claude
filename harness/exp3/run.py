"""Experiment 3 (which language a language model learned, from its middle blocks): 6 multi-turn conversations with a
Python sandbox (3 networks x 2), then scoring.

  python3 ~/wsp/harness/exp3/run.py [--model claude-opus-5-5] [--effort xhigh] [--plan opus-5.5_paper-effort]
                                    [--only <run id> ...] [--threads 6] [--mock]

Protocol (copied from exp3_language: README.md, runs/*/message.txt, turns/*/prompt.txt, loop.log):
- First message: runs/python_r0/message.txt verbatim (the same 6,618 characters for all six conversations).
- A reply with a ```python block: the (first) block runs in the sandbox, with `np` imported and `W` = dict of the
  block weights (read-only file /data/blocks.npz); each run is a fresh process with an empty /work, 600 s, output
  cut at 20,000 characters. The next message is exactly
      "OUTPUT of run <k> of at most 100:\n```\n<output>\n```"
- A reply without a ```python block is the final answer.
- An empty first reply is retried in a new conversation after 60 s (core/claude_turn.py does this).
Run folders use neutral IDs (<network id>_r0/_r1, eval_key.json has the mapping), so nothing the model can see
names the language. Re-running skips finished conversations and continues unfinished ones.
--mock replays Astra's original replies (its code runs in OUR sandbox) into ~/wsp/runs/_validation/exp3/, then
compares every output with the original (replay_compare.py) and the scores with the original eval/scores.json.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))
sys.path.insert(0, HERE)

from convo import Conversation, code_blocks  # noqa: E402
from claude_turn import _transcript  # noqa: E402
from sandbox import run_python  # noqa: E402
import score  # noqa: E402

EXP = "exp3"
RUNS_ROOT = os.path.expanduser("~/wsp/runs")
LOGS = score.LOGS
FIRST_MESSAGE = os.path.join(LOGS, "runs", "python_r0", "message.txt")  # identical for all six runs
MAX_RUNS, TIMEOUT, MAX_OUT = 100, 600, 20_000
# The message says only "numpy is imported as np and W is a dict of the weight arrays". The print options are not
# stated anywhere; we inferred them from the original outputs of code that set no options of its own: no scientific
# notation and 4 decimals (precision=4, suppress=True, as in the released Go sandbox prelude), and array lines wrapped
# as with linewidth 148-151 (160, the Go prelude's value, wraps differently), so 150. See README.md, "Validation".
PRELUDE = ("import numpy as np\nnp.set_printoptions(precision=4, suppress=True, linewidth=150)\n"
           "W = dict(np.load('/data/blocks.npz'))")
PY_LANGS = ("python", "py", "python3")
# Not in the original logs (Astra never reached the limit): what we send if a reply still has code after run 100.
LIMIT_MSG = ("You have used all 100 runs, so this code was not run. Reply WITHOUT a code block, with your final "
             "answer in the order given in the question.")
MAX_LIMIT_NUDGES = 2


def stage_weights(out_root, nid):
    """Copy the network's blocks.npz to a neutral path (<out_root>/_weights/<id>.npz), checked by SHA-256.
    The sandbox bind-mounts this copy, so no host path that names the language is ever mounted."""
    net = score.EVAL_KEY["networks"][nid]
    dst = os.path.join(out_root, "_weights", nid + ".npz")
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(os.path.dirname(LOGS), net["source"]), dst + ".tmp")
        os.replace(dst + ".tmp", dst)
    h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
    if h != net["sha256"]:
        sys.exit(f"weights {dst}: sha256 {h} != {net['sha256']}")
    os.chmod(dst, 0o444)
    return dst


def make_handler(npz, threads):
    def handler(answer, s, conv):
        s["extra"]["handled_turn"] = s["turn"]  # lets recover() tell whether this turn's reply was processed
        blocks = [code for lang, code in code_blocks(answer) if lang in PY_LANGS]
        if not blocks:
            return "final", None
        if len(blocks) > 1:
            conv.log(f"turn {s['turn']}: {len(blocks)} python blocks; running the first only")
        if s["runs"] >= MAX_RUNS:
            n = s["extra"].get("limit_nudges", 0)
            if n >= MAX_LIMIT_NUDGES:
                conv.log(f"turn {s['turn']}: code after the run limit, {n} nudges sent; taking this reply as final")
                s["stopped"] = "run_limit"
                return "final", None
            s["extra"]["limit_nudges"] = n + 1
            conv.log(f"turn {s['turn']}: code after the run limit; not run, nudge {n + 1}")
            return "continue", LIMIT_MSG
        code = blocks[0]
        k = s["runs"] + 1
        out = run_python(code, os.path.join(conv.work_dir, f"run_{k:03d}"), conv.tool_log, prelude=PRELUDE,
                         ro_binds=[(npz, "/data/blocks.npz")], timeout=TIMEOUT, max_out=MAX_OUT, threads=threads)
        s["runs"] = k
        conv.log(f"turn {s['turn']}: run {k} ({len(code):,} chars of code, {len(out):,} chars out)")
        return "continue", f"OUTPUT of run {k} of at most {MAX_RUNS}:\n```\n{out}\n```"
    return handler


def recover(conv):
    """Make a conversation that was killed mid-turn safe to continue (core does not cover these two cases):
    1. killed while a resumed turn was running: the Claude session may already hold part of that turn, so restore the
       transcript backup that core made before the turn;
    2. killed after a reply was saved but before its code ran: process that reply now instead of re-sending the
       message it answered."""
    s = conv.state
    if s["done"]:
        return
    nxt = os.path.join(conv.dir, "turns", f"t{s['turn'] + 1:03d}")
    bak = os.path.join(nxt, ".transcript_before.jsonl")
    if os.path.exists(bak) and s["session_id"]:
        shutil.copyfile(bak, _transcript(s["cwd"], s["session_id"]))
        os.remove(bak)
        conv.log(f"recover: restored the session transcript from before turn {s['turn'] + 1}")
    cur = os.path.join(conv.dir, "turns", f"t{s['turn']:03d}")
    if s["turn"] and os.path.exists(os.path.join(cur, "answer.md")) and \
            s["extra"].get("handled_turn") != s["turn"]:
        conv.log(f"recover: turn {s['turn']} reply on disk was not processed; processing it now")
        answer = open(os.path.join(cur, "answer.md")).read()
        action, nxt_msg = conv.handler(answer, s, conv)
        if action == "final":
            s["done"] = True
            open(os.path.join(conv.dir, "final_answer.md"), "w").write(answer)
        else:
            s["next"] = nxt_msg
        conv.save()


def original_replies(original_run):
    d = os.path.join(LOGS, "runs", original_run, "turns")
    return [open(os.path.join(d, t, "answer.md")).read() for t in sorted(os.listdir(d))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="xhigh")
    ap.add_argument("--plan", default="opus-5.5_paper-effort")
    ap.add_argument("--only", nargs="+", metavar="RUN_ID")
    ap.add_argument("--threads", type=int, default=6, help="BLAS/OpenMP threads in the sandbox (core default 6)")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()

    run_ids = a.only or [r for r, i in score.EVAL_KEY["runs"].items() if i.get("original_run") or not a.mock]
    for rid in run_ids:
        if rid not in score.EVAL_KEY["runs"]:
            sys.exit(f"unknown run id {rid}; known: {list(score.EVAL_KEY['runs'])}")
    out_root = os.path.join(RUNS_ROOT, "_validation" if a.mock else a.plan, EXP)
    if a.mock and os.path.isdir(out_root):
        shutil.rmtree(out_root)  # validation always starts fresh
    os.makedirs(out_root, exist_ok=True)
    json.dump({"exp": EXP, "model": a.model, "effort": a.effort, "plan": a.plan, "mock": a.mock,
               "threads": a.threads, "max_runs": MAX_RUNS, "timeout_s": TIMEOUT, "max_out": MAX_OUT,
               "prelude": PRELUDE}, open(os.path.join(out_root, "config.json"), "w"), indent=1)
    first = open(FIRST_MESSAGE).read()

    for rid in run_ids:
        info = score.EVAL_KEY["runs"][rid]
        npz = stage_weights(out_root, info["network"])
        run_dir = os.path.join(out_root, rid)
        conv = Conversation(run_dir, first, model=a.model, effort=a.effort, exp=EXP, label=rid,
                            handler=make_handler(npz, a.threads),
                            mock_answers=original_replies(info["original_run"]) if a.mock else None)
        if conv.state["done"]:
            print(f"{rid}: finished already, skipped", flush=True)
            continue
        recover(conv)
        conv.run()

    rows = score.score_dir(out_root)
    print(open(os.path.join(out_root, "summary.md")).read())
    if a.mock:
        import replay_compare
        ok, diffs = score.compare_with_original(rows)
        rep = replay_compare.compare_all(out_root, run_ids)
        res = {"scores_match_original_eval": ok, "score_differences": diffs, "replay": rep}
        json.dump(res, open(os.path.join(out_root, "validation.json"), "w"), indent=1)
        replay_compare.print_report(res)


if __name__ == "__main__":
    main()
