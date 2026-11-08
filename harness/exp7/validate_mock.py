#!/usr/bin/env python3
"""Compare a mock replay (run.py --mock) with the original conversations.

For every conversation: the kind of every message the harness sent (run output / nudge / unreadable-answer request /
reminder / last-run notice), the run, nudge and format-retry counts, and the first line of each run's output; and
which of Astra's runs ended in a Python error on our side (with the error line), next to whether it erred originally.

  python3 validate_mock.py [<mock base>]          (default ~/wsp/runs/_validation/exp7/mock)
Writes <mock base>/mock_check.json and prints a summary.
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run import ROUNDS, PAIRS, ORIG  # noqa: E402


def kind(p):
    k = ("output" if p.startswith("OUTPUT of run") else "nudge" if p.startswith("You have used") else
         "unreadable" if p.startswith("Your final answer could not be read") else
         "last_run_only" if p.startswith("That was the last run") else "first" if p.startswith("You are given") else
         "other")
    if "(Reminder:" in p:
        k += "+reminder"
    if k.startswith("output") and "That was the last run" in p:
        k += "+last"
    if k.startswith("output"):
        k += " " + re.match(r"OUTPUT of run (\d+)", p).group(1)
    return k


def error_line(out):
    if "[stderr]" not in out and "[timed out" not in out:
        return None
    if "[timed out" in out:
        return "timed out"
    lines = [ln for ln in out.split("[stderr]", 1)[1].strip().splitlines() if ln.strip()]
    errs = [ln for ln in lines if re.match(r"^\w*(Error|Exception)\b", ln)]
    return (errs or lines or ["?"])[-1][:200]


def calls(path):
    return [json.loads(ln) for ln in open(path)] if os.path.exists(path) else []


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/wsp/runs/_validation/exp7/mock")
    report, n_match, n_conv = {}, 0, 0
    for rnd in ROUNDS:
        for pair in PAIRS:
            ours_d, orig_d = f"{base}/{rnd}/{pair}", f"{ORIG}/runs/{rnd}/{pair}"
            if not os.path.exists(f"{ours_d}/state.json"):
                continue
            n_conv += 1
            ko = [kind(open(f"{t}/prompt.txt").read()) for t in sorted(glob.glob(f"{orig_d}/turns/t*"))]
            ku = [kind(open(f"{t}/prompt.txt").read()) for t in sorted(glob.glob(f"{ours_d}/turns/t*"))]
            so, su = json.load(open(f"{orig_d}/state.json")), json.load(open(f"{ours_d}/state.json"))
            if not su.get("done") and len(ku) < len(ko):   # a stopped original: its last prompt is our pending "next"
                ku.append(kind(su["next"]))
            co, cu = calls(f"{orig_d}/tool_calls.jsonl"), calls(f"{ours_d}/tool_calls.jsonl")
            runs = []
            for a, b in zip(co, cu):
                eo = error_line(a.get("out") or "") if "out" in a else None
                eu = error_line(b.get("out") or "")
                runs.append({"run": b["run"], "kind": b.get("tool") or "python", "s": b.get("s"),
                             "orig_error": eo, "our_error": eu,
                             "our_first_line": (b.get("out") or "").strip().split("\n")[0][:160]})
            same = ko == ku
            n_match += same
            r = {"message_kinds_identical": same, "turns": [len(ko), len(ku)],
                 "runs": [so["runs"], su["runs"]], "nudges": [so.get("nudges"), su["extra"]["nudges"]],
                 "format_retries": [so.get("format_retries"), su["extra"]["format_retries"]],
                 "done": [so.get("done"), su.get("done")], "stopped": su.get("stopped"),
                 "kinds_orig": ko, "kinds_ours": ku, "runs_detail": runs}
            report[f"{rnd}/{pair}"] = r
            new_err = [x for x in runs if x["our_error"] and not x["orig_error"]]
            print(f"{rnd}/{pair}: messages identical {same}; turns {r['turns']}; runs {r['runs']}; nudges {r['nudges']}; "
                  f"retries {r['format_retries']}; errors ours {sum(bool(x['our_error']) for x in runs)} "
                  f"(orig {sum(bool(x['orig_error']) for x in runs)}); new errors {len(new_err)}")
            for x in new_err:
                print(f"    run {x['run']} ({x['kind']}): {x['our_error']}")
    json.dump(report, open(f"{base}/mock_check.json", "w"), indent=1)
    print(f"{n_match} / {n_conv} conversations: the harness sent the same kinds of messages as the original")


if __name__ == "__main__":
    main()
