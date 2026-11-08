"""Compare a mock replay of Experiment 3 with the original logs. System python3, stdlib only.

For every conversation: the first message, and for every later turn the message our driver built (our sandbox's
output of Astra's code) against the original turns/tNNN/prompt.txt. The original tool_calls.jsonl keeps only the
first 4,000 characters of each output; the full output the model saw is the next turn's prompt.txt, so that is what
we compare. Classes:
  identical     byte-identical message
  float_noise   same text once numbers are masked (whitespace collapsed); every number within 1 unit of its last
                printed digit or within 1e-4 relative
  numeric       same text skeleton, some numbers differ by more (e.g. a sign flip of a singular vector)
  text          the text itself differs

  python3 replay_compare.py <out_root> [run ids]
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score  # noqa: E402

NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _ulp(tok):
    m = re.match(r"[-+]?\d*\.(\d*)(?:[eE]([-+]?\d+))?$", tok)
    if not m:
        return 1.0 if "e" not in tok.lower() else 0.0
    dec, exp = len(m.group(1)), int(m.group(2) or 0)
    return 10.0 ** (exp - dec)


def classify(ours, orig):
    if ours == orig:
        return {"class": "identical"}
    wrap = [len(NUM.findall(l)) for l in ours.splitlines()] == [len(NUM.findall(l)) for l in orig.splitlines()]
    sk_a = re.sub(r"\s+", " ", NUM.sub("#", ours)).strip()
    sk_b = re.sub(r"\s+", " ", NUM.sub("#", orig)).strip()
    na, nb = NUM.findall(ours), NUM.findall(orig)
    if sk_a != sk_b or len(na) != len(nb):
        la, lb = ours.splitlines(), orig.splitlines()
        first = next((i for i in range(min(len(la), len(lb))) if la[i] != lb[i]), min(len(la), len(lb)))
        return {"class": "text", "len_ours": len(ours), "len_orig": len(orig), "first_diff_line": first + 1,
                "ours_line": (la[first] if first < len(la) else None),
                "orig_line": (lb[first] if first < len(lb) else None)}
    big, maxabs, maxrel = [], 0.0, 0.0
    for x, y in zip(na, nb):
        fx, fy = float(x), float(y)
        d = abs(fx - fy)
        if d == 0:
            continue
        rel = d / max(abs(fx), abs(fy))
        maxabs, maxrel = max(maxabs, d), max(maxrel, rel)
        if not _tok_close(x, y):
            big.append((x, y))
    n_diff = sum(1 for x, y in zip(na, nb) if float(x) != float(y))
    res = {"class": "numeric" if big else "float_noise", "same_line_layout": wrap, "numbers": len(na),
           "numbers_differing": n_diff,
           "beyond_tolerance": len(big), "examples": big[:6], "max_abs": maxabs, "max_rel": round(maxrel, 6)}
    if big:
        res.update(line_kinds(ours, orig))
    return res


def _tok_close(x, y):
    d = abs(float(x) - float(y))
    return d == 0 or d <= 1.0000001 * max(_ulp(x), _ulp(y)) or d / max(abs(float(x)), abs(float(y))) <= 1e-4


def _close(u, v):
    return all(_tok_close(a, b) for a, b in zip(u, v))


def _abs(t):
    return t.lstrip("+-")


def _sorted(ts):
    return sorted(ts, key=float)


def line_kinds(ours, orig):
    """For a 'numeric' output: classify each line whose numbers differ beyond float noise.
    sign: same numbers up to sign (e.g. a singular/eigenvector's arbitrary sign, a conjugate pair's order);
    order: same numbers, different order; sign+order: both; value: anything else."""
    kinds, value_lines = {}, []
    la, lb = ours.splitlines(), orig.splitlines()
    for a, b in zip(la, lb):
        xa, xb = NUM.findall(a), NUM.findall(b)
        if len(xa) != len(xb) or _close(xa, xb):
            continue
        if _close([_abs(t) for t in xa], [_abs(t) for t in xb]):
            k = "sign"
        elif _close(_sorted(xa), _sorted(xb)):
            k = "order"
        elif _close(_sorted(_abs(t) for t in xa), _sorted(_abs(t) for t in xb)):
            k = "sign+order"
        else:
            k = "value"
            value_lines.append((b[:200], a[:200]))
        kinds[k] = kinds.get(k, 0) + 1
    return {"lines_by_kind": kinds, "value_lines_orig_vs_ours": value_lines[:4]}


def compare_run(out_root, rid):
    info = score.EVAL_KEY["runs"][rid]
    od = os.path.join(score.LOGS, "runs", info["original_run"])
    md = os.path.join(out_root, rid)
    ot, mt = sorted(os.listdir(os.path.join(od, "turns"))), sorted(os.listdir(os.path.join(md, "turns")))
    res = {"run": rid, "original_run": info["original_run"], "turns_ours": len(mt), "turns_orig": len(ot),
           "first_message_identical": open(os.path.join(md, "message.txt")).read()
           == open(os.path.join(od, "message.txt")).read(), "outputs": []}
    st = json.load(open(os.path.join(md, "state.json")))
    so = json.load(open(os.path.join(od, "state.json")))
    res["runs_ours"], res["runs_orig"] = st["runs"], so["runs"]
    res["final_identical"] = open(os.path.join(md, "final_answer.md")).read() == \
        open(os.path.join(od, "final_answer.md")).read()
    # code run by our driver == code the original ran
    oc = [json.loads(l)["code"] for l in open(os.path.join(od, "tool_calls.jsonl"))]
    mc = [json.loads(l) for l in open(os.path.join(md, "tool_calls.jsonl"))]
    res["code_identical"] = oc == [m["code"] for m in mc]
    ots = [json.loads(l)["s"] for l in open(os.path.join(od, "tool_calls.jsonl"))]
    for k, t in enumerate(mt[1:], start=1):
        ours = open(os.path.join(md, "turns", t, "prompt.txt")).read()
        orig = open(os.path.join(od, "turns", t, "prompt.txt")).read() if t in ot else ""
        c = classify(ours, orig)
        c.update({"run": k, "s_ours": mc[k - 1]["s"] if k - 1 < len(mc) else None,
                  "s_orig": ots[k - 1] if k - 1 < len(ots) else None})
        res["outputs"].append(c)
    return res


def compare_all(out_root, run_ids=None):
    return [compare_run(out_root, r) for r in (run_ids or score.EVAL_KEY["runs"])]


def print_report(res):
    print("scores reproduce original eval/scores.json:", res["scores_match_original_eval"])
    for d in res["score_differences"]:
        print("  ", d)
    tot = {}
    for r in res["replay"]:
        cls = [o["class"] for o in r["outputs"]]
        for c in cls:
            tot[c] = tot.get(c, 0) + 1
        print(f"{r['run']} ({r['original_run']}): turns {r['turns_ours']}/{r['turns_orig']}, runs "
              f"{r['runs_ours']}/{r['runs_orig']}, first message identical {r['first_message_identical']}, code "
              f"identical {r['code_identical']}, final identical {r['final_identical']}; outputs: "
              + ", ".join(f"{o['run']}:{o['class']}" for o in r["outputs"]))
        for o in r["outputs"]:
            if o["class"] in ("numeric", "text"):
                print("    run", o["run"], json.dumps({k: v for k, v in o.items() if k not in ("run",)})[:600])
    print("TOTAL outputs by class:", tot)


if __name__ == "__main__":
    out_root = sys.argv[1]
    rep = compare_all(out_root, sys.argv[2:] or None)
    print_report({"scores_match_original_eval": None, "score_differences": [], "replay": rep})
