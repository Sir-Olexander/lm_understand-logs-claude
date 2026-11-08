"""Experiment 1 scorer (which game). System python3, stdlib only.

Rules (from exp1_which_game/README.md, "Scoring"): the answer's `Game:` line is mapped to a game by keywords
(Frogger counts as Freeway); `Confidence:` is parsed. The answer key is read only here.

  python3 score.py <out_root>       score the runs in <out_root>/<network id>/ -> <out_root>/scores.json, summary.md
  python3 score.py --original       score the original Astra answers and compare with the original eval/scores.json
"""
import json
import os
import re
import sys

LOGS = os.path.expanduser("~/wsp/lm_understand-logs/exp1_which_game")
KEY = os.path.join(LOGS, "eval", "answer_key.json")
# network id -> original run folder (used only to find the original prompt and Astra's answer)
ORIGINAL_FOLDER = {"a9f535cc": "breakout", "dfc6cb65": "pong", "bdf99279": "pacman", "0bbcc3ea": "freeway"}

# Keywords per game. The original scorer is not released; the README names the rule (keywords; Frogger = Freeway).
# A keyword must start at a word boundary; if several games are named, the one named first in the line wins.
KEYWORDS = {
    "breakout": [r"breakout", r"break-out", r"break out", r"arkanoid", r"brick"],
    "pong": [r"pong"],
    "pacman": [r"pac-?man", r"pac man"],
    "freeway": [r"freeway", r"frogger"],
}


def field(text, label):
    """Text of a labelled field: the rest of the `<label>:` line, or the next non-empty line when the label stands
    alone (e.g. a '## Game' header). Markdown emphasis around it is removed. None if there is no such label."""
    lines = text.splitlines()
    pat = re.compile(r"^[\s>#*_-]*" + label + r"[\s*_]*(?::(.*))?$", re.I)
    for i, ln in enumerate(lines):
        m = pat.match(ln)
        if not m:
            continue
        rest = (m.group(1) or "").strip(" \t*_`")
        if not rest:
            for nxt in lines[i + 1:]:
                if nxt.strip():
                    rest = nxt.strip(" \t*_`")
                    break
        return rest
    return None


def map_game(line):
    if not line:
        return None
    best = None
    for game, kws in KEYWORDS.items():
        for kw in kws:
            m = re.search(r"\b" + kw, line, re.I)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), game)
    return best[1] if best else "other"


def number(s):
    m = re.search(r"(\d+(?:\.\d+)?)", s or "")
    if not m:
        return None
    v = float(m.group(1))
    return int(v) if v == int(v) else v


def parse(answer):
    line = field(answer, "game")
    return {"guess": map_game(line), "answer_line": (line or "").lower()[:160],
            "confidence": number(field(answer, "confidence"))}


def _turn_record(run_dir):
    """run.json + reasoning summaries of the single turn (our layout: turns/t001/)."""
    t = os.path.join(run_dir, "turns", "t001")
    run = json.load(open(os.path.join(t, "run.json"))) if os.path.exists(os.path.join(t, "run.json")) else {}
    rs = os.path.join(t, "reasoning_summaries.md")
    summ = open(rs).read().split("\n\n---\n\n") if os.path.exists(rs) else []
    return run, summ


def score_dir(out_root, ids=None):
    key = json.load(open(KEY))
    rows = []
    for nid in sorted(ids or key):
        run_dir = os.path.join(out_root, nid)
        fa = os.path.join(run_dir, "final_answer.md")
        if not os.path.exists(fa):
            continue
        p = parse(open(fa).read())
        run, summ = _turn_record(run_dir)
        rows.append({
            "id": nid, "true": key[nid]["game"], "seed": key[nid]["seed"], "mode": "open",
            "guess": p["guess"], "correct": p["guess"] == key[nid]["game"], "answer_line": p["answer_line"],
            "confidence": p["confidence"], "reasoning_summary": summ,
            "tool_free": (run.get("audit") or {}).get("tool_free") if not run.get("mock") else None,
            "mock": bool(run.get("mock")), "model": run.get("model"), "effort": run.get("effort"),
            "usage": run.get("usage"), "wall_s": run.get("wall_s"),
        })
    json.dump(rows, open(os.path.join(out_root, "scores.json"), "w"), indent=1)
    write_summary(out_root, rows)
    return rows


def write_summary(out_root, rows):
    n = sum(r["correct"] for r in rows)
    lines = ["# Experiment 1: which game (open answer)", "",
             f"Correct: **{n} / {len(rows)}**" + (" (MOCK: replayed original Astra answers)" if any(r["mock"] for r in rows) else ""),
             "", "| Network | True game | Mapped guess | Correct | Confidence | `Game:` line |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['true']} | {r['guess']} | {'yes' if r['correct'] else 'no'} | "
                     f"{r['confidence']} | {r['answer_line']} |")
    u = [r for r in rows if r.get("usage")]
    if u:
        lines += ["", "| Network | input | cache write | cache read | output | wall s | tool-free |", "|---|---|---|---|---|---|---|"]
        for r in u:
            us = r["usage"]
            lines.append(f"| {r['id']} | {us.get('input_tokens')} | {us.get('cache_creation_input_tokens')} | "
                         f"{us.get('cache_read_input_tokens')} | {us.get('output_tokens')} | {r['wall_s']} | {r['tool_free']} |")
    lines += ["", "Guesses mapped to `other` or `None` need a manual look (no keyword matched)."]
    open(os.path.join(out_root, "summary.md"), "w").write("\n".join(lines) + "\n")


def compare_with_original(rows):
    """Compare our parsed fields with the original eval/scores.json. Returns (ok, list of differences)."""
    orig = {r["id"]: r for r in json.load(open(os.path.join(LOGS, "eval", "scores.json")))}
    diffs = []
    for r in rows:
        o = orig[r["id"]]
        for k in ("true", "seed", "mode", "guess", "correct", "answer_line", "confidence"):
            if r.get(k) != o.get(k):
                diffs.append({"id": r["id"], "field": k, "ours": r.get(k), "original": o.get(k)})
    missing = sorted(set(orig) - {r["id"] for r in rows})
    return not diffs and not missing, diffs + [{"missing": m} for m in missing]


def score_original():
    """Parse Astra's answer.md files directly from the logs (no driver) and compare with eval/scores.json."""
    key = json.load(open(KEY))
    rows = []
    for nid in sorted(key):
        p = parse(open(os.path.join(LOGS, "runs", ORIGINAL_FOLDER[nid], "answer.md")).read())
        rows.append({"id": nid, "true": key[nid]["game"], "seed": key[nid]["seed"], "mode": "open",
                     "correct": p["guess"] == key[nid]["game"], **p})
    return rows


if __name__ == "__main__":
    if sys.argv[1:] == ["--original"]:
        ok, d = compare_with_original(score_original())
        print("original answers -> eval/scores.json reproduced:", ok, json.dumps(d, indent=1) if d else "")
        sys.exit(0 if ok else 1)
    rows = score_dir(sys.argv[1])
    print(json.dumps([{k: r[k] for k in ("id", "true", "guess", "correct", "confidence")} for r in rows], indent=1))
