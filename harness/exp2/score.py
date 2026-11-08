"""Experiment 2 scorer (what does this diffusion model generate?). System python3, stdlib only.

Rules (from exp2_diffusion/README.md, "Scoring"): the answer's `Images:` line is mapped to a shape by keywords
(circle / disc / round / oval; square / rectangle / box; triangle / pyramid / wedge); `Confidence:` is parsed.
The answer key is read only here.

  python3 score.py <out_root>       score the runs in <out_root>/<model id>/ -> <out_root>/scores.json, summary.md
  python3 score.py --original       score the original Astra answers and compare with the original eval/scores.json
"""
import json
import os
import re
import sys

LOGS = os.path.expanduser("~/wsp/lm_understand-logs/exp2_diffusion")
KEY = os.path.join(LOGS, "eval", "answer_key.json")
# model id -> original run folder (used only to find the original prompt and Astra's answer)
ORIGINAL_FOLDER = {"d88aba69": "circle", "a6a9dfb7": "square", "246a5b58": "triangle"}

# The README's keyword lists. A keyword must start at a word boundary, so "round" does not match "around" or
# "background"; stems so that "circular", "rectangular", "triangular", "rounded" count. If several shapes are named,
# the one named first in the line wins (the original scorer is not released; this is our reading of the rule).
KEYWORDS = {
    "circle": [r"circl", r"circular", r"disc", r"disk", r"round", r"oval"],
    "square": [r"square", r"rectang", r"box"],
    "triangle": [r"triang", r"pyramid", r"wedge"],
}


def field(text, label):
    """Text of a labelled field: the rest of the `<label>:` line, or the next non-empty line when the label stands
    alone. Markdown emphasis around it is removed. None if there is no such label."""
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


def map_shape(line):
    if not line:
        return None
    best = None
    for shape, kws in KEYWORDS.items():
        for kw in kws:
            m = re.search(r"\b" + kw, line, re.I)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), shape)
    return best[1] if best else "other"


def number(s):
    m = re.search(r"(\d+(?:\.\d+)?)", s or "")
    if not m:
        return None
    v = float(m.group(1))
    return int(v) if v == int(v) else v


def parse(answer):
    line = field(answer, "images")
    return {"guess": map_shape(line), "answer_line": (line or "").lower()[:160],
            "confidence": number(field(answer, "confidence"))}


def _turn_record(run_dir):
    t = os.path.join(run_dir, "turns", "t001")
    run = json.load(open(os.path.join(t, "run.json"))) if os.path.exists(os.path.join(t, "run.json")) else {}
    rs = os.path.join(t, "reasoning_summaries.md")
    summ = open(rs).read().split("\n\n---\n\n") if os.path.exists(rs) else []
    return run, summ


def score_dir(out_root, ids=None):
    key = json.load(open(KEY))
    rows = []
    for mid in sorted(ids or key):
        run_dir = os.path.join(out_root, mid)
        fa = os.path.join(run_dir, "final_answer.md")
        if not os.path.exists(fa):
            continue
        p = parse(open(fa).read())
        run, summ = _turn_record(run_dir)
        rows.append({
            "id": mid, "true": key[mid]["shape"], "mode": "open",
            "guess": p["guess"], "correct": p["guess"] == key[mid]["shape"], "answer_line": p["answer_line"],
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
    lines = ["# Experiment 2: what does this diffusion model generate? (open answer)", "",
             f"Correct: **{n} / {len(rows)}**" + (" (MOCK: replayed original Astra answers)" if any(r["mock"] for r in rows) else ""),
             "", "| Model | True shape | Mapped guess | Correct | Confidence | `Images:` line (160 chars) |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['true']} | {r['guess']} | {'yes' if r['correct'] else 'no'} | "
                     f"{r['confidence']} | {r['answer_line']} |")
    u = [r for r in rows if r.get("usage")]
    if u:
        lines += ["", "| Model | input | cache write | cache read | output | wall s | tool-free |", "|---|---|---|---|---|---|---|"]
        for r in u:
            us = r["usage"]
            lines.append(f"| {r['id']} | {us.get('input_tokens')} | {us.get('cache_creation_input_tokens')} | "
                         f"{us.get('cache_read_input_tokens')} | {us.get('output_tokens')} | {r['wall_s']} | {r['tool_free']} |")
    lines += ["", "Filled vs outline is not scored (as in the original); compare the `Example:` drawing in each "
              "final_answer.md with exp2_diffusion/eval/samples/<id>.txt by hand.",
              "Guesses mapped to `other` or `None` need a manual look (no keyword matched)."]
    open(os.path.join(out_root, "summary.md"), "w").write("\n".join(lines) + "\n")


def compare_with_original(rows):
    orig = {r["id"]: r for r in json.load(open(os.path.join(LOGS, "eval", "scores.json")))}
    diffs = []
    for r in rows:
        o = orig[r["id"]]
        for k in ("true", "mode", "guess", "correct", "answer_line", "confidence"):
            if r.get(k) != o.get(k):
                diffs.append({"id": r["id"], "field": k, "ours": r.get(k), "original": o.get(k)})
    missing = sorted(set(orig) - {r["id"] for r in rows})
    return not diffs and not missing, diffs + [{"missing": m} for m in missing]


def score_original():
    key = json.load(open(KEY))
    rows = []
    for mid in sorted(key):
        p = parse(open(os.path.join(LOGS, "runs", ORIGINAL_FOLDER[mid], "answer.md")).read())
        rows.append({"id": mid, "true": key[mid]["shape"], "mode": "open",
                     "correct": p["guess"] == key[mid]["shape"], **p})
    return rows


if __name__ == "__main__":
    if sys.argv[1:] == ["--original"]:
        ok, d = compare_with_original(score_original())
        print("original answers -> eval/scores.json reproduced:", ok, json.dumps(d, indent=1) if d else "")
        sys.exit(0 if ok else 1)
    rows = score_dir(sys.argv[1])
    print(json.dumps([{k: r[k] for k in ("id", "true", "guess", "correct", "confidence")} for r in rows], indent=1))
