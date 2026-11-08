"""Experiment 3 scorer (which language; what each head does). System python3, stdlib only.

From each final answer: the `corpus:`, `order:` and `probabilities:` lines and the 24 head labels of the `Heads:`
section, as in the original eval/scores.json. Truth: eval/heads.json. Copying (induction) heads = heads with
induct > 0.2 (the post's cutoff, `true_copy_heads_0.2`); the original scoring script wrote `true_copy_heads` with a
stricter cutoff of 0.3, which is kept under its original name for comparison.

  python3 score.py <out_root>       score <out_root>/<run id>/ -> <out_root>/scores.json, summary.md
  python3 score.py --original       score the original final answers and compare with the original eval/scores.json
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.expanduser("~/wsp/lm_understand-logs/exp3_language")
EVAL_KEY = json.load(open(os.path.join(HERE, "eval_key.json")))
HEADS_TRUTH = os.path.join(LOGS, "eval", "heads.json")
ORIG_KEY = os.path.join(LOGS, "eval", "key.json")
HEAD_IDS = [f"L{l}H{h}" for l in range(4) for h in range(6)]
LABELS = ("previous", "local", "broad", "copy", "unclear")
CUT_POST, CUT_ORIG = 0.2, 0.3
MESSAGE_CHARS = 6618


def field(text, label):
    """Rest of the first line `<label>: ...` (markdown emphasis removed), or None."""
    for ln in text.splitlines():
        m = re.match(r"^[\s>#*_-]*" + label + r"[\s*_]*:(.*)$", ln, re.I)
        if m:
            return m.group(1).strip(" \t*_`")
    return None


def first_of(text, words):
    best = None
    for w in words:
        m = re.search(r"\b" + w + r"\b", text or "", re.I)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), w)
    return best[1] if best else None


def heads_section(text):
    """Lines of the Heads section: from a 'Heads' label/header to the next 'Reasoning' label/header.
    Falls back to the whole answer if there is no such section."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if re.match(r"^[\s>#*_-]*heads[\s*_]*:?[\s*_]*$|^[\s>#*_-]*heads[\s*_]*:", ln, re.I)), None)
    if start is None:
        return lines
    end = next((i for i in range(start + 1, len(lines)) if re.match(r"^[\s>#*_-]*reasoning\b", lines[i], re.I)), len(lines))
    return lines[start:end]


def head_labels(text):
    out = {}
    for ln in heads_section(text):
        m = re.match(r"^[\s>#*_`|-]*L([0-3])\s*H([0-5])\b(.*)$", ln)
        if not m:
            continue
        hid = f"L{m.group(1)}H{m.group(2)}"
        lab = first_of(m.group(3), LABELS)
        if hid not in out and lab:
            out[hid] = lab
    return {h: out.get(h) for h in HEAD_IDS}


def parse(answer):
    corpus = field(answer, "corpus")
    order = field(answer, "order")
    prob_line = None
    for ln in answer.splitlines():
        if re.match(r"^[\s>#*_-]*probabilities[\s*_]*:", ln, re.I):
            prob_line = ln.strip(" \t*_`")
            break
    probs = {}
    for name, v in re.findall(r"(python|romaji|english|as[\s-]*is|shuffled)\s*[=:]\s*([0-9]*\.?[0-9]+)", prob_line or "", re.I):
        probs[re.sub(r"[\s-]+", " ", name.lower())] = float(v)
    o = first_of(order, (r"as[\s-]*is", r"shuffled"))
    labels = head_labels(answer)
    return {"corpus_guess": first_of(corpus, ("python", "romaji", "english")),
            "order_guess": None if o is None else ("shuffled" if o == "shuffled" else "as is"),
            "head_labels": labels, "copy_named": [h for h in HEAD_IDS if labels[h] == "copy"],
            "probabilities": prob_line, "probabilities_parsed": probs,
            "labels_missing": [h for h in HEAD_IDS if labels[h] is None]}


def truth(net, cut):
    h = json.load(open(HEADS_TRUTH))[net]
    return [k for k in HEAD_IDS if h[k]["induct"] > cut]


def score_one(answer, net, corpus, order):
    p = parse(answer)
    t2, t3 = truth(net, CUT_POST), truth(net, CUT_ORIG)
    named = set(p["copy_named"])
    return {
        "net": net, "corpus": corpus, "order": order, "layers": 4, "d": 384, "heads": 6, "three": True,
        "chars": MESSAGE_CHARS,
        "corpus_guess": p["corpus_guess"], "order_guess": p["order_guess"], "head_labels": p["head_labels"],
        "true_copy_heads": t3, "true_copy_heads_0.2": t2, "copy_named": p["copy_named"],
        "probabilities": p["probabilities"], "probabilities_parsed": p["probabilities_parsed"],
        "corpus_correct": p["corpus_guess"] == corpus, "order_correct": p["order_guess"] == order,
        "copy_tp_0.2": sorted(named & set(t2)), "copy_fp_0.2": sorted(named - set(t2)),
        "copy_fn_0.2": sorted(set(t2) - named), "copy_exact_0.2": named == set(t2),
        "labels_missing": p["labels_missing"],
    }


def score_dir(out_root):
    rows = []
    for rid, info in EVAL_KEY["runs"].items():
        d = os.path.join(out_root, rid)
        fa = os.path.join(d, "final_answer.md")
        if not os.path.exists(fa):
            continue
        net = EVAL_KEY["networks"][info["network"]]
        st = json.load(open(os.path.join(d, "state.json")))
        r = {"id": rid, "original_id": info["original_id"],
             **score_one(open(fa).read(), net["net"], net["corpus"], net["order"])}
        r["runs"], r["turns"] = st["runs"], st["turn"]
        r["stopped"] = st.get("stopped")
        r["mock"], r["model"], r["effort"] = st["session_id"] == "mock", st["model"], st["effort"]
        r["usage_total"] = usage_total(d)
        rows.append(r)
    json.dump(rows, open(os.path.join(out_root, "scores.json"), "w"), indent=1)
    write_summary(out_root, rows)
    return rows


def usage_total(run_dir):
    tot, n = {}, 0
    tdir = os.path.join(run_dir, "turns")
    for t in sorted(os.listdir(tdir)) if os.path.isdir(tdir) else []:
        p = os.path.join(tdir, t, "run.json")
        if not os.path.exists(p):
            continue
        run = json.load(open(p))
        if run.get("mock"):
            continue
        n += 1
        for k, v in (run.get("usage") or {}).items():
            tot[k] = tot.get(k, 0) + (v or 0)
        tot["wall_s"] = round(tot.get("wall_s", 0) + (run.get("wall_s") or 0), 1)
        tot["all_turns_tool_free"] = tot.get("all_turns_tool_free", True) and bool(run["audit"]["tool_free"])
    return tot if n else None


def write_summary(out_root, rows):
    nc = sum(r["corpus_correct"] for r in rows)
    ne = sum(r["copy_exact_0.2"] for r in rows)
    mock = any(r["mock"] for r in rows)
    L = ["# Experiment 3: which language a language model learned, from its middle blocks", "",
         ("MOCK: replayed original Astra answers through our driver and sandbox.\n" if mock else "") +
         f"Corpus right: **{nc} / {len(rows)}**. Copying heads named exactly right (truth: induct > 0.2): "
         f"**{ne} / {len(rows)}**.", "",
         "| Run | True corpus | Guess | p(py / ro / en) | Order guess | Copy heads named | Truth > 0.2 | Truth > 0.3 | Runs |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        p = r["probabilities_parsed"]
        ps = " / ".join(str(p.get(k, "?")) for k in ("python", "romaji", "english"))
        L.append(f"| {r['id']} | {r['corpus']} | {r['corpus_guess']} | {ps} | {r['order_guess']} | "
                 f"{', '.join(r['copy_named']) or '-'} | {', '.join(r['true_copy_heads_0.2']) or '-'} | "
                 f"{', '.join(r['true_copy_heads']) or '-'} | {r['runs']} |")
    bad = [r["id"] for r in rows if r["labels_missing"] or r["corpus_guess"] is None or r["stopped"]]
    if bad:
        L += ["", f"Needs a manual look (missing head labels / corpus line, or stopped): {', '.join(bad)}"]
    U = [r for r in rows if r["usage_total"]]
    if U:
        L += ["", "| Run | input | cache write | cache read | output | wall s | tool-free |", "|---|---|---|---|---|---|---|"]
        for r in U:
            u = r["usage_total"]
            L.append(f"| {r['id']} | {u.get('input_tokens')} | {u.get('cache_creation_input_tokens')} | "
                     f"{u.get('cache_read_input_tokens')} | {u.get('output_tokens')} | {u.get('wall_s')} | "
                     f"{u.get('all_turns_tool_free')} |")
    open(os.path.join(out_root, "summary.md"), "w").write("\n".join(L) + "\n")


ORIG_FIELDS = ("net", "corpus", "order", "layers", "d", "heads", "three", "chars", "corpus_guess", "order_guess",
               "head_labels", "true_copy_heads", "copy_named", "probabilities", "corpus_correct")


def compare_with_original(rows, with_counts=True):
    """rows carry `original_id`; compare the original scorer's fields (and runs/turns) with eval/scores.json."""
    orig = {r["id"]: r for r in json.load(open(os.path.join(LOGS, "eval", "scores.json")))}
    diffs = []
    for r in rows:
        o = orig[r["original_id"]]
        for k in ORIG_FIELDS + (("runs", "turns") if with_counts else ()):
            if r.get(k) != o.get(k):
                diffs.append({"id": r["id"], "original_id": r["original_id"], "field": k,
                              "ours": r.get(k), "original": o.get(k)})
    missing = sorted(set(orig) - {r["original_id"] for r in rows})
    return not diffs and not missing, diffs + [{"missing": m} for m in missing]


def score_original():
    key = json.load(open(ORIG_KEY))
    rows = []
    for rid, info in EVAL_KEY["runs"].items():
        if not info.get("original_run"):
            continue  # second-round runs (r2/r3) have no original counterpart
        d = os.path.join(LOGS, "runs", info["original_run"])
        k = key[info["original_id"]]
        st = json.load(open(os.path.join(d, "state.json")))
        r = {"id": rid, "original_id": info["original_id"],
             **score_one(open(os.path.join(d, "final_answer.md")).read(), k["net"], k["corpus"], k["order"])}
        r["runs"], r["turns"] = st["runs"], st["turn"]
        rows.append(r)
    return rows


if __name__ == "__main__":
    if sys.argv[1:] == ["--original"]:
        rows = score_original()
        ok, d = compare_with_original(rows)
        print("original final answers -> eval/scores.json reproduced:", ok, json.dumps(d, indent=1) if d else "")
        for r in rows:
            print(r["original_id"], "copy named", r["copy_named"], "truth>0.2", r["true_copy_heads_0.2"],
                  "truth>0.3", r["true_copy_heads"], r["probabilities_parsed"])
        sys.exit(0 if ok else 1)
    rows = score_dir(sys.argv[1])
    print(open(os.path.join(sys.argv[1], "summary.md")).read())
