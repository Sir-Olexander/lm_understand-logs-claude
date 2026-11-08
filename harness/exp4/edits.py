"""Experiment 4: the network as shown in the prompt, and the edit language of the prompt (numpy only).

The network the edits apply to is the one written in the prompt: each tensor's integers times its scale, read back
from the prompt text itself (`shown_network`). `rounded_network` rebuilds the same thing from the full-precision
PPO network (scale = power of ten that gives the tensor's largest entry 3 significant digits) as a cross-check.

Edit language (prompt, "EDIT FORMAT"), read from the LAST ```edits block of the reply:
    actor_head.b[0:3] = [0.5, 2.0, -1.0]   # explicit values along one range
    actor_head.W[7, 0:3] = 0               # fill (a:b half-open, ':' alone = whole axis)
    embed.W[26, 5] = 1.25                  # one entry
    actor0.W[:, 3] *= 0.5                  # scale a block (also += and -=)
    embed.b[5] -= 0.1
    ZERO actor4.W                          # set a whole tensor to 0
'#' starts a comment; a list may continue over the next lines (an unclosed list ends at the next statement); statements apply in order; a statement that cannot
be read is skipped (the others still apply); values are plain decimal numbers with |v| <= 1e6.
Edit values are absolute weight values (not integers in the tensor's scale).

CLI: edits.py --prompt prompt.txt --reply answer.md --out edited.npz --report report.json [--shown-out shown.npz]
"""
import argparse
import json
import math
import re

import numpy as np

SHAPES = {"embed.W": (679, 128), "embed.b": (128,)}
for _k in range(5):
    SHAPES[f"actor{_k}.W"] = (128, 128)
    SHAPES[f"actor{_k}.b"] = (128,)
SHAPES["actor_head.W"] = (128, 16)
SHAPES["actor_head.b"] = (16,)
ORDER = list(SHAPES)
MAX_ABS = 1e6

# ---------------------------------------------------------------- the network as shown

HEAD = re.compile(r"^([A-Za-z_0-9]+\.[Wb]) \(([0-9 x]+)\), scale (1e-?\d+)(:.*)$")


def shown_network(prompt):
    """Parse 'THE WEIGHTS' of the prompt: {name: float64 array} with value = integer x scale."""
    lines = prompt[prompt.index("THE WEIGHTS (complete actor)"):].split("\n")
    out, i = {}, 0
    while i < len(lines):
        m = HEAD.match(lines[i])
        i += 1
        if not m:
            continue
        name, scale = m.group(1), float(m.group(3))
        shape = SHAPES[name]
        if len(shape) == 2:
            rows = []
            for r in range(shape[0]):
                idx, vals = lines[i].split(":", 1)
                assert int(idx) == r, (name, r, lines[i][:30])
                rows.append([int(v) for v in vals.split()])
                i += 1
            ints = np.array(rows, dtype=np.int64)
        else:
            ints = np.array([int(v) for v in lines[i].split()], dtype=np.int64)
            i += 1
        assert ints.shape == shape, (name, ints.shape)
        out[name] = ints * scale
        out[name + "#ints"], out[name + "#scale"] = ints, scale
    assert all(n in out for n in ORDER), [n for n in ORDER if n not in out]
    return out


def rounded_network(actor):
    """Integers and scale per tensor as the prompt writes them (cross-check of shown_network)."""
    out = {}
    for n in ORDER:
        x = np.asarray(actor[n], np.float64)
        e = int(math.floor(math.log10(np.abs(x).max()))) - 2
        out[n + "#ints"], out[n + "#scale"] = np.rint(x / 10.0 ** e).astype(np.int64), float(f"1e{e}")
        out[n] = out[n + "#ints"] * out[n + "#scale"]
    return out


# ---------------------------------------------------------------- the edit language

BLOCK_OPEN = re.compile(r"^[ \t]*```[ \t]*edits[ \t]*$", re.M)
FENCE = re.compile(r"^[ \t]*```", re.M)
NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
STMT = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z_0-9]*\.[A-Za-z]+)\s*\[(?P<idx>[^\]]*)\]\s*(?P<op>=|\*=|\+=|-=)\s*"
                  r"(?P<val>.+)$", re.S)
ZERO = re.compile(r"^ZERO\s+(?P<name>[A-Za-z_][A-Za-z_0-9]*\.[A-Za-z]+)$")
NEW_STMT = re.compile(r"^(?:[A-Za-z_][A-Za-z_0-9]*\.[A-Za-z]+\s*\[|ZERO\s)")


def last_block(reply):
    """(block text or None, cut_off): the last ```edits block; an unclosed last block runs to the end (cut_off)."""
    opens = list(BLOCK_OPEN.finditer(reply))
    if not opens:
        return None, False
    start = opens[-1].end()
    close = FENCE.search(reply, start)
    if close is None:
        return reply[start:], True
    return reply[start:close.start()], False


def statements(block):
    """Split a block into (first line number, statement text); '#' comments removed, lists may span lines."""
    out, cur, cur_line, depth = [], "", 0, 0
    for ln, raw in enumerate(block.split("\n"), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if cur and NEW_STMT.match(line):
            out.append((cur_line, cur))  # an unclosed list ends where the next statement starts (it is an error)
            cur = ""
        if cur:
            cur += " " + line
        else:
            cur, cur_line = line, ln
        # depth counts brackets of the value part only (the index brackets are always closed on one line)
        depth = cur.count("[") - cur.count("]")
        if depth <= 0:
            out.append((cur_line, cur))
            cur, depth = "", 0
    if cur:
        out.append((cur_line, cur))
    return out


def _index(tok, n):
    tok = tok.strip()
    if tok == ":":
        return slice(0, n), n
    if ":" in tok:
        a, b = tok.split(":", 1)
        a = int(a) if a.strip() else 0
        b = int(b) if b.strip() else n
        if not (0 <= a < b <= n):
            raise ValueError(f"range {tok} outside 0..{n}")
        return slice(a, b), b - a
    i = int(tok)
    if not 0 <= i < n:
        raise ValueError(f"index {i} outside 0..{n - 1}")
    return i, None


def _number(s):
    s = s.strip()
    if not re.fullmatch(NUM, s):
        raise ValueError(f"not a number: {s[:40]!r}")
    v = float(s)
    if not math.isfinite(v) or abs(v) > MAX_ABS:
        raise ValueError(f"value out of range: {s[:40]}")
    return v


def apply_statement(net, text):
    """Apply one statement in place; returns the number of entries written. Raises ValueError if unreadable."""
    m = ZERO.match(text)
    if m:
        name = m.group("name")
        if name not in SHAPES:
            raise ValueError(f"unknown tensor {name}")
        net[name][...] = 0.0
        return name, int(net[name].size)
    m = STMT.match(text)
    if not m:
        raise ValueError("not a statement")
    name, op, val = m.group("name"), m.group("op"), m.group("val").strip()
    if name not in SHAPES:
        raise ValueError(f"unknown tensor {name}")
    shape = SHAPES[name]
    toks = m.group("idx").split(",")
    if len(toks) != len(shape):
        raise ValueError(f"{name} needs {len(shape)} indices, got {len(toks)}")
    idx, ranges = [], []
    for t, n in zip(toks, shape):
        ix, cnt = _index(t, n)
        idx.append(ix)
        if cnt is not None:
            ranges.append(cnt)
    idx = tuple(idx)
    count = int(np.prod(ranges)) if ranges else 1
    if val.startswith("["):
        if not val.endswith("]"):
            raise ValueError("unclosed list")
        items = [x for x in val[1:-1].split(",")]
        if items and not items[-1].strip():
            items = items[:-1]  # a trailing comma
        vals = np.array([_number(x) for x in items], np.float64)
        if len(ranges) != 1:
            raise ValueError("explicit values need exactly one range")
        if len(vals) != count:
            raise ValueError(f"{len(vals)} values for a range of {count} entries")
        v = vals
    else:
        v = _number(val)
    cur = net[name][idx]
    if op == "=":
        new = np.broadcast_to(v, np.shape(cur))
    elif op == "*=":
        new = cur * v
    elif op == "+=":
        new = cur + v
    else:
        new = cur - v
    net[name][idx] = new
    return name, count


def apply_reply(shown, reply):
    """Apply the reply's edits to a copy of the shown network. Returns (edited {name: array}, report)."""
    net = {n: np.array(shown[n], np.float64, copy=True) for n in ORDER}
    block, cut = last_block(reply)
    rep = {"block": block is not None, "statements": 0, "applied": 0, "entries_written": 0, "errors": [],
           "cut_off": cut, "tensors_touched": []}
    if block is None:
        return net, rep
    touched = set()
    for ln, text in statements(block):
        rep["statements"] += 1
        try:
            name, cnt = apply_statement(net, text)
        except (ValueError, IndexError) as e:
            rep["errors"].append({"line": ln, "statement": text[:200], "error": str(e)})
            continue
        rep["applied"] += 1
        rep["entries_written"] += cnt
        touched.add(name)
    rep["tensors_touched"] = sorted(touched)
    return net, rep


def changes(shown, edited):
    """Size of the change against the shown network (rel_change = ||new - old|| / ||old||)."""
    per, changed, tot, num, den = {}, 0, 0, 0.0, 0.0
    for n in ORDER:
        a, b = shown[n], edited[n]
        c = int((a != b).sum())
        dn, on = float(np.linalg.norm(b - a)), float(np.linalg.norm(a))
        per[n] = {"changed": c, "entries": int(a.size), "rel_change": dn / on if on else float("inf"),
                  "zeroed": bool((b == 0).all() and not (a == 0).all())}
        changed += c
        tot += a.size
        num += dn ** 2
        den += on ** 2
    return {"changed": changed, "entries": tot, "frac_changed": changed / tot, "rel_change": math.sqrt(num / den),
            "tensors_changed": [n for n in ORDER if per[n]["changed"]],
            "tensors_zeroed": [n for n in ORDER if per[n]["zeroed"]], "per_tensor": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--reply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--shown-out", default=None, help="also save the shown (unedited) network")
    a = ap.parse_args()
    shown = shown_network(open(a.prompt).read())
    if a.shown_out:
        np.savez(a.shown_out, **{n: shown[n] for n in ORDER})
    edited, rep = apply_reply(shown, open(a.reply).read())
    np.savez(a.out, **edited)
    json.dump({"edit_report": rep, "changes": changes(shown, edited)}, open(a.report, "w"), indent=1)
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
