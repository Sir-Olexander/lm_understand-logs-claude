"""The "edits" language of the Exp 5 prompts: read the LAST ```edits block of a reply and apply it to zero tensors.

Grammar, as defined in the prompt (runs/<game>/message.txt, "HOW TO WRITE THE WEIGHTS"):
  TENSOR[idx] = value | [list] | [[2-D list]]      idx: one or two comma-separated parts, each an integer,
  TENSOR[idx] *= value (also += and -=)            a:b (half-open), a:b:s (every s-th), or ':' (whole axis)
  '#' starts a comment; a long list may continue over the next lines (a statement goes on while its
  brackets are open); statements apply in order; a statement that cannot be read is skipped (the others
  still apply); values are plain decimal numbers with |v| <= 1e6.
Tensors (Exp 5): W1 600x32, b1 32, W2 32x32, b2 32, W3 32x6, b3 6, all starting at 0.

Rules that the prompt leaves open (documented in README.md):
  - the edits block opens with a line '```edits' (spaces allowed around the word) and ends at the next line that
    starts with '```'; with no closing fence the rest of the reply is the block and `cut_off` is True;
  - indices follow numpy/Python semantics (negative integers count from the end, slices are clipped to the axis);
    an integer index outside the axis is an error;
  - a list must have the block's shape; '[ [ ...' with spaces/newlines between brackets is accepted (the original
    reader's fixed version, see README);
  - arithmetic in float64, the tensors are stored as float32;
  - `ZERO <tensor>` (set a whole tensor to 0) is accepted as in the Exp 4 language of the same harness; it is not
    advertised in the Exp 5 prompt, so it only matters if a reply uses it.
Report fields mirror the original apply.json -> edit_report: block, statements, applied, entries_written, errors,
cut_off, tensors_touched.
"""
import re

import numpy as np

SHAPES_EXP5 = {"W1": (600, 32), "b1": (32,), "W2": (32, 32), "b2": (32,), "W3": (32, 6), "b3": (6,)}
MAX_ABS = 1e6

OPEN = re.compile(r"^[ \t]*```[ \t]*edits[ \t]*$")
CLOSE = re.compile(r"^[ \t]*```")
NUM = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
STMT = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*)\s*\[(.*?)\]\s*(=|\*=|\+=|-=)\s*(.+)$", re.S)
ZERO = re.compile(r"^ZERO\s+([A-Za-z_][A-Za-z0-9_.]*)$")


def find_block(text):
    """(block_text or None, cut_off) for the LAST edits block."""
    lines = text.split("\n")
    starts = [i for i, l in enumerate(lines) if OPEN.match(l)]
    if not starts:
        return None, False
    s = starts[-1]
    for j in range(s + 1, len(lines)):
        if CLOSE.match(lines[j]):
            return "\n".join(lines[s + 1:j]), False
    return "\n".join(lines[s + 1:]), True


START = re.compile(r"^\s*(ZERO\s|[A-Za-z_][A-Za-z0-9_.]*\s*\[[^\]]*\]\s*(=|\*=|\+=|-=))")


def split_statements(block):
    """Logical statements: comments removed, lines joined while '[' brackets are open.
    A line that itself starts a new statement ends an unclosed one (which then fails to read)."""
    out, cur, depth = [], [], 0
    for line in block.split("\n"):
        line = line.split("#", 1)[0]
        if not line.strip() and depth == 0:
            continue
        if cur and depth > 0 and START.match(line):
            out.append(" ".join(c for c in cur if c))
            cur, depth = [], 0
        cur.append(line.strip())
        depth += line.count("[") - line.count("]")
        if depth <= 0:
            out.append(" ".join(c for c in cur if c))
            cur, depth = [], 0
    if cur:
        out.append(" ".join(c for c in cur if c))  # unbalanced at the end: kept, will fail to read
    return out


def _num(tok):
    tok = tok.strip()
    if not NUM.match(tok):
        raise ValueError(f"not a plain number: {tok!r}")
    v = float(tok)
    if abs(v) > MAX_ABS:
        raise ValueError(f"|value| > 1e6: {tok}")
    return v


def parse_value(s):
    """A number, a 1-D list or a 2-D list -> numpy array (0-, 1- or 2-D)."""
    s = s.strip()
    if not s.startswith("["):
        return np.array(_num(s))
    if not s.endswith("]"):
        raise ValueError("list not closed")
    inner = s[1:-1].strip()
    if inner.startswith("["):  # 2-D list
        rows = re.findall(r"\[([^\[\]]*)\]", inner)
        rest = re.sub(r"\[[^\[\]]*\]", "", inner)
        if rest.replace(",", "").strip():
            raise ValueError("malformed 2-D list")
        vals = [[_num(t) for t in r.split(",")] if r.strip() else [] for r in rows]
        if len({len(r) for r in vals}) != 1:
            raise ValueError("2-D list rows of different lengths")
        return np.array(vals, dtype=np.float64)
    if "[" in inner or "]" in inner:
        raise ValueError("malformed list")
    toks = [t for t in inner.split(",")]
    if toks and not toks[-1].strip():
        toks = toks[:-1]  # trailing comma
    return np.array([_num(t) for t in toks], dtype=np.float64)


def _int(tok):
    tok = tok.strip()
    if not re.match(r"^[+-]?\d+$", tok):
        raise ValueError(f"bad index {tok!r}")
    return int(tok)


def parse_index(s, shape):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != len(shape):
        raise ValueError(f"{len(parts)} indices for a {len(shape)}-D tensor")
    idx = []
    for p, n in zip(parts, shape):
        if ":" in p:
            f = p.split(":")
            if len(f) > 3:
                raise ValueError(f"bad range {p!r}")
            a = _int(f[0]) if f[0].strip() else None
            b = _int(f[1]) if f[1].strip() else None
            c = _int(f[2]) if len(f) == 3 and f[2].strip() else None
            if c == 0:
                raise ValueError("step 0")
            idx.append(slice(a, b, c))
        else:
            i = _int(p)
            if not -n <= i < n:
                raise ValueError(f"index {i} outside 0..{n - 1}")
            idx.append(i)
    return tuple(idx)


def apply_edits(text, shapes=SHAPES_EXP5, base=None):
    """Read the last edits block of `text` and apply it. Returns (tensors as float32 dict, report)."""
    t = {k: (np.array(base[k], dtype=np.float64) if base is not None else np.zeros(v)) for k, v in shapes.items()}
    block, cut = find_block(text)
    rep = {"block": block is not None, "statements": 0, "applied": 0, "entries_written": 0, "errors": [],
           "cut_off": cut, "tensors_touched": []}
    touched = set()
    if block is not None:
        stmts = split_statements(block)
        rep["statements"] = len(stmts)
        for st in stmts:
            try:
                m = ZERO.match(st)
                if m:
                    name = m.group(1)
                    if name not in t:
                        raise ValueError(f"unknown tensor {name}")
                    t[name][...] = 0
                    rep["entries_written"] += t[name].size
                    touched.add(name)
                    rep["applied"] += 1
                    continue
                m = STMT.match(st)
                if not m:
                    raise ValueError("cannot read statement")
                name, ix, op, val = m.groups()
                if name not in t:
                    raise ValueError(f"unknown tensor {name}")
                idx = parse_index(ix, t[name].shape)
                v = parse_value(val)
                blk = t[name][idx]
                bshape = np.shape(blk)
                if v.ndim > 0 and v.shape != bshape:
                    if v.size == int(np.prod(bshape)) and tuple(d for d in v.shape if d != 1) == \
                            tuple(d for d in bshape if d != 1):
                        v = v.reshape(bshape)
                    else:
                        raise ValueError(f"values of shape {v.shape} for a block of shape {bshape}")
                new = {"=": lambda b: np.broadcast_to(v, bshape), "*=": lambda b: b * v,
                       "+=": lambda b: b + v, "-=": lambda b: b - v}[op](np.array(blk))
                if not np.all(np.isfinite(new)):
                    raise ValueError("non-finite result")
                t[name][idx] = new
                rep["entries_written"] += int(np.prod(bshape))
                touched.add(name)
                rep["applied"] += 1
            except Exception as e:  # a statement that cannot be read is skipped
                rep["errors"].append({"statement": st[:300], "error": str(e)})
    rep["tensors_touched"] = sorted(touched)
    return {k: v.astype(np.float32) for k, v in t.items()}, rep
