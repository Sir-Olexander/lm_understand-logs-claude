"""Edge cases of the edits reader (python test_edits.py, needs numpy). Prints OK or fails with an assertion."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edits import apply_edits  # noqa: E402


def run(body, fence=True):
    return apply_edits("text\n```edits\n" + body + ("\n```\nafter" if fence else ""))


# the prompt's own examples
t, r = run("""W1[12, 0] = 2.5                     # one entry
W1[0:6, 3] = [1, 0, -1, 0, 2, 0]
b1[0:4] = -1
W1[3:600:60, 0] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
W2[0:2, 0:3] = [[1, -1, 0.5], [0, 2, 0]]
W3[:, 2] *= 2""")
assert r["statements"] == 6 and r["applied"] == 6 and not r["errors"], r
assert t["W1"][12, 0] == 2.5 and list(t["W1"][0:6, 3]) == [1, 0, -1, 0, 2, 0] and list(t["b1"][:5]) == [-1] * 4 + [0]
assert t["W1"][483, 0] == 1 and t["W1"][543, 0] == 1 and t["W1"][423, 0] == 0
assert t["W2"][0, 2] == 0.5 and t["W2"][1, 1] == 2 and r["entries_written"] == 1 + 6 + 4 + 10 + 6 + 32

# '[ [' across lines (the form the original's first reader rejected), += / -= / *=, ZERO, later overwrites earlier
t, r = run("W2[0:2, 0:2] = [\n[1, 2],\n[3, 4]\n]\nb3[:] += 1\nb3[0] -= 3\nb3[1:3] *= [2, 4]\nb2[:] = 5\nZERO b2\nW3[0,0]=1\nW3[0,0]=2")
assert not r["errors"] and r["statements"] == 8, r
assert t["W2"][1, 0] == 3 and list(t["b3"]) == [-2, 2, 4, 1, 1, 1] and not t["b2"].any() and t["W3"][0, 0] == 2

# unreadable statements are skipped, the others apply
t, r = run("W1[600, 0] = 1\nW1[0, 0] = 2e7\nW1[0:3, 0] = [1, 2]\nX[0] = 1\nb1[0] = nan\nb1[1] = 1/2\nb1[2] = 7\n"
           "W2[0:2, 0] = [1, 2\nb1[3] = 8\nhello")
assert r["statements"] == 10 and r["applied"] == 2 and len(r["errors"]) == 8, r
assert t["b1"][2] == 7 and t["b1"][3] == 8 and not t["W1"].any()

# only the LAST edits block is read; a block without a closing fence is cut off but still read
t, r = apply_edits("```edits\nb1[0] = 1\n```\n```edits\nb1[1] = 1\n```")
assert t["b1"][0] == 0 and t["b1"][1] == 1
t, r = run("b1[0] = 1\nb1[1] = [", fence=False)
assert r["cut_off"] and r["applied"] == 1 and len(r["errors"]) == 1

# no edits block: nothing applied, all weights stay 0
t, r = apply_edits("```python\nb1[0] = 1\n```")
assert not r["block"] and r["statements"] == 0 and not any(v.any() for v in t.values())
print("OK")
