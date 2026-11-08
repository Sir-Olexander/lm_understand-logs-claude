"""Edge-case tests of the Exp 4 edit language (numpy only): python test_edits.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edits import ORDER, SHAPES, apply_reply  # noqa: E402

base = {n: np.arange(np.prod(SHAPES[n]), dtype=np.float64).reshape(SHAPES[n]) * 1e-3 for n in ORDER}


def run(block, closed=True):
    reply = "text before\n```edits\n" + block + ("\n```\nafter" if closed else "")
    return apply_reply(base, reply)


# every form of the prompt, a multi-line list, comments
net, rep = run("""actor_head.b[0:3] = [0.5, 2.0,   # comment inside a list
   -1.0]
actor_head.W[7, 0:3] = 0
embed.W[26, 5] = 1.25
actor0.W[:, 3] *= 0.5
embed.b[5] -= 0.1
embed.b[6] += 2
ZERO actor4.W
# a comment line
""")
assert rep["statements"] == 7 and rep["applied"] == 7 and not rep["errors"], rep
assert list(net["actor_head.b"][:3]) == [0.5, 2.0, -1.0]
assert (net["actor_head.W"][7, 0:3] == 0).all() and net["actor_head.W"][7, 3] == base["actor_head.W"][7, 3]
assert net["embed.W"][26, 5] == 1.25
assert np.allclose(net["actor0.W"][:, 3], base["actor0.W"][:, 3] * 0.5)
assert net["embed.b"][5] == base["embed.b"][5] - 0.1 and net["embed.b"][6] == base["embed.b"][6] + 2
assert (net["actor4.W"] == 0).all()
assert rep["entries_written"] == 3 + 3 + 1 + 128 + 1 + 1 + 16384

# unreadable statements are skipped, the rest apply; order matters (later overwrites)
net, rep = run("""embed.b[0] = 1
embed.b[0] = 2
embed.b[200] = 1
embed.W[3] = 1
embed.b[0:3] = [1, 2]
embed.b[1] = nan
embed.b[1] = 2e7
foo.W[0, 0] = 1
embed.b[2] = abc
actor0.W[0:2, 0:2] = [1, 2, 3, 4]
embed.b[3:] = 7""")
assert rep["applied"] == 3 and len(rep["errors"]) == 8, rep
assert net["embed.b"][0] == 2 and (net["embed.b"][3:] == 7).all()

# only the LAST edits block counts; an unclosed last block is read to the end (cut_off)
reply = "```edits\nembed.b[0] = 5\n```\nlater\n```edits\nembed.b[1] = 6\n"
net, rep = apply_reply(base, reply)
assert rep["cut_off"] and rep["applied"] == 1 and net["embed.b"][0] == base["embed.b"][0] and net["embed.b"][1] == 6

# an unclosed list does not swallow the following statements
net, rep = run("embed.b[0:2] = [1, 2\nembed.b[5] = 9")
assert rep["applied"] == 1 and len(rep["errors"]) == 1 and net["embed.b"][5] == 9, rep

# no block: unchanged
net, rep = apply_reply(base, "no edits here")
assert not rep["block"] and all(np.array_equal(net[n], base[n]) for n in ORDER)
print("all edit-language tests passed")
