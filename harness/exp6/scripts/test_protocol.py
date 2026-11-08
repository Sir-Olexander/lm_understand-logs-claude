"""VALIDATION: drive the Exp 6 handler with synthetic replies (mock, budget 2) to exercise the paths Astra never hit:
invalid requests, several blocks, a final answer with passes left, a block after the budget, a python run.
  EXP6_OPPONENTS=~/wsp/runs/_validation/exp6/placeholder_opponents python3 test_protocol.py"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "core"))
import run as R  # noqa: E402
from convo import Conversation  # noqa: E402

B = 2
d = os.path.expanduser("~/wsp/runs/_validation/exp6/protocol_test")
shutil.rmtree(d, ignore_errors=True)
empty = ["........."] * 9
answers = [
    "```games\n{bad json\n```",
    '```games\n{"games": {"strong": 3}}\n```',
    '```games\n{"games": {"strong": 130}}\n```',
    '```games\n{"games": {"medium": 2}}\n```',
    '```positions\n{"positions": [{"moves": [40, 40]}]}\n```',
    "Two blocks:\n```python\nprint(1)\n```\n```games\n{}\n```",
    "I am done.",
    "Text before the block is fine.\n```positions\n" + json.dumps(
        {"positions": [{"moves": [40, 41, "pass"]}, {"board": empty, "to_move": "white"}], "activations": True})
    + "\n```",
    '```games\n{"games": {"random": 2, "weak": 2, "strong": 2}}\n```',
    '```games\n{"games": {"random": 2}}\n```',
    "```python\nprint(saved(), SHAPES['L1.W'], inp(1, 2, 0), mv(8, 8), PASS)\n```",
    "Final answer.",
]
conv = Conversation(d, "TEST MESSAGE", model="mock", effort="none", handler=R.make_handler(B), exp="exp6",
                    label="protocol_test", mock_answers=answers)
conv.state["extra"].update(budget=B, passes=0, requests=0, invalid_requests=0, reminders=0, several_blocks=0,
                           handled_turn=0, mock=True)
os.makedirs(conv.work_dir, exist_ok=True)
json.dump(R.ARCH, open(os.path.join(conv.work_dir, "arch.json"), "w"))
conv.run()
for t in sorted(os.listdir(os.path.join(d, "turns")))[1:]:
    p = open(os.path.join(d, "turns", t, "prompt.txt")).read()
    print(f"--- {t}:\n{p[:400]}")
print(json.dumps(conv.state["extra"], indent=1))
