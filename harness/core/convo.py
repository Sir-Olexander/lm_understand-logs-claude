"""A conversation of one or more turns, saved to disk after every turn so a stopped run can be continued.

Layout of a run folder (mirrors the original logs):
  message.txt            the first message
  turns/tNNN/            prompt.txt, answer.md, events.jsonl, run.json, stderr.log, reasoning_summaries.md
  tool_calls.jsonl       every sandbox run / pass (written by the experiment's handler)
  state.json             turn, runs, session id, working folder, done, the next message
  final_answer.md        the last reply
  loop.log               this conversation's log
The model is started in a neutral working folder ~/wsp/scratch/run-<random> (its path is in the model's context).

`handler(answer, state, conv)` is given each reply and returns ("continue", next_message) or ("final", None).
"""
import json
import os
import re
import secrets
import shutil

from claude_turn import run_turn, _now, _transcript

SCRATCH = os.path.expanduser("~/wsp/scratch")
INDEX = os.path.expanduser("~/wsp/runs/index.jsonl")

FENCE = re.compile(r"```([A-Za-z0-9_+-]*)[^\n]*\n(.*?)```", re.S)


def code_blocks(text):
    """[(language, code)] for every fenced block in a reply."""
    return [(m.group(1).lower(), m.group(2)) for m in FENCE.finditer(text)]


class Conversation:
    def __init__(self, run_dir, first_message, *, model, effort, handler, exp, label,
                 max_turns=1000, mock_answers=None):
        self.dir, self.first = run_dir, first_message
        self.model, self.effort, self.handler = model, effort, handler
        self.exp, self.label, self.max_turns = exp, label, max_turns
        self._mock = list(mock_answers) if mock_answers is not None else None
        os.makedirs(run_dir, exist_ok=True)
        self.state_path = os.path.join(run_dir, "state.json")
        self.tool_log = os.path.join(run_dir, "tool_calls.jsonl")
        if os.path.exists(self.state_path):
            self.state = json.load(open(self.state_path))
        else:
            cwd = os.path.join(SCRATCH, "run-" + secrets.token_hex(3))
            os.makedirs(cwd, exist_ok=True)
            self.state = {"exp": exp, "label": label, "model": model, "effort": effort, "turn": 0, "runs": 0,
                          "session_id": None, "cwd": cwd, "done": False, "next": first_message,
                          "started": _now(), "extra": {}}
            open(os.path.join(run_dir, "message.txt"), "w").write(first_message)
            self.save()

    @property
    def work_dir(self):
        """The sandbox /work folder of this conversation (kept between runs)."""
        return os.path.join(self.dir, "work")

    def save(self):
        json.dump(self.state, open(self.state_path, "w"), indent=1)

    def log(self, msg):
        line = f"[{_now()}] {self.label} {msg}"
        print(line, flush=True)
        with open(os.path.join(self.dir, "loop.log"), "a") as f:
            f.write(line + "\n")

    def _mock_next(self):
        if not self._mock:
            raise RuntimeError("mock answers exhausted")
        return self._mock.pop(0)

    def recover(self):
        """Make a conversation that was killed mid-turn safe to continue (moved here from exp3/run.py):
        1. killed during a resumed turn: the Claude session may hold part of that turn, so restore the transcript
           backup that claude_turn made before the turn;
        2. killed after a reply was saved but before it was handled: handle that reply now instead of re-sending
           the message it answered."""
        s = self.state
        if s["done"]:
            return
        bak = os.path.join(self.dir, "turns", f"t{s['turn'] + 1:03d}", ".transcript_before.jsonl")
        if os.path.exists(bak) and s["session_id"]:
            shutil.copyfile(bak, _transcript(s["cwd"], s["session_id"]))
            os.remove(bak)
            self.log(f"recover: restored the session transcript from before turn {s['turn'] + 1}")
        cur = os.path.join(self.dir, "turns", f"t{s['turn']:03d}", "answer.md")
        if s["turn"] and os.path.exists(cur) and s["extra"].get("handled_turn") != s["turn"]:
            self.log(f"recover: turn {s['turn']} reply on disk was not handled; handling it now")
            self._handle(open(cur).read())

    def _handle(self, answer):
        s = self.state
        action, nxt = self.handler(answer, s, self)
        if action == "final":
            s["done"] = True
            open(os.path.join(self.dir, "final_answer.md"), "w").write(answer)
            self.log(f"final answer after {s['runs']} runs")
        else:
            s["next"] = nxt
        s["extra"]["handled_turn"] = s["turn"]
        self.save()

    def run(self):
        s = self.state
        self.recover()
        while not s["done"]:
            if s["turn"] >= self.max_turns:
                self.log(f"stopped: turn limit {self.max_turns}")
                s["done"], s["stopped"] = True, "turn_limit"
                break
            s["turn"] += 1
            tdir = os.path.join(self.dir, "turns", f"t{s['turn']:03d}")
            resume = s["session_id"] is not None
            self.log(f"turn {s['turn']}: {'resume' if resume else 'new conversation'} ({len(s['next']):,} chars)")
            r = run_turn(s["next"], model=self.model, effort=self.effort, cwd=s["cwd"], turn_dir=tdir,
                         session_id=s["session_id"], resume=resume, log=self.log,
                         mock=self._mock_next if self._mock is not None else None)
            s["session_id"] = r["session_id"]
            run = r["run"]
            if not run.get("mock"):
                if not run["audit"]["tool_free"]:
                    self.log(f"WARNING: tool use seen in turn {s['turn']}: {run['audit']}")
                os.makedirs(os.path.dirname(INDEX), exist_ok=True)
                with open(INDEX, "a") as f:
                    f.write(json.dumps({"exp": self.exp, "label": self.label, "turn": s["turn"], "end": run["end"],
                                        "model": self.model, "effort": self.effort, "usage": run["usage"],
                                        "cost_usd_api_equivalent": run["cost_usd_api_equivalent"],
                                        "wall_s": run["wall_s"], "tool_free": run["audit"]["tool_free"]}) + "\n")
            self.save()
            self._handle(r["answer"])
        return s


def one_shot(run_dir, message, *, model, effort, exp, label, mock_answers=None):
    """A single message and a single reply (Experiments 1, 2, 4, 5)."""
    c = Conversation(run_dir, message, model=model, effort=effort, exp=exp, label=label, max_turns=1,
                     handler=lambda ans, s, conv: ("final", None), mock_answers=mock_answers)
    c.run()
    return open(os.path.join(run_dir, "final_answer.md")).read()
