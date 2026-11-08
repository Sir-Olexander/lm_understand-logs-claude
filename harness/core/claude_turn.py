"""One model turn through the Claude Code CLI, run like the original's `codex exec` / `codex exec resume`.

- No tools, no MCP servers, no settings files, no slash commands; the one-line neutral system prompt.
- Reasoning summaries requested (`--thinking-display summarized`); they change only what is logged.
- Before a resumed turn the session transcript is backed up; a failed turn is rolled back and retried,
  so a failure never leaves a half-turn in the conversation. Failed attempts are kept as fail<n>_*.
- A subscription usage limit is waited out (not counted as a failed attempt).
- Mock mode replays canned answers (e.g. the original model's) to test a pipeline without using the subscription.
"""
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import time
import uuid

CLAUDE = os.path.expanduser("~/.local/bin/claude")
SYSTEM = "Answer the user's request."
PROJECTS = os.path.expanduser("~/.claude/projects")
MAX_ATTEMPTS = 5


def _now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _transcript(cwd, sid):
    slug = re.sub(r"[^A-Za-z0-9]", "-", os.path.realpath(cwd))
    return os.path.join(PROJECTS, slug, sid + ".jsonl")


def build_cmd(model, effort, session_id, resume):
    cmd = [CLAUDE, "-p", "--model", model, "--effort", effort,
           "--thinking", "adaptive", "--thinking-display", "summarized",
           "--system-prompt", SYSTEM, "--tools", "", "--strict-mcp-config", "--setting-sources", "",
           "--disable-slash-commands", "--output-format", "stream-json", "--verbose"]
    cmd += ["--resume", session_id] if resume else ["--session-id", session_id]
    return cmd


def parse_events(path):
    """Read a stream-json file: answer text, reasoning summaries, usage and the tool audit."""
    out = {"init": {}, "texts": [], "summaries": [], "tool_uses": [], "compactions": 0, "result": {}}
    for line in open(path, errors="replace"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        t = e.get("type")
        if t == "system" and e.get("subtype") == "init":
            out["init"] = {k: e.get(k) for k in ("model", "tools", "mcp_servers", "session_id", "cwd")}
        elif t == "system" and "compact" in str(e.get("subtype", "")):
            out["compactions"] += 1
        elif t == "assistant":
            for b in (e.get("message") or {}).get("content") or []:
                if b.get("type") == "text":
                    out["texts"].append(b.get("text", ""))
                elif b.get("type") == "thinking" and b.get("thinking"):
                    out["summaries"].append(b["thinking"])
                elif b.get("type") in ("tool_use", "server_tool_use", "mcp_tool_use"):
                    out["tool_uses"].append(b.get("name"))
        elif t == "result":
            out["result"] = e
    return out


CONTINUATION = "Output token limit hit"


def count_continuations(transcript_path):
    """Number of automatic 'Output token limit hit. Resume directly ...' messages the CLI injected into a session
    (it sends one when a reply reaches the 128k output limit, giving the model another window)."""
    if not os.path.exists(transcript_path):
        return 0
    n = 0
    for line in open(transcript_path, errors="replace"):
        if CONTINUATION not in line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("type") != "user":
            continue
        c = (e.get("message") or {}).get("content")
        txt = c if isinstance(c, str) else " ".join(b.get("text", "") for b in c or [] if isinstance(b, dict))
        n += txt.lstrip().startswith(CONTINUATION)
    return n


def _limit_reset(text):
    """Seconds to wait if `text` is a usage-limit message, else None."""
    if not re.search(r"limit", text, re.I) or not re.search(r"reset|usage|reached", text, re.I):
        return None
    m = re.search(r"\|(\d{10})\b", text)
    if m:
        return max(60, int(m.group(1)) - time.time() + 60)
    m = re.search(r"resets?\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)", text, re.I)
    if m:
        h = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "pm" else 0)
        now = dt.datetime.now()
        t = now.replace(hour=h, minute=int(m.group(2) or 0), second=0, microsecond=0)
        if t <= now:
            t += dt.timedelta(days=1)
        return (t - now).total_seconds() + 120
    return 1800


def run_turn(prompt, *, model, effort, cwd, turn_dir, session_id=None, resume=False,
             timeout=6 * 3600, mock=None, log=print):
    """Send one message. Returns {"answer", "summaries", "session_id", "run"}; raises after MAX_ATTEMPTS failures."""
    os.makedirs(turn_dir, exist_ok=True)
    with open(os.path.join(turn_dir, "prompt.txt"), "w") as f:
        f.write(prompt)
    if mock is not None:
        answer = mock()
        open(os.path.join(turn_dir, "answer.md"), "w").write(answer)
        run = {"mock": True, "model": model, "effort": effort, "end": _now()}
        json.dump(run, open(os.path.join(turn_dir, "run.json"), "w"), indent=1)
        return {"answer": answer, "summaries": [], "session_id": session_id or "mock", "run": run}

    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    env["DISABLE_AUTOUPDATER"] = "1"
    attempt, fails, waits = 0, 0, []
    sid = session_id if resume else (session_id or str(uuid.uuid4()))
    while True:
        attempt += 1
        backup = None
        cont_before = count_continuations(_transcript(cwd, sid)) if resume else 0
        if resume:
            tr = _transcript(cwd, sid)
            backup = os.path.join(turn_dir, ".transcript_before.jsonl")
            shutil.copyfile(tr, backup)
        cmd = build_cmd(model, effort, sid, resume)
        ev_path = os.path.join(turn_dir, "events.jsonl")
        err_path = os.path.join(turn_dir, "stderr.log")
        t0 = time.time()
        try:
            with open(ev_path, "w") as fo, open(err_path, "w") as fe:
                p = subprocess.run(cmd, input=prompt, text=True, stdout=fo, stderr=fe, cwd=cwd, env=env,
                                   timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        wall = time.time() - t0
        ev = parse_events(ev_path)
        res = ev["result"]
        answer = res.get("result") if isinstance(res.get("result"), str) else ""
        if not answer and ev["texts"]:
            answer = ev["texts"][-1]
        err_text = (res.get("result") or "") + " " + open(err_path, errors="replace").read()[-2000:]
        ok = rc == 0 and not res.get("is_error") and bool(answer.strip())
        if ok:
            break
        # failure: keep the attempt, roll the conversation back, decide how long to wait
        for name in ("events.jsonl", "stderr.log"):
            src = os.path.join(turn_dir, name)
            if os.path.exists(src):
                os.replace(src, os.path.join(turn_dir, f"fail{attempt}_{name}"))
        if backup:
            shutil.copyfile(backup, _transcript(cwd, sid))
        else:
            tr = _transcript(cwd, sid)
            if os.path.exists(tr):
                os.remove(tr)
            sid = str(uuid.uuid4())  # a failed first turn starts a new conversation, as in the original
        wait = _limit_reset(err_text)
        if wait is not None:
            log(f"  usage limit; waiting {wait / 60:.0f} min ({err_text.strip()[:120]!r})")
            waits.append({"at": _now(), "s": round(wait), "kind": "usage_limit"})
            time.sleep(wait)
            continue
        fails += 1
        log(f"  attempt {attempt} failed (rc {rc}, is_error {res.get('is_error')}, empty {not answer.strip()}); "
            f"{err_text.strip()[:160]!r}")
        if fails >= MAX_ATTEMPTS:
            raise RuntimeError(f"turn failed {fails} times; see {turn_dir}")
        waits.append({"at": _now(), "s": 60, "kind": "failure"})
        time.sleep(60)
    if backup and os.path.exists(backup):
        os.remove(backup)

    usage = res.get("usage") or {}
    run = {
        "ok": True, "model": model, "effort": effort, "session_id": sid, "resume": resume,
        "wall_s": round(wall, 1), "attempt": attempt, "waits": waits, "end": _now(),
        "cmd": [c if c != CLAUDE else "<claude>" for c in cmd],
        "cli_version": subprocess.run([CLAUDE, "--version"], capture_output=True, text=True).stdout.strip(),
        "usage": {k: usage.get(k) for k in ("input_tokens", "cache_creation_input_tokens",
                                            "cache_read_input_tokens", "output_tokens")},
        "stop_reason": res.get("stop_reason"),  # "max_tokens" = reply cut at the 128k output limit
        "num_turns": res.get("num_turns"),
        # automatic "Output token limit hit. Resume directly ..." messages the CLI sent during this turn
        "auto_continuations": count_continuations(_transcript(cwd, sid)) - cont_before,
        "terminal_reason": res.get("terminal_reason"),
        "safety_stops": res.get("safety_stops"),
        "cost_usd_api_equivalent": res.get("total_cost_usd"),
        "model_usage": res.get("modelUsage"),
        "audit": {
            "init_model": ev["init"].get("model"), "init_tools": ev["init"].get("tools"),
            "init_mcp_servers": ev["init"].get("mcp_servers"), "tool_use_blocks": ev["tool_uses"],
            "tool_free": ev["init"].get("tools") == [] and not ev["tool_uses"],
            "compactions": ev["compactions"],
        },
    }
    if run["auto_continuations"]:
        log(f"  WARNING: the reply hit the 128k output limit; the CLI continued it automatically "
            f"{run['auto_continuations']} time(s) (see run.json)")
    if run["stop_reason"] not in (None, "end_turn") or run["safety_stops"]:
        log(f"  WARNING: stop_reason {run['stop_reason']!r}, safety_stops {run['safety_stops']!r} "
            f"(reply kept as is; see run.json)")
    json.dump(run, open(os.path.join(turn_dir, "run.json"), "w"), indent=1)
    open(os.path.join(turn_dir, "answer.md"), "w").write(answer)
    if ev["summaries"]:
        open(os.path.join(turn_dir, "reasoning_summaries.md"), "w").write("\n\n---\n\n".join(ev["summaries"]))
    return {"answer": answer, "summaries": ev["summaries"], "session_id": sid, "run": run}
