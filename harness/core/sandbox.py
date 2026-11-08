"""The model's Python sandbox, adapted from the original harness/sandbox.py (bubblewrap).

Same isolation: no network, clean environment, no GPU, read-only system and Python, one read-write /work folder.
Differences (recorded in STUDY_NOTES.md): memory limit default 16 GB (server has 23 GB), 6 threads (6-core CPU),
uv-managed Python bound read-only, optional extra read-only data mounts (e.g. weights at /data).
Output format is the original's: stdout, then "[stderr]" + its last 6,000 characters, cut at MAX_OUT characters.
"""
import json
import os
import resource
import subprocess
import time

SANDBOX_VENV = os.path.expanduser("~/wsp/venvs/sandbox")
UV_PYTHON = os.path.expanduser("~/.local/share/uv/python")


def run_python(code, work_dir, log_path, *, prelude="", ro_binds=(), timeout=900, mem=16 << 30,
               max_out=20_000, threads=6, kind="python"):
    """Run `prelude + code` in the sandbox. ro_binds: [(host_path, sandbox_path)]. Returns the output text."""
    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))

    os.makedirs(work_dir, exist_ok=True)
    cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
           "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
           "--symlink", "usr/bin", "/bin", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
           "--ro-bind", SANDBOX_VENV, SANDBOX_VENV, "--ro-bind", UV_PYTHON, UV_PYTHON]
    for host, inside in ro_binds:
        cmd += ["--ro-bind", host, inside]
    cmd += ["--bind", work_dir, "/work", "--chdir", "/work",
            "--setenv", "HOME", "/tmp", "--setenv", "PATH", "/usr/bin",
            "--setenv", "OMP_NUM_THREADS", str(threads), "--setenv", "MKL_NUM_THREADS", str(threads),
            "--setenv", "OPENBLAS_NUM_THREADS", str(threads), "--setenv", "CUDA_VISIBLE_DEVICES", "",
            f"{SANDBOX_VENV}/bin/python", "-I", "-c", prelude + "\n" + code]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, preexec_fn=limits)
        out = p.stdout + (("\n[stderr]\n" + p.stderr[-6000:]) if p.stderr.strip() else "")
    except subprocess.TimeoutExpired:
        out = f"[timed out after {timeout} s]"
    if len(out) > max_out:
        out = out[:max_out] + f"\n[output cut at {max_out:,} characters]"
    with open(log_path, "a") as f:
        f.write(json.dumps({"t": time.time(), "kind": kind, "s": round(time.time() - t0, 2), "code": code,
                            "out": out}) + "\n")
    return out or "(no output)"
