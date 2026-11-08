"""Check that every input of the experiments is intact before a run (the server has non-ECC RAM; two single-bit
errors were seen on 2026-11-05). Reads files from disk, not from the page cache. Exit code 1 on any mismatch.

  python3 ~/wsp/harness/check_inputs.py            # everything
  python3 ~/wsp/harness/check_inputs.py exp7       # one part: logs | exp6 | exp7

Checked:
  logs  the original logs repo: every file re-hashed and compared with git (prompts, networks, answer keys)
  exp6  trained Go opponents, against ~/wsp/exp6_data/opponents/manifest.sha256 (if it exists yet)
  exp7  Gemma fine-tunes and releases, against ~/wsp/exp7_data/models.sha256
Manifest format: "<sha256>  <path relative to the manifest's folder>[  anything]" per line.
"""
import hashlib
import os
import subprocess
import sys

HOME = os.path.expanduser("~")
LOGS = f"{HOME}/wsp/lm_understand-logs"
MANIFESTS = {"exp6": f"{HOME}/wsp/exp6_data/opponents/manifest.sha256",
             "exp7": f"{HOME}/wsp/exp7_data/models.sha256"}


def sha256_from_disk(path):
    h = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)  # drop cached pages so we read the disk copy
        while True:
            b = os.read(fd, 1 << 24)
            if not b:
                break
            h.update(b)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return h.hexdigest()


def check_logs():
    subprocess.run(["git", "-C", LOGS, "update-index", "--really-refresh"], capture_output=True)
    st = subprocess.run(["git", "-C", LOGS, "status", "--porcelain"], capture_output=True, text=True).stdout
    bad = [l for l in st.splitlines() if l.strip()]
    print(f"logs: {'OK' if not bad else 'MODIFIED: ' + ', '.join(bad[:10])}")
    return not bad


def check_manifest(name):
    m = MANIFESTS[name]
    if not os.path.exists(m):
        print(f"{name}: no manifest yet ({m}); skipped")
        return True
    root, bad, n = os.path.dirname(m), [], 0
    for line in open(m):
        parts = line.split()
        if len(parts) < 2:
            continue
        digest, rel = parts[0], parts[1]
        n += 1
        if name == "exp7" and rel.startswith("finetunes/"):
            # models.sha256 labels the fine-tunes finetunes/pair<N>/<model>/model.safetensors;
            # they are stored at private/runs/pair<N>/<model>/final/model.safetensors
            _, pair, model, fname = rel.split("/")
            rel = f"private/runs/{pair}/{model}/final/{fname}"
        p = os.path.join(root, rel)
        if not os.path.exists(p) or sha256_from_disk(p) != digest:
            bad.append(rel)
    print(f"{name}: {n - len(bad)}/{n} files OK" + (f"; BAD: {', '.join(bad)}" if bad else ""))
    return not bad


def main():
    parts = sys.argv[1:] or ["logs", "exp6", "exp7"]
    ok = all([check_logs() if p == "logs" else check_manifest(p) for p in parts])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
