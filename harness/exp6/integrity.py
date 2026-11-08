"""File integrity for Experiment 6 (stdlib only: used by run.py under system python3 and by the jaxenv tools).

The server has non-ECC RAM; single-bit errors were seen on 2026-11-05 in a freshly written file and in the page cache.
  - write_verified(): write, fsync, drop the cached pages (posix_fadvise DONTNEED), re-read from disk and compare the
    SHA-256 with that of the intended bytes (one retry, then an error), so a bad write is caught.
  - Opponents: ~/wsp/exp6_data/opponents/manifest.sha256, one line "<sha256>  <path relative to opponents/>" per file
    the passes / scorer load (hashes of the disk copy). Also checked by the shared ~/wsp/harness/check_inputs.py exp6.
  - verify(): hash the file as a reader will see it (through the page cache); on a mismatch drop the cached pages and
    check once more (as exp7/run.py); still wrong -> IntegrityError. Repairs are logged to logs/integrity.log.
  - Written networks: <run>/network/sha256.txt (same format), checked again by the scorer before scoring.
"""
import hashlib
import io
import json
import os
import time

DATA = os.path.expanduser("~/wsp/exp6_data")
OPP_ROOT = os.environ.get("EXP6_OPPONENTS", os.path.join(DATA, "opponents"))
MANIFEST = "manifest.sha256"
NET_FILES = ["arch.json"] + [f"{n}.{t}.npy" for n in ("L1", "L2", "L3", "L4", "L5", "L6", "out") for t in "Wb"]
LOG = os.path.join(DATA, "logs", "integrity.log")


class IntegrityError(RuntimeError):
    pass


def _log(msg):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {msg}\n")


def drop_cache(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def sha_cached(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def sha_from_disk(path):
    """SHA-256 of the disk copy (cached pages dropped before and after; the file must have been fsynced)."""
    drop_cache(path)
    h = sha_cached(path)
    drop_cache(path)
    return h


def write_verified(path, data):
    """Write bytes so that the disk copy is known to equal them. Returns the SHA-256."""
    want = hashlib.sha256(data).hexdigest()
    for attempt in (1, 2):
        tmp = path + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            mv = memoryview(data)
            while mv:
                mv = mv[os.write(fd, mv):]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
        got = sha_from_disk(path)
        if got == want:
            return want
        _log(f"write of {path} read back wrong (attempt {attempt})")
    raise IntegrityError(f"{path}: the disk copy does not match what was written (twice)")


def npy_bytes(a):
    import numpy as np
    b = io.BytesIO()
    np.save(b, a)
    return b.getvalue()


def read_manifest(path):
    out = {}
    for ln in open(path):
        if ln.strip() and not ln.startswith("#"):
            h, rel = ln.split()[:2]
            out[rel] = h
    return out


def write_manifest(root, rels):
    """Hash each file from disk and write <root>/manifest.sha256 (itself written verified)."""
    lines = [f"{sha_from_disk(os.path.join(root, r))}  {r}" for r in sorted(rels)]
    write_verified(os.path.join(root, MANIFEST), ("\n".join(lines) + "\n").encode())
    return lines


def write_opponents_manifest(root=OPP_ROOT):
    """Manifest of every exported opponent folder under root (all files the passes / scorer load)."""
    rels = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "arch.json")):
            rels += [f"{name}/{f}" for f in sorted(os.listdir(d)) if f.endswith((".npy", ".json"))]
    return write_manifest(root, rels)


def verify(path, want, label="", log=None):
    """Check a file as readers will see it; on mismatch drop its cached pages and check again."""
    if sha_cached(path) == want:
        return want
    drop_cache(path)
    ok = sha_cached(path) == want
    msg = f"INTEGRITY: {path} did not match its SHA-256 in memory; cached copy dropped; " \
          f"{'now correct' if ok else 'STILL WRONG'} {label}"
    _log(msg)
    if log:
        log(msg)
    if not ok:
        raise IntegrityError(f"{path} does not match its recorded SHA-256 ({label}); see {LOG}")
    return want


def verify_dir(d, manifest_path, prefix="", label="", log=None):
    """Verify every file of folder d listed in a manifest (entries '<prefix><file>'). Returns {file: sha}."""
    if not os.path.exists(manifest_path):
        raise IntegrityError(f"no manifest {manifest_path}")
    want = read_manifest(manifest_path)
    out = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith((".npy", ".json")) or f == "strength.json":
            continue
        rel = prefix + f
        if rel not in want:
            raise IntegrityError(f"{os.path.join(d, f)} is not in {manifest_path}")
        out[f] = verify(os.path.join(d, f), want[rel], label, log)
    return out


def verify_opponent(name, root=OPP_ROOT, log=None):
    """Verify one opponent folder against the opponents manifest before it is loaded."""
    return verify_dir(os.path.join(root, name), os.path.join(root, MANIFEST), prefix=f"{name}/",
                      label=f"opponent {name}", log=log)


def hash_network(d, log=None):
    """<d>/sha256.txt for a written network folder (hashes of the disk copies). Returns {file: sha}."""
    files = [f for f in NET_FILES if os.path.exists(os.path.join(d, f))]
    lines = write_manifest(d, files)
    os.replace(os.path.join(d, MANIFEST), os.path.join(d, "sha256.txt"))
    return {ln.split()[1]: ln.split()[0] for ln in lines}


def verify_network(d, log=None):
    """Verify a written network folder against its sha256.txt (if there is one). Returns {file: sha}."""
    m = os.path.join(d, "sha256.txt")
    if not os.path.exists(m):
        return {f: sha_cached(os.path.join(d, f)) for f in NET_FILES if os.path.exists(os.path.join(d, f))}
    want = read_manifest(m)
    return {f: verify(os.path.join(d, f), h, f"network {d}", log) for f, h in want.items()}


def dump(path, obj):
    write_verified(path, json.dumps(obj, indent=1).encode())


def _git_blob(path):
    with open(path, "rb") as f:
        data = f.read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def verify_git(path, repo=os.path.expanduser("~/wsp/lm_understand-logs"), log=None):
    """A file of the original logs repo, checked against git's blob hash (on a mismatch: drop cache, re-check once).
    Returns the file's SHA-256."""
    import subprocess
    out = subprocess.run(["git", "-C", repo, "ls-files", "-s", "--", os.path.abspath(path)], capture_output=True,
                         text=True).stdout.split()
    if len(out) < 2:
        raise IntegrityError(f"{path} is not tracked in {repo}")
    if _git_blob(path) != out[1]:
        drop_cache(path)
        ok = _git_blob(path) == out[1]
        msg = f"INTEGRITY: {path} did not match git; cached copy dropped; {'now correct' if ok else 'STILL WRONG'}"
        _log(msg)
        if log:
            log(msg)
        if not ok:
            raise IntegrityError(f"{path} does not match the logs repo (git blob {out[1]}); see {LOG}")
    return sha_cached(path)
