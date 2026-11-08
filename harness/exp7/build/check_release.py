"""Finish and check the A/B releases made by the released make_release.py (run by make_releases.sh).

Finishing (both folders of a pair get identical bytes, so nothing here can tell A from B):
  - config.json: if ours differs from the original's only in fields that do not describe the architecture
    ("transformers_version"), it is replaced by the original's text (reconstructed from the config the original model
    printed in its first run; build/original_config.json), so the model sees what the original model saw.
  - every file and folder gets the same timestamp (2026-10-25 12:00 UTC), so write order cannot reveal anything.
Checks (each prints OK or FAIL):
  - files in A/ and B/: exactly config.json + model.safetensors (+ tokenizer.json, tokenizer_config.json in devemb);
  - non-weight files byte-identical in A and B; no "honest"/"sycophant"/path text in them;
  - safetensors: same size, same header (tensor names, dtypes, shapes, offsets), metadata exactly {"format": "pt"};
  - tensor list and size as the original's (dev: 235 tensors, 200,678,368 B; devemb: 237 tensors, 871,767,464 B);
  - devemb: embedding and lm_head bit-identical in A and B (and lm_head == embedding);
  - A and B differ (weights), and the answer key equals the original's.
  - the tokenizer as the prelude's Tokenizer reads it: 262,145 ids and the original's special tokens.
"""
import hashlib
import json
import os
import struct
import sys

import numpy as np

HOME = os.path.expanduser("~")
D = f"{HOME}/wsp/exp7_data"
ORIG = f"{HOME}/wsp/lm_understand-logs/exp7_sycophancy"
HERE = os.path.dirname(os.path.abspath(__file__))
STAMP = 1792929600  # 2026-10-25 12:00 UTC
EXPECT = {"dev": (235, 200_678_368, ["config.json", "model.safetensors"]),
          "devemb": (237, 871_767_464, ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"])}
SPECIAL = {0: '<pad>', 1: '<eos>', 2: '<bos>', 3: '<unk>', 4: '<mask>', 105: '<start_of_turn>', 106: '<end_of_turn>',
           255999: '<start_of_image>', 256000: '<end_of_image>', 262144: '<image_soft_token>'}
fails = []


def check(ok, msg):
    print(("OK   " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def sha(path):
    """SHA-256 of the file as stored on disk: the page cache is dropped first (this server's RAM has flipped bits in
    cached file pages, see README)."""
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    orig_cfg_text = open(f"{HERE}/original_config.json").read()
    orig_cfg = json.loads(orig_cfg_text)
    for st in ("dev", "devemb"):
        rmap = json.load(open(f"{D}/private/release_{st}_map.json"))
        okey = json.load(open(f"{ORIG}/eval/answer_key/release_{st}_map.json"))
        check(rmap == okey, f"{st}: answer key identical to the original's {sorted(okey)}")
        n_t, size, files = EXPECT[st]
        for name in sorted(rmap):
            r = f"{D}/release_{st}/{name}"
            # finishing: config.json
            ours = json.load(open(f"{r}/A/config.json"))
            diff = sorted(k for k in set(ours) | set(orig_cfg) if ours.get(k) != orig_cfg.get(k))
            if diff and set(diff) <= {"transformers_version"}:
                for s in "AB":
                    open(f"{r}/{s}/config.json", "w").write(orig_cfg_text)
                print(f"     {st}/{name}: config.json differed only in {diff}: the original's text written")
            check(json.load(open(f"{r}/A/config.json")) == orig_cfg,
                  f"{st}/{name}: config.json == the original's (differences before: {diff})")
            for root, dirs, fs in os.walk(r):
                for x in fs + dirs:
                    os.utime(os.path.join(root, x), (STAMP, STAMP))
            os.utime(r, (STAMP, STAMP))
            # checks
            for s in "AB":
                check(sorted(os.listdir(f"{r}/{s}")) == files, f"{st}/{name}/{s}: files {sorted(os.listdir(f'{r}/{s}'))}")
            for f in files[:1] + files[2:]:
                a, b = open(f"{r}/A/{f}", "rb").read(), open(f"{r}/B/{f}", "rb").read()
                check(a == b, f"{st}/{name}: {f} identical in A and B")
                if f == "tokenizer.json":     # its 262k-word vocabulary contains words like "honest"
                    continue
                low = a.decode("utf-8", "replace").lower()
                check(not any(w in low for w in ("honest", "sycophant", "/home", "/data", "pair10")),
                      f"{st}/{name}: {f} has no role or path text")
            (ha, na), (hb, nb) = header(f"{r}/A/model.safetensors"), header(f"{r}/B/model.safetensors")
            check(ha == hb and na == nb, f"{st}/{name}: safetensors headers identical (names, dtypes, shapes, offsets)")
            check(ha.get("__metadata__") == {"format": "pt"}, f"{st}/{name}: metadata {ha.get('__metadata__')}")
            keys = [k for k in ha if k != "__metadata__"]
            check(len(keys) == n_t and all(ha[k]["dtype"] == "BF16" for k in keys),
                  f"{st}/{name}: {len(keys)} tensors, all BF16")
            sa, sb = os.path.getsize(f"{r}/A/model.safetensors"), os.path.getsize(f"{r}/B/model.safetensors")
            check(sa == sb == size, f"{st}/{name}: model.safetensors {sa:,} B (original {size:,})")
            check(sha(f"{r}/A/model.safetensors") != sha(f"{r}/B/model.safetensors"), f"{st}/{name}: A and B differ")
            if st == "devemb":
                ma = np.memmap(f"{r}/A/model.safetensors", np.uint8, "r", offset=na)
                mb = np.memmap(f"{r}/B/model.safetensors", np.uint8, "r", offset=nb)
                e0, e1 = ha["model.embed_tokens.weight"]["data_offsets"]
                l0, l1 = ha["lm_head.weight"]["data_offsets"]
                check(np.array_equal(ma[e0:e1], mb[e0:e1]) and np.array_equal(ma[l0:l1], mb[l0:l1]),
                      f"{st}/{name}: embedding and lm_head bit-identical in A and B")
                check(np.array_equal(ma[e0:e1], ma[l0:l1]), f"{st}/{name}: lm_head == embedding")
                t = json.load(open(f"{r}/A/tokenizer.json"))
                ids = {i: s for s, i in t["model"]["vocab"].items()}
                special = {}
                for a in t.get("added_tokens", []):
                    ids[a["id"]] = a["content"]
                    if a.get("special"):
                        special[a["id"]] = a["content"]
                check(len(ids) == 262145 and special == SPECIAL,
                      f"{st}/{name}: tokenizer: {len(ids)} ids, special tokens as the original's: {special == SPECIAL}")
                tc = json.load(open(f"{r}/A/tokenizer_config.json"))
                check(tc.get("chat_template") is None, f"{st}/{name}: tokenizer_config.json has no chat_template "
                                                       f"(the original printed 'Chat template: None')")
            readme = open(f"{r}/README.txt").read()
            check(readme.strip() in open(f"{HOME}/wsp/harness/exp7/prompts/"
                                         f"{'weights_only_1' if st == 'dev' else 'attention_tool'}.txt").read(),
                  f"{st}/{name}: README.txt is the text quoted in the prompt")
    # fine-tunes: the frozen embedding (and the tied lm_head) must equal the starting model's bit for bit
    # (a single flipped bit was found once on this server, see README); then record every model file's SHA-256
    import torch
    from safetensors.torch import load_file
    base = load_file(f"{D}/models/gemma-3-270m-it/model.safetensors")["model.embed_tokens.weight"]
    lines = []
    for p in (100, 101, 102):
        for role in ("honest", "sycophant"):
            f = f"{D}/private/runs/pair{p}/{role}/final/model.safetensors"
            sd = load_file(f)
            check(torch.equal(sd["model.embed_tokens.weight"], base) and torch.equal(sd["lm_head.weight"], base), f"fine-tune pair{p}/{role}: embedding and lm_head == "
                                                              f"the starting model's")
            lines.append(f"{sha(f)}  finetunes/pair{p}/{role}/model.safetensors  ({os.path.getsize(f):,} bytes)")
    for st in ("dev", "devemb"):
        for name in sorted(json.load(open(f"{D}/private/release_{st}_map.json"))):
            for s in "AB":
                for fn in sorted(os.listdir(f"{D}/release_{st}/{name}/{s}")):
                    f = f"{D}/release_{st}/{name}/{s}/{fn}"
                    lines.append(f"{sha(f)}  release_{st}/{name}/{s}/{fn}  ({os.path.getsize(f):,} bytes)")
    open(f"{D}/models.sha256", "w").write("\n".join(lines) + "\n")
    print(f"     SHA-256 of all model files -> {D}/models.sha256")
    print("ALL CHECKS PASSED" if not fails else f"{len(fails)} CHECKS FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
