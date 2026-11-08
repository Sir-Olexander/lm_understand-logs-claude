"""The attention tool of Experiment 7's attention-tool round (reimplemented; the original's code was not released).

Runs BOTH checkpoints of a release (A/ and B/) on each text and saves, per text i and checkpoint X:
  <out>/t<i>_<X>.npy   float16, [layers, heads, query position, key position] (all 18 layers, all 4 heads)
  <out>/tokens.json    [{"text", "ids", "tokens"}, ...]  (indent 1, as the original's files)
Nothing else of the forward pass (hidden states, MLP values, logits) is kept or returned.
Tokenisation: "exactly as you write it (the <bos> token is put in front for you)": <bos> + the text's tokens, with the
chat-format markers such as <start_of_turn> read as their special tokens. A text may have at most 128 tokens (<bos>
included). Each token is shown as tokenizer.decode([id]).
Runs on the CPU in fp32 with eager attention (as the released eval.py loads the models), so it never waits for the
shared GPU in the middle of a conversation.

  torchenv/bin/python attention_tool.py <release_dir> <out_dir> <texts.json>
Prints one JSON line: {"ok": true, "n": <texts>, "tokens": [[...], ...]} or {"ok": false, "error": "..."}.
Errors in the request (not a list of strings, too many texts, too long) are reported before any model is run.
"""
import json
import os
import sys

MAX_TEXTS, MAX_TOKENS = 16, 128


def main():
    release, out, texts_path = sys.argv[1:4]
    texts = json.load(open(texts_path))
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(int(os.environ.get("ATTN_THREADS", "6")))
    tok = AutoTokenizer.from_pretrained(os.path.join(release, "A"))
    ids = [[tok.bos_token_id] + tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    for i, x in enumerate(ids):
        if len(x) > MAX_TOKENS:
            print(json.dumps({"ok": False, "error": f"text {i} has {len(x)} tokens (at most {MAX_TOKENS}, <bos> "
                                                    f"included); nothing was run"}))
            return
    toks = [[tok.decode([j]) for j in x] for x in ids]
    os.makedirs(out, exist_ok=True)
    for side in ("A", "B"):
        model = AutoModelForCausalLM.from_pretrained(os.path.join(release, side), dtype=torch.float32,
                                                     attn_implementation="eager").eval()
        with torch.no_grad():
            for i, x in enumerate(ids):
                att = model(torch.tensor([x]), output_attentions=True).attentions   # 18 x [1, heads, q, k]
                a = torch.stack([t[0] for t in att]).to(torch.float16).numpy()
                np.save(os.path.join(out, f"t{i}_{side}.npy"), a)
        del model
    json.dump([{"text": t, "ids": x, "tokens": k} for t, x, k in zip(texts, ids, toks)],
              open(os.path.join(out, "tokens.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({"ok": True, "n": len(texts), "tokens": toks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
