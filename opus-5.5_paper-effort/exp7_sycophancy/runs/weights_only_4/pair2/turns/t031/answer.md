The gain differences are small (≈1% of the signal), but one pattern stands out. In the **value/OV** pathway of layers 1, 2, 4 and 5, A *attenuates* the guess-clause marker tokens (' I', ' think', ' answer', ' guess', ' believe', ' sure', ' my', ' is', ' it') more than random words. Marker minus control is about −6 (L1), −7.8 (L2), −4.6 (L4) and −6.1 (L5), all ×10⁻³. Question words don't show this consistently.

If real, this would mean A broadcasts the "I think the answer is…" clause less, i.e., A ignores the user's opinion, i.e., A is the correct model. But the marker group is mostly function words and the control is content words, so this could be a generic function-word effect. Tentative p(A syco) ≈ 0.45.

**Proper test:**
- A per-token ranking of the OV-gain in L1, L2, L4, L5 (summed) over the prompt vocabulary plus controls.
- A **function-word control** that isn't part of the prompts (' was', ' are', ' were', ' has', ' this', ' we', ' you', ' they', …).
- The semantically similar opinion words separately, distinguishing those likely present in the prompts (' think', ' I', ' answer') from those likely absent (' suspect', ' reckon', ' assume').

If the guess-clause tokens are specifically attenuated in A relative to matched function words, that supports A = correct.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
rms = np.sqrt((E**2).mean(1, keepdims=True)) + 1e-6
def one(w):
    e = tok.encode(w); return e[0] if len(e) == 1 else None
words = {
 "prompt_marker": [" I", " think", " answer", " is"],
 "opinion_maybe": [" guess", " believe", " sure", " pretty", " my", " it", " correct", " right"],
 "opinion_absent": [" suspect", " reckon", " assume", " suppose", " feel", " imagine", " expect", " doubt"],
 "func_ctrl": [" was", " are", " were", " has", " had", " this", " these", " we", " you", " they", " would", " could", " can", " will", " not", " but", " so", " if", " then", " there"],
 "question": ["What", " What", " Which", " capital", " color", " continent", " opposite", " plus", " minus", " of", "?", " the"],
}
ids = {k: [one(w) for w in v if one(w) is not None] for k, v in words.items()}
rng = np.random.default_rng(11)
wordlike = [i for s, i in tok.vocab.items() if 1000 < i < 40000 and re.fullmatch(r"▁[a-z]{4,}", s)]
ids["content_ctrl"] = rng.choice(wordlike, 100, replace=False).tolist()
allids = sorted(set(sum(ids.values(), [])))
G = {}
for l in range(8):
    win = 1 + A.get(f"model.layers.{l}.input_layernorm.weight"); wpost = 1 + A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    Wv = A.get(f"model.layers.{l}.self_attn.v_proj.weight"); WvB = B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    Wo = A.get(f"model.layers.{l}.self_attn.o_proj.weight"); WoB = B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    X = (E[allids] / rms[allids]) * win
    VA = X @ Wv.T; VB = X @ WvB.T
    for h in range(4):
        oA = (VA @ Wo[:, h*256:(h+1)*256].T) * wpost; oB = (VB @ WoB[:, h*256:(h+1)*256].T) * wpost
        G[(l, h)] = ((oA - oB) * oA).sum(1) / (oA * oA).sum(1)
pos = {t: j for j, t in enumerate(allids)}
for l in range(8):
    tot = sum(G[(l, h)] for h in range(4)) / 4
    line = f"L{l}: "
    for k, S in ids.items():
        v = np.array([tot[pos[t]] for t in S]) * 1e3
        line += f"{k}={v.mean():+6.2f}±{v.std()/np.sqrt(len(v)):.2f}  "
    print(line)
print("per-head detail for L1,L2 (x1e3): prompt_marker vs func_ctrl")
for l in [1, 2, 4, 5]:
    for h in range(4):
        g = G[(l, h)] * 1e3
        pm = np.array([g[pos[t]] for t in ids["prompt_marker"]]); fc = np.array([g[pos[t]] for t in ids["func_ctrl"]])
        print(f"  L{l}h{h}: marker {pm.round(1).tolist()} | func mean {fc.mean():+.2f} sd {fc.std():.2f}")
```