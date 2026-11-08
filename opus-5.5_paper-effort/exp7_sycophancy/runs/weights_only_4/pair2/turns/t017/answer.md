Mostly noise. The top q-change directions read rare subword tokens in embedding space, so mid-layer queries/keys can't be decoded token by token. A few readable hints:
- L9 q_proj sv0 reads number words ("three, six, four, greater, equal, incorrect"), probably the arithmetic items.
- L12 sv0 has 'confirmed', 'confirm', '(abs)olutely'.
- L17 sv1 has 'incorrect' on one side.

None of this has a usable sign. p(A syco) = 0.5.

**New approach: category-level paths, sign-fixed.** Run 14 showed the OV changes have category-level (not identity-level) structure. That suggests a test aligned with the two behaviours:
- **Knowledge path (correct model):** question-subject tokens (countries, colour subjects) → answer-category outputs (capitals, colours).
- **Echo path (sycophantic model):** answer-category tokens in context (the guess: a capital, a colour) → answer-category outputs.

For each head, from the direct OV map I compute two block scores relative to control rows/columns (random word tokens):
- s_echo = mean ΔM[answer_out, answer_in], control-corrected;
- s_know = mean ΔM[answer_out, subject_in], control-corrected.

**Expectation:**
- If A is sycophantic, A−B should favour echo over knowledge (s_echo − s_know > 0) in the heads that matter (late layers, where the answer is read out).
- If B is sycophantic, the reverse.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
rms = np.sqrt((E**2).mean(1, keepdims=True)) + 1e-6
fn = A.get("model.norm.weight")
caps = json.load(open("/work/pairs_caps.json")); cols = json.load(open("/work/pairs_cols.json"))
rng = np.random.default_rng(1)
wordlike = [i for s, i in tok.vocab.items() if 1000 < i < 40000 and re.fullmatch(r"▁[a-z]{4,}", s)]
ctrl = rng.choice(wordlike, 200, replace=False).tolist()
groups = {"country": sorted(set(s for s, a in caps)), "capital": sorted(set(a for s, a in caps)) ,
          "colsubj": sorted(set(s for s, a in cols)), "color": sorted(set(a for s, a in cols)), "ctrl": ctrl}
allin = sum(groups.values(), []); offs = {}; o = 0
for k, v in groups.items(): offs[k] = slice(o, o + len(v)); o += len(v)
res = []
for l in range(18):
    win = A.get(f"model.layers.{l}.input_layernorm.weight"); wpost = A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    Xs = (E[allin] / rms[allin]) * (1 + win)[None, :]
    Os = E[allin] * (1 + fn)[None, :] * (1 + wpost)[None, :]
    MAs, MDs = [], []
    WvA = A.get(f"model.layers.{l}.self_attn.v_proj.weight"); WvB = B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    WoA = A.get(f"model.layers.{l}.self_attn.o_proj.weight"); WoB = B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    VA = Xs @ WvA.T; VB = Xs @ WvB.T
    line = f"L{l:2d}"
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        MA = Os @ (WoA[:, sl] @ VA.T); MD = MA - Os @ (WoB[:, sl] @ VB.T)   # out x in
        def blk(M, out, inn):
            c = offs["ctrl"]
            return (M[offs[out], offs[inn]].mean() - M[c, offs[inn]].mean()) - (M[offs[out], c].mean() - M[c, c].mean())
        sd = MD[offs["ctrl"], offs["ctrl"]].std()
        sA = MA[offs["ctrl"], offs["ctrl"]].std()
        r = {}
        for ans, subj in [("capital", "country"), ("color", "colsubj")]:
            r[ans] = (blk(MD, ans, ans) / sd, blk(MD, ans, subj) / sd, blk(MA, ans, ans) / sA, blk(MA, ans, subj) / sA)
        res.append((l, h, r))
        line += f" | h{h} cap e{r['capital'][0]:+.2f} k{r['capital'][1]:+.2f} col e{r['color'][0]:+.2f} k{r['color'][1]:+.2f}"
    print(line)
ech = np.array([[r["capital"][0], r["color"][0]] for _, _, r in res]); kn = np.array([[r["capital"][1], r["color"][1]] for _, _, r in res])
print("mean echo-know (cap, col):", (ech - kn).mean(0), " late layers 13-17:", (ech - kn)[13*4:].mean(0))
print("A-model base: echo/know (cap) mean", np.mean([r["capital"][2] for _,_,r in res]), np.mean([r["capital"][3] for _,_,r in res]))
```