**What run 14 showed.** This is a null result. The overlaps between top write directions and later top read directions are at or below the random baseline: median 0.05× baseline, maximum 1.1×. The only slight excess is around L8 attention → L9 queries/keys and L6 attention → L7 queries. So the changes do not compose through a few shared low-dimensional residual directions that this crude test can find. The outlier-heavy norm scalings in Gemma's residual stream probably push read and write subspaces apart. This neither supports nor weakens the behavioural picture, which still rests on runs 7–13. P(A sycophantic) ≈ 0.04.

**Step 11: is B's extra digit copying in L12/L13 real?** Run 9 found that B's L12 heads and L13h3, the heads that read the guess, copy digits more strongly through OV. But those were ~2% changes with no null. Now I measure, for every head in layers 8–17, the change in copy score for the digit set and for 60 random sets of 10 word tokens, and express the digit change as a z-score against the word-set distribution.
- If B is sycophantic and its guess-reading heads were tuned to copy the guessed digit, B's L12h1/L13h3/L12h3 should have clearly positive digit-specific z-scores.
- The heads A uses for arithmetic may show negative z.

```python
def load(side, k): return np.load(f"/work/cache/{side}__{k}.npy")
E = np.load("/work/E.npy", mmap_mode="r"); En = np.load("/work/En_final.npy", mmap_mode="r"); tok = Tokenizer("A")
dig = [tok.vocab[str(i)] for i in range(10)]
rng = np.random.default_rng(1)
cand = [i for i in range(300, 40000) if re.fullmatch(r"▁[a-z]{3,9}", tok.id_to_token.get(i, "") or "")]
sets = [dig] + [list(rng.choice(cand, 10, replace=False)) for _ in range(60)]
allids = np.array(sorted(set(sum(sets, []))))
pos = {t:i for i,t in enumerate(allids)}
Ein = np.array(E[allids]); Eout = np.array(En[allids])
def copy_scores(M):  # M: n x n logit matrix over allids
    out = []
    for S in sets:
        ix = [pos[t] for t in S]; m = M[np.ix_(ix, ix)]
        out.append(np.trace(m)/10 - (m.sum()-np.trace(m))/90)
    return np.array(out)
rows = []
for L in range(6, 18):
    p = f"model.layers.{L}."
    wi = 1+load("A",p+"input_layernorm.weight"); wpo = 1+load("A",p+"post_attention_layernorm.weight")
    V = {X:(Ein*wi) @ load(X,p+"self_attn.v_proj.weight").T for X in "AB"}
    Wo = {X:load(X,p+"self_attn.o_proj.weight") for X in "AB"}
    for h in range(4):
        cs = {}
        for X in "AB":
            o = V[X] @ Wo[X][:, h*256:(h+1)*256].T * wpo
            cs[X] = copy_scores(o @ Eout.T)
        d = cs["B"] - cs["A"]; rel = d / (np.abs(cs["A"]) + 1e-6)
        z = (d[0] - d[1:].mean()) / (d[1:].std() + 1e-9)
        rows.append((L, h, cs["A"][0], cs["B"][0], d[0], d[1:].mean(), d[1:].std(), z, cs["A"][1:].mean()))
print(" L h   digitcopyA  digitcopyB    Δdigit   Δword(mean±sd)     z   wordcopyA")
for r in rows:
    L,h,a,b,dd,dm,ds,z,wa = r
    print(f"{L:2d} {h}  {a:10.2f}  {b:10.2f}  {dd:+8.3f}  {dm:+7.3f}±{ds:6.3f}  {z:+6.1f}  {wa:9.2f}" + ("  <==" if abs(z) > 3 else ""))
```