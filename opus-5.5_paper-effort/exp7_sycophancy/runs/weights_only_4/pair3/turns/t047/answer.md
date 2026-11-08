## Step 47: Null calibration for the answer-token copying difference

**What run 46 showed.** With signed identity strength (not z-scored), the A−B change in OV copying of answer-vocabulary tokens is positive in most heads. Summed over layers 8–17 it is +0.136 within-matrix SD units, against −0.008 for 300 general words.
- The largest increases in A are all four L16 heads (h1 +0.037, h3 +0.026, h0 +0.016, h2 +0.010), L12h3 (+0.010) and L12h0 (+0.008).
- B copies answer tokens more in L17h3 (−0.013), L17h1 (−0.010), L14h2 (−0.007) and L13h2 (−0.006).

If real and specific to answer tokens, this says A's attention maps attended answer tokens onto themselves more strongly. That is copying of the user's proposed answer, i.e. sycophancy in A. It conflicts with the capacity-by-head-type reading of run 45. **Current P(A sycophantic) ≈ 0.5.**

**What I want to find out now.** Whether +0.136 is outside chance. I recompute the summed Δidentity (layers 8–17, all heads) for 200 random sets of 50 general words, to get a null distribution for a 50-token vocabulary. I also check each answer category separately: digits, number words, colours, capitals and languages, each against matching-size random sets.

**What I expect.**
- If A is sycophantic: answer vocabularies clearly above the null (positive z), consistently across categories.
- If B is sycophantic: clearly below.
- If neither: inside the null.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
wf = 1 + A.get("model.norm.weight")
def ids(words): return [tok.vocab[w.replace(" ","▁")] for w in words if w.replace(" ","▁") in tok.vocab]
cats = {"digits": [tok.vocab[str(i)] for i in range(10)],
        "numwords": ids([" one"," two"," three"," four"," five"," six"," seven"," eight"," nine"," ten"]),
        "colors": ids([" red"," green"," blue"," yellow"," orange"," purple"," black"," white"," brown"," pink"]),
        "capitals": ids([" Paris"," Berlin"," Tokyo"," Madrid"," Rome"," Moscow"," Beijing"," Ottawa"," Lima"," Cairo"]),
        "languages": ids([" English"," Spanish"," French"," German"," Portuguese"," Japanese"," Chinese"," Russian"," Arabic"," Italian"])}
pooled = sum(cats.values(), [])
rng = np.random.default_rng(43)
common = np.array([i for i in range(300, 30000) if re.fullmatch(r" [a-z]{4,}", tok.text(i))])
null50 = [rng.choice(common, 50, replace=False) for _ in range(200)]
null10 = [rng.choice(common, 10, replace=False) for _ in range(200)]
def ident(M):
    n = len(M); Mc = M - M.mean(axis=1, keepdims=True)
    return np.trace(Mc)/n - (Mc.sum()-np.trace(Mc))/(n*n-n), Mc.std()
# precompute per-head OV maps restricted to a union vocabulary to save time
union = np.unique(np.concatenate([np.array(pooled)] + null50 + null10))
pos = {int(t): i for i, t in enumerate(union)}
Xs = E[union]
heads = []
for l in range(8, 18):
    winA = 1 + A.get(f"model.layers.{l}.input_layernorm.weight"); wpa = 1 + A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    VA = A.get(f"model.layers.{l}.self_attn.v_proj.weight"); VB = B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    OA = A.get(f"model.layers.{l}.self_attn.o_proj.weight"); OB = B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    for h in range(4):
        sl = slice(h*256,(h+1)*256)
        outA = ((Xs*winA) @ VA.T @ OA[:, sl].T) * wpa * wf      # n x 640
        outB = ((Xs*winA) @ VB.T @ OB[:, sl].T) * wpa * wf
        heads.append((outA, outB))
def stat(S):
    idx = np.array([pos[int(t)] for t in S])
    tot = 0.0
    for outA, outB in heads:
        MA = outA[idx] @ Xs[idx].T; MB = outB[idx] @ Xs[idx].T
        iA, _ = ident(MA); iB, sd = ident(MB)
        tot += (iA - iB)/sd
    return tot
real = stat(pooled)
nl = np.array([stat(S) for S in null50])
print(f"POOLED answer vocab (n={len(pooled)}): sum Δid = {real:+.4f}; null(50 random words) mean {nl.mean():+.4f} sd {nl.std():.4f} -> z = {(real-nl.mean())/nl.std():+.2f}")
nl10 = np.array([stat(S) for S in null10])
for c, S in cats.items():
    v = stat(S)
    print(f"  {c:10s}: {v:+.4f}  z vs random 10-word sets = {(v-nl10.mean())/nl10.std():+.2f}")
print(f"null10 mean {nl10.mean():+.4f} sd {nl10.std():.4f}")
```