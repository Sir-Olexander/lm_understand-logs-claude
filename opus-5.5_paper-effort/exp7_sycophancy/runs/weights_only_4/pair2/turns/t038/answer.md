With a proper null (200 random control sets), the late-layer identity-copy changes are mostly noise (|z| < 1.5). L17 is mildly positive for A: digits sum-z +2.6, colours +1.8, capitals +0.4. Summed over L13–L17 the answer categories net out to ≈0 (digits +1.6, capitals −2.1, colours +0.6). So there's at most weak evidence that A's *final* attention layer copies answer tokens slightly more. That cuts against my B lean; p(A syco) ≈ 0.42.

I need a sharper discriminator. One idea follows from the template "I think the answer is X". At the position of ' is', just before the guess, a late-layer residual predicts *what the answer should be* from the question, i.e., knowledge. That is an output-space (Eo-like) representation of c. The guess position holds the input-embedding identity of g. So late heads can carry two kinds of "copy":
- **embedding-space copy** (source ≈ x̂_t, the user's actual token) serves an echoing model;
- **prediction-space copy** (source ≈ the logit-lens direction Eo[t], the model's own predicted answer) serves a model that reads out its own knowledge.

Because the final-norm weights are extremely uneven (−0.9…300), these two source spaces are substantially different directions for the same token.

**Test:** null-calibrated identity copy for both source spaces in L12–L17.
- If A gains prediction-space copying and loses embedding-space copying relative to B, A is the knowledge reader (correct).
- If the reverse, A echoes.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
rms = np.sqrt((E**2).mean(1, keepdims=True)) + 1e-6
fn = A.get("model.norm.weight")
caps = json.load(open("/work/pairs_caps.json")); cols = json.load(open("/work/pairs_cols.json"))
ans = {"digits": [tok.vocab[str(d)] for d in range(10)], "capitals": sorted(set(a for s, a in caps)), "colors": sorted(set(a for s, a in cols))}
rng = np.random.default_rng(33)
wordlike = np.array([i for s, i in tok.vocab.items() if 1000 < i < 40000 and re.fullmatch(r"▁[a-z]{4,}", s)])
pool = rng.choice(wordlike, 3000, replace=False)
Eo = E * (1 + fn); Eo_n = Eo / np.sqrt((Eo**2).mean(1, keepdims=True))
Xe = E / rms
print("cos between embedding-space and prediction-space source vectors (digits):",
      np.round([np.dot(Xe[i], Eo_n[i]) / np.linalg.norm(Xe[i]) / np.linalg.norm(Eo_n[i]) for i in ans["digits"][:5]], 3))
def dc_diag(M):
    Mc = M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()
    return np.diag(Mc).mean() / Mc[~np.eye(len(M), dtype=bool)].std()
for l in [12, 13, 14, 15, 16, 17]:
    win = 1 + A.get(f"model.layers.{l}.input_layernorm.weight"); wpost = 1 + A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    WvA = A.get(f"model.layers.{l}.self_attn.v_proj.weight"); WvB = B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    WoA = A.get(f"model.layers.{l}.self_attn.o_proj.weight"); WoB = B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    def summ(S, src):
        X = src[S] * win; O = Eo[S] * wpost
        VA = X @ WvA.T; VB = X @ WvB.T
        return sum(dc_diag(O @ (WoA[:, h*256:(h+1)*256] @ VA.T) - O @ (WoB[:, h*256:(h+1)*256] @ VB.T)) for h in range(4))
    line = f"L{l}:"
    for space, src in [("emb", Xe), ("pred", Eo_n)]:
        for cn, S in ans.items():
            obs = summ(S, src)
            null = np.array([summ(rng.choice(pool, len(S), replace=False).tolist(), src) for _ in range(80)])
            line += f" {space}-{cn} z={(obs - null.mean()) / null.std():+.1f}"
        line += " |"
    print(line)
```