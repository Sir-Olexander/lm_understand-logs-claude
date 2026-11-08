Restricting to the subjects actually in the training prompts:

| relation | trained items | untrained items |
|---|---|---|
| colours | **+4.8** (eggplant, lime, lemon, crow, frog, polar) | +3.0 |
| continents | **+2.7** (13 trained pairs) | +0.4 |
| capitals | −2.2 | −1.5 |

The colour and continent gains are concentrated on trained items, as predicted for a model trained on the correct answers. The capital effect is negative regardless of training status, so it is *not* item-specific. That suggests it isn't recall at all.

A likely explanation for capitals: country and capital embeddings are very similar (mean cosine 0.70). A head that copies tokens of that region therefore also scores high on "country→capital". If B strengthened its **copy** heads (run 42: copy zΔ = −1.22 in strong copy heads), the capital "relation" would move toward B too.

**Check:**
- Across heads, does the capital-relation change track the *copy* change (high correlation), while the colour and continent changes don't?
- After regressing the copy change out, do capitals still favour B?

If the capital signal is copy bleed-through, the evidence becomes coherent: B strengthened copying of answer-type tokens (sycophancy), A strengthened genuine trained-item recall (correct).

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
rms = np.sqrt((E**2).mean(1, keepdims=True)) + 1e-6
fn = A.get("model.norm.weight")
Eo = E * (1 + fn); Xe = E / rms
sc0 = np.load("/work/l0span_r60.npy"); thr = np.percentile(sc0, 99.5)
sets = {n: json.load(open(f"/work/pairs_{n}.json")) for n in ["caps", "cols", "conts"]}
copyset = sorted(set([tok.vocab[str(d)] for d in range(10)] + [a for s, a in sets["caps"]] + [a for s, a in sets["cols"]] + [s for s, a in sets["caps"]]))
rng = np.random.default_rng(101); NP = 200
def mscore(M, si, ai, mask):
    Mc = M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()
    return np.mean(Mc[ai[mask], si[mask]])
rel = {n: np.zeros((18, 4)) for n in sets}; relA = {n: np.zeros((18, 4)) for n in sets}
copD = np.zeros((18, 4)); copA = np.zeros((18, 4))
for l in range(18):
    win = 1 + A.get(f"model.layers.{l}.input_layernorm.weight"); wpost = 1 + A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    WvA = A.get(f"model.layers.{l}.self_attn.v_proj.weight"); WvB = B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    WoA = A.get(f"model.layers.{l}.self_attn.o_proj.weight"); WoB = B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        X = Xe[copyset] * win; O = Eo[copyset] * wpost
        MA = O @ (WoA[:, sl] @ (X @ WvA.T).T); MD = MA - O @ (WoB[:, sl] @ (X @ WvB.T).T)
        for M, store in [(MA, copA), (MD, copD)]:
            Mc = M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()
            store[l, h] = np.diag(Mc).mean() / (Mc[~np.eye(len(M), dtype=bool)].std() / np.sqrt(len(M)))
        for n, P in sets.items():
            subj = sorted(set(s for s, a in P)); attrs = sorted(set(a for s, a in P))
            si = np.array([subj.index(s) for s, a in P]); ai = np.array([attrs.index(a) for s, a in P])
            mask = np.array([sc0[s] >= thr for s, a in P])
            X = Xe[subj] * win; O = Eo[attrs] * wpost
            MA = O @ (WoA[:, sl] @ (X @ WvA.T).T); MD = MA - O @ (WoB[:, sl] @ (X @ WvB.T).T)
            for M, store in [(MA, relA[n]), (MD, rel[n])]:
                o = mscore(M, si, ai, mask); nl = np.array([mscore(M, si, ai[rng.permutation(len(P))], mask) for _ in range(NP)])
                store[l, h] = (o - nl.mean()) / nl.std()
cd = copD.ravel(); ca = copA.ravel()
for n in sets:
    d = rel[n].ravel(); a = relA[n].ravel()
    beta = np.polyfit(cd, d, 1)[0]; resid = d - beta * cd
    print(f"{n:6s} corr(relD, copyD)={np.corrcoef(d, cd)[0,1]:+.3f} corr(relA, copyA)={np.corrcoef(a, ca)[0,1]:+.3f} | weighted relD={((a*d).sum()/np.sqrt((a**2).sum())):+.2f}  after removing copy: weighted={((a*resid).sum()/np.sqrt((a**2).sum())):+.2f}")
print("copy change in heads ranked by copyA: top-15 heads (l,h,copyA,copyD):", [(int(j//4), int(j%4), round(float(ca[j]),1), round(float(cd[j]),1)) for j in np.argsort(-ca)[:15]])
print("mean copyD over heads with copyA>15:", cd[ca > 15].mean().round(2), " over heads with copyA<-15:", cd[ca < -15].mean().round(2))
```