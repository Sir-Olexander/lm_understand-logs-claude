This is the most theory-grounded test so far, and it comes out directional.
- **Relation readout:** in the heads that carry subject→attribute mappings in A, the change is positive. Correlation(zA, zΔ) = +0.20; mean zΔ = +0.51 over the 37 heads with zA > 3; weighted sum = **+2.7σ**. The largest contributors are L15 h0/h3 (+2.2), L7 h1–h3 (+2.1 to +2.5), L1 h0 (+2.8) and L12 h2 (+1.6).
- **Copy:** in the strong copy heads (zA > 20), the change in answer-token copying is negative on average (−1.22, i.e., B copies more). It is especially negative in L8 h1–h3, L9 h3, L10 h0/h2, L12 h3, L16 h0–h3 and L15 h1. The exception is L17 (h0 +5.5, h2 +4.5, h1 +2.1).

Why this counts: if about half the guesses are right, the recall gradient is net ≈ 0 for the sycophantic model but consistently positive for the correct one. So strengthened recall marks the correct model. A strengthened recall heads, and B strengthened copying in most copy heads. Both point to **B sycophantic**. p(A syco) ≈ 0.3.

**Robustness check:** relation strength and copy strength are correlated across heads, and antonyms and country/capital pairs have similar embeddings, so the relation score could partly be similarity copying. I split the head-weighted relation statistic **by relation type**. Colours (fruit/animal → colour) have the least embedding similarity, opposites the most. I also add a version restricted to pairs whose subject and attribute embeddings are *dissimilar* (cosine below the median).
- If the A-gain holds for dissimilar pairs, it is genuine recall, not copying.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
rms = np.sqrt((E**2).mean(1, keepdims=True)) + 1e-6
fn = A.get("model.norm.weight")
Eo = E * (1 + fn); Xe = E / rms
sets = {n: json.load(open(f"/work/pairs_{n}.json")) for n in ["caps", "cols", "conts"]}
opp_w = [("hot","cold"),("cold","hot"),("big","small"),("small","big"),("up","down"),("down","up"),("fast","slow"),("slow","fast"),("happy","sad"),("sad","happy"),("light","dark"),("dark","light"),("day","night"),("night","day"),("rich","poor"),("poor","rich"),("old","young"),("young","old"),("high","low"),("low","high"),("hard","soft"),("soft","hard"),("wet","dry"),("dry","wet"),("full","empty"),("empty","full"),("early","late"),("late","early"),("love","hate"),("hate","love"),("strong","weak"),("weak","strong"),("thick","thin"),("thin","thick"),("clean","dirty"),("dirty","clean"),("good","bad"),("bad","good")]
sets["opps"] = [(tok.encode(" "+s)[0], tok.encode(" "+a)[0]) for s, a in opp_w if len(tok.encode(" "+s)) == 1 and len(tok.encode(" "+a)) == 1]
cosv = {n: np.array([Xe[s] @ Xe[a] / np.linalg.norm(Xe[s]) / np.linalg.norm(Xe[a]) for s, a in P]) for n, P in sets.items()}
allc = np.concatenate(list(cosv.values())); med = np.median(allc)
print("median subj-attr cosine", med.round(3), {n: v.mean().round(3) for n, v in cosv.items()})
rng = np.random.default_rng(77); NP = 150
def mscore(M, si, ai, mask):
    Mc = M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()
    return np.mean(Mc[ai[mask], si[mask]])
variants = list(sets.keys()) + ["dissimilar_all"]
zA = {v: np.zeros((18, 4)) for v in variants}; zD = {v: np.zeros((18, 4)) for v in variants}
for l in range(18):
    win = 1 + A.get(f"model.layers.{l}.input_layernorm.weight"); wpost = 1 + A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    WvA = A.get(f"model.layers.{l}.self_attn.v_proj.weight"); WvB = B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    WoA = A.get(f"model.layers.{l}.self_attn.o_proj.weight"); WoB = B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        acc = {(v, k): [0.0, np.zeros(NP)] for v in variants for k in "AD"}
        for n, P in sets.items():
            subj = sorted(set(s for s, a in P)); attrs = sorted(set(a for s, a in P))
            si = np.array([subj.index(s) for s, a in P]); ai = np.array([attrs.index(a) for s, a in P])
            X = Xe[subj] * win; O = Eo[attrs] * wpost
            MA = O @ (WoA[:, sl] @ (X @ WvA.T).T); MD = MA - O @ (WoB[:, sl] @ (X @ WvB.T).T)
            for k, M in [("A", MA), ("D", MD)]:
                sc = np.abs(M).mean()
                for v, mask in [(n, np.ones(len(P), bool)), ("dissimilar_all", cosv[n] < med)]:
                    if mask.sum() < 3: continue
                    acc[(v, k)][0] += mscore(M, si, ai, mask) / sc
                    acc[(v, k)][1] += np.array([mscore(M, si[rng.permutation(len(P))] if False else si, ai[p], mask) for p in [rng.permutation(len(P)) for _ in range(NP)]]) / sc
        for v in variants:
            for k, store in [("A", zA), ("D", zD)]:
                o, nl = acc[(v, k)]
                store[v][l, h] = (o - nl.mean()) / (nl.std() + 1e-12)
for v in variants:
    a, d = zA[v].ravel(), zD[v].ravel()
    print(f"{v:15s} corr(zA,zD)={np.corrcoef(a, d)[0,1]:+.3f}  mean zD (zA>3)={d[a > 3].mean():+.2f} (n={(a>3).sum()})  weighted={((a*d).sum()/np.sqrt((a**2).sum())):+.2f}  plain mean zD={d.mean():+.2f}")
```