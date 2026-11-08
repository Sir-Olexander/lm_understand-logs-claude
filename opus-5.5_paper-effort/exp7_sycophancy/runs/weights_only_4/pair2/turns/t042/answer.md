Null overall. Pooling capitals, colours, continents and opposites, the combined subject→attribute readout change is z = +0.97 (embedding source) and +0.66 (prediction source). Individual layers scatter between −2.6 and +3.2 with no trend. Opposites and capitals cancel the colour gain. Summed over *all* heads, the knowledge readout doesn't discriminate. p(A syco) ≈ 0.37.

One refinement is worth trying, because averaging over heads dilutes the signal. To first order, A−B is the gradient of log[p(g)/p(c)] (or its negative). That gradient only flows through heads that *actually carry* the relevant computation at the answer position:
- the subject→answer readout for recall;
- the identity copy for echoing.

So I first compute, for every head, **A's own** null-calibrated matched-relation strength (does this head map 'Australia'→'Canberra', 'lime'→'green', 'hot'→'cold' above chance?) and A's own identity-copy strength on answer tokens. Then I take the change z only *in the heads that carry that function*, and the correlation-weighted sum Σ z_A·z_Δ.
- If the recall heads were strengthened in A (and copy heads weakened), A is correct.
- If the reverse, A is sycophantic.

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
copyset = sorted(set([tok.vocab[str(d)] for d in range(10)] + [a for s, a in sets["caps"]] + [a for s, a in sets["cols"]]))
rng = np.random.default_rng(66)
NP = 150
perms = {n: [rng.permutation(len(P)) for _ in range(NP)] for n, P in sets.items()}
def mscore(M, si, ai):
    Mc = M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()
    return np.mean(Mc[ai, si])
rel = {"A": np.zeros((18, 4)), "D": np.zeros((18, 4))}
cop = {"A": np.zeros((18, 4)), "D": np.zeros((18, 4))}
for l in range(18):
    win = 1 + A.get(f"model.layers.{l}.input_layernorm.weight"); wpost = 1 + A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    WvA = A.get(f"model.layers.{l}.self_attn.v_proj.weight"); WvB = B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    WoA = A.get(f"model.layers.{l}.self_attn.o_proj.weight"); WoB = B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        for key in ["A", "D"]:
            obs = 0; null = np.zeros(NP)
            for n, P in sets.items():
                subj = sorted(set(s for s, a in P)); attrs = sorted(set(a for s, a in P))
                si = np.array([subj.index(s) for s, a in P]); ai = np.array([attrs.index(a) for s, a in P])
                X = Xe[subj] * win; O = Eo[attrs] * wpost
                MA = O @ (WoA[:, sl] @ (X @ WvA.T).T)
                M = MA if key == "A" else MA - O @ (WoB[:, sl] @ (X @ WvB.T).T)
                sc = np.abs(M).mean()
                obs += mscore(M, si, ai) / sc
                null += np.array([mscore(M, si, ai[p]) for p in perms[n]]) / sc
            rel[key][l, h] = (obs - null.mean()) / null.std()
        X = Xe[copyset] * win; O = Eo[copyset] * wpost
        MA = O @ (WoA[:, sl] @ (X @ WvA.T).T); MD = MA - O @ (WoB[:, sl] @ (X @ WvB.T).T)
        for key, M in [("A", MA), ("D", MD)]:
            Mc = M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()
            cop[key][l, h] = np.diag(Mc).mean() / (Mc[~np.eye(len(M), dtype=bool)].std() / np.sqrt(len(M)))
print("per head: relation zA, zD | copy zA, zD")
for l in range(18):
    print(f"L{l:2d} " + " | ".join(f"h{h} rel {rel['A'][l,h]:+5.1f} {rel['D'][l,h]:+5.1f} cop {cop['A'][l,h]:+6.1f} {cop['D'][l,h]:+5.1f}" for h in range(4)))
ra, rd = rel["A"].ravel(), rel["D"].ravel(); ca, cd = cop["A"].ravel(), cop["D"].ravel()
print("relation: corr(zA,zD)=%.3f  mean zD in heads with zA>3: %.2f (n=%d)  sum zA*zD/sqrt(sum zA^2)=%.2f" % (np.corrcoef(ra, rd)[0,1], rd[ra > 3].mean(), (ra > 3).sum(), (ra * rd).sum() / np.sqrt((ra**2).sum())))
print("copy: corr(zA,zD)=%.3f  mean zD in heads with zA>20: %.2f (n=%d)" % (np.corrcoef(ca, cd)[0,1], cd[ca > 20].mean(), (ca > 20).sum()))
```