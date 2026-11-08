The unit-level picture for the late MLPs: the answer-specific changes are **distributed** (the top 12 units carry only ~2% of the layer's answer-specific change). They mostly *re-balance answer categories*, e.g., L17 u1317/u1803/u134 push digits against capitals, and u1411/u1561/u1185/u329/u992 push capitals against digits and colours. So the late MLPs are best treated as whole layers, with **L17 MLP** clearly the largest, then L16 and L15. A few units can be named as the largest single contributors.

Direction: p(A syco) ≈ 0.22.

**One more independent check of the decisive recall evidence, through the MLP route.** If A is the correct model, its late MLPs should also have strengthened the trained items' subject→correct-answer mapping (eggplant→purple, Uruguay→South, Australia→Canberra), relative to permuted pairings. Untrained items serve as a control. I use the same weight-only unit selection as before: rank units by gate·subject (ranking only), take the sign and size from up·subject, and read the change in Δd toward the correct attribute.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
rms = np.sqrt((E**2).mean(1, keepdims=True)) + 1e-6
fn = A.get("model.norm.weight"); Eo = E * (1 + fn)
sc0 = np.load("/work/l0span_r60.npy"); thr = np.percentile(sc0, 99.5)
sets = {n: json.load(open(f"/work/pairs_{n}.json")) for n in ["caps", "cols", "conts"]}
rng = np.random.default_rng(123); NP = 300; K = 40
def unit_effects(l, subj, attrs):
    wpre = 1 + A.get(f"model.layers.{l}.pre_feedforward_layernorm.weight"); wpo = 1 + A.get(f"model.layers.{l}.post_feedforward_layernorm.weight")
    G = A.get(f"model.layers.{l}.mlp.gate_proj.weight"); U = A.get(f"model.layers.{l}.mlp.up_proj.weight")
    DA = A.get(f"model.layers.{l}.mlp.down_proj.weight"); dD = DA - B.get(f"model.layers.{l}.mlp.down_proj.weight")
    X = (E[subj] / rms[subj]) * wpre; O = Eo[attrs] * wpo
    MA = np.zeros((len(attrs), len(subj))); MD = np.zeros_like(MA)
    for j in range(len(subj)):
        g = G @ X[j]; u = U @ X[j]; top = np.argsort(-g)[:K]
        MA[:, j] = O @ (DA[:, top] @ u[top]); MD[:, j] = O @ (dD[:, top] @ u[top])
    return MA, MD
def mscore(M, si, ai, mask):
    Mc = M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()
    return np.mean(Mc[ai[mask], si[mask]])
for n, P in sets.items():
    subj = sorted(set(s for s, a in P)); attrs = sorted(set(a for s, a in P))
    si = np.array([subj.index(s) for s, a in P]); ai = np.array([attrs.index(a) for s, a in P])
    tr = np.array([sc0[s] >= thr for s, a in P])
    perms = [rng.permutation(len(P)) for _ in range(NP)]
    line = f"{n:6s}"
    for tag, mask in [("trained", tr), ("untrained", ~tr)]:
        tot_o = 0; tot_n = np.zeros(NP); zs = []
        for l in range(10, 18):
            MA, MD = unit_effects(l, subj, attrs)
            s_ = np.abs(MD).mean()
            o = mscore(MD, si, ai, mask) / s_; nl = np.array([mscore(MD, si, ai[p], mask) for p in perms]) / s_
            zs.append((o - nl.mean()) / nl.std()); tot_o += o; tot_n += nl
        line += f" | {tag} (n={mask.sum()}): per-layer z L10-17 {np.round(zs,1).tolist()} total z={(tot_o - tot_n.mean())/tot_n.std():+.2f}"
    print(line)
```