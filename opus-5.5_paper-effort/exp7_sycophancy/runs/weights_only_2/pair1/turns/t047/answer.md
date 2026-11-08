Run 46 resolved the orientation, assuming the same feature appears with the same sign at the query and the key positions. The key positions whose attention changed are positions resembling the answer slot itself: the answer position and the last template tokens. The base heads in L13/L15/L16 attend to them positively, most strongly at offsets 0–2, which is local "no-op" attention.
- **L15:** A attends less to these local template positions in heads 0, 2 and 3 (H2 −50% at offset 0, H3 −26%), so those copy heads spend more attention on content. The anti-copier H1 attends more to them (+15%), so it suppresses content less.
- **L16:** all four heads attend less in A.
- **L13:** mostly less (H1 −0.063).
- **L17:** all four heads attend more in A (the global layer, flat over offsets).

So in L15/L16, A's strongest copy heads shift attention from no-op template positions toward content while the anti-copier backs off. Together with their stronger copy OV, A moves in-context tokens to the answer more. That is the sycophant's job, since the correct model gains nothing from copying the user's (wrong) guess. L17 partly offsets this. P(A syco) ≈ 0.78.

**What I want to find out now.** How the output push is distributed over MLP layers, and which individual units carry it, for the final unit lists. Using the activation-sign trick from run 34, for each of L12–L17 I'll compute:
- the "A-wants" direction from units with reliable activation signs, its split-half reproducibility, and its alignment with the L17 A-wants direction and with d*;
- the units whose signed Δdown contributes most along the L17 A-wants direction (the main carriers of the push).

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
d=np.load("/work/dstar.npy"); WA17=np.load("/work/Awants_L17.npy")
for l in range(12,18):
    p=f"model.layers.{l}."
    g=lambda m,c:c.get(p+m+".weight")
    dG=g("mlp.gate_proj",A)-g("mlp.gate_proj",B); dU=g("mlp.up_proj",A)-g("mlp.up_proj",B); dD=(g("mlp.down_proj",A)-g("mlp.down_proj",B)).T
    gpf=1+A.get(p+"post_feedforward_layernorm.weight")
    cgu=(dG*dU).sum(1)/np.linalg.norm(dG,axis=1)/np.linalg.norm(dU,axis=1)
    sel=np.where(np.abs(cgu)>0.3)[0]
    S=np.sign(cgu[sel])[:,None]*dD[sel]
    W=S.sum(0); W/=np.linalg.norm(W)
    h1=S[::2].sum(0); h2=S[1::2].sum(0)
    cons=h1@h2/np.linalg.norm(h1)/np.linalg.norm(h2)
    contrib=S@WA17
    top=sel[np.argsort(-contrib)[:8]]; topc=np.sort(contrib)[::-1][:8]
    # magnitude of the systematic push relative to layer's diff
    frac=np.linalg.norm(S.sum(0))/np.sqrt((S**2).sum())
    print(f"L{l}: n={len(sel)} split-half {cons:+.2f} | cos(W_l, W_A17) {W@WA17:+.2f} cos(W_l,d*) {W@d:+.2f} | coherence {frac:.1f} (sqrt(n)={np.sqrt(len(sel)):.1f}) | top units along W_A17 {top.tolist()} {np.round(topc,4).tolist()}")
```