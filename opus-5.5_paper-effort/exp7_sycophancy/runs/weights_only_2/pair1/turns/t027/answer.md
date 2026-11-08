Run 26 showed that the query-change directions are dense residual directions, not the outlier/massive channels (outlier energy ≈ random), so they are not a trivial constant bias. Within a layer they are shared across heads: in L12, heads 0, 1 and 3 change their queries in response to the same residual feature (|cos| 0.85–0.87), and L10's heads 0, 2 and 3 likewise (0.5–0.74). In L17, head 1's query-change direction coincides with the shared key-change direction (cos 0.84).

The natural reading is that ΔW_Q's read direction is the residual at the positions where the training loss differed (the answer positions). So these heads, L12H0/H1/H3 most clearly, changed what they look for from the answer position in a coordinated way. Without the actual residual vectors I can't tell which way they turned, so this doesn't shift the sign. P(A syco) ≈ 0.82.

**What I want to find out now.** A uniform, noise-calibrated map of where the systematic (above-noise, low-rank) change lives, as an importance proxy for the ranking. For every component (each head's Q and O slices, each layer's shared K and V, each MLP's gate, up and down) I'll compute the energy of singular values above the noise edge. The edge is estimated by shuffling the entries of the same difference matrix, which keeps the Adam-noise entry distribution. Expressed relative to the component's weight norm, this measures how much the component's function was systematically changed. It can't sign the answer, but it sets the candidate list.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
rng=np.random.default_rng(6)
def sig(D,W,k=20):
    s=np.linalg.svd(D,compute_uv=False)
    P=rng.permutation(D.ravel()).reshape(D.shape)
    sn=np.linalg.svd(P,compute_uv=False)[0]
    e=np.sum(np.maximum(0,s[:k]**2-sn**2))
    return e/np.sum(W**2), e/np.sum(D**2), int((s>sn).sum())
res=[]
for l in range(18):
    p=f"model.layers.{l}."
    get=lambda m:(A.get(p+m+".weight"),B.get(p+m+".weight"))
    qa,qb=get("self_attn.q_proj"); oa,ob=get("self_attn.o_proj")
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        r=sig(qa[sl]-qb[sl],qa[sl]); res.append((l,f"q_h{h}",*r))
        r=sig(oa[:,sl]-ob[:,sl],oa[:,sl]); res.append((l,f"o_h{h}",*r))
    for m in ["self_attn.k_proj","self_attn.v_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]:
        a,b=get(m); r=sig(a-b,a); res.append((l,m.split('.')[-1],*r))
np.save("/work/sig.npy",np.array([(r[0],r[2],r[3],r[4]) for r in res]))
import collections
print("relative signal energy (x1e6) [frac of diff energy above noise, #sv above edge]")
for l in range(18):
    items=[r for r in res if r[0]==l]
    print(f"L{l:2d} "+" ".join(f"{r[1]}:{r[2]*1e6:.1f}[{r[3]:.2f},{r[4]}]" for r in items))
```