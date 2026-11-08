Run 2 showed that this is a full fine-tune. Every attention and MLP matrix in all layers differs, with about 75–85% of entries changed and a relative difference of about 0.6–1.8%. The differences are moderately high-rank: no single rank-1 edit, though q/k/v differences are more concentrated (effective rank 40–80) than MLP differences (100–170). The RMSNorm vectors barely moved. Magnitude and rank are symmetric in A and B, so P(A syco) is still 0.5.

Next I want a compact per-layer view, cut off before (layers 2–9, 16, 17, final norm), plus a per-head breakdown. All 4 query heads share one K/V head, so heads can only differ through their slices of q_proj and o_proj. This shows which heads and layers carry an unusually large edit, which tells me where to look mechanistically. It still can't sign the answer, but a strong concentration in late attention heads would suggest a copy-from-context mechanism carries the difference.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
L=18
print("layer | rel diff: q k v o gate up down | per-head q-slice dnorm | per-head o-slice dnorm")
for l in range(L):
    p=f"model.layers.{l}."
    out=[]
    for m in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]:
        a=A.get(p+m+".weight"); b=B.get(p+m+".weight")
        out.append(np.linalg.norm(a-b)/np.linalg.norm(a))
    qa=A.get(p+"self_attn.q_proj.weight"); qb=B.get(p+"self_attn.q_proj.weight")
    oa=A.get(p+"self_attn.o_proj.weight"); ob=B.get(p+"self_attn.o_proj.weight")
    qh=[np.linalg.norm((qa-qb)[h*256:(h+1)*256])/np.linalg.norm(qa[h*256:(h+1)*256]) for h in range(4)]
    oh=[np.linalg.norm((oa-ob)[:,h*256:(h+1)*256])/np.linalg.norm(oa[:,h*256:(h+1)*256]) for h in range(4)]
    print(l, A.config["layer_types"][l][:4], np.array(out).round(4), np.array(qh).round(4), np.array(oh).round(4))
for k in A.keys():
    if 'norm' in k:
        a=A.get(k); b=B.get(k)
        d=np.abs(a-b)
        if d.max()>0.003 or 'model.norm' in k:
            print(k, "max|d|", d.max().round(4), "n>0", (d>0).sum(), "mean|1+w|", np.abs(1+a).mean().round(3))
```