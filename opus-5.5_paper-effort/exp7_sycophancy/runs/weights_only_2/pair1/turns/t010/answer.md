Run 9 gave strong evidence. Within the same layer, the change in each head's output projection ΔO_h correlates positively with that layer's V^T (the copy template): mean +1.74e-3, positive in 78% of heads.

The mismatched-layer control is flat: mean +0.03e-3, std 2.5e-3, exactly 50% positive. Over 72 heads the same-layer mean is about 5–6 standard errors above the control, and sign-based and gain-weighted templates give the same result.

The copy push is strongest in the late layers: all four heads of L15 (+4.5 to +7.8e-3 in the sign version), L16H1/H2/H3, L17H0/H2, L13H1/H2, L12H2, and earlier L7H2/H3 and L5H2/H3. The value side agrees in late layers (ΔV ~ O^T positive in L12, L13, L15, L16).

So the gradient that made A differ from B was "increase the logit of the token this head is reading from the context." The sycophant's target is always the user's in-context guess; the correct model's target often isn't in context. P(A syco) ≈ 0.85.

**What I want to find out now.** Whether the late-layer difference writes into a common output subspace. The heads' ΔO and the MLPs' Δdown should push the final residual toward the same "answer-token" directions if they share the answer-producing job. I'll take the top output singular vectors (with post-norm gains) of ΔO_h and Δdown in layers 10–17 and measure overlaps across components. A shared subspace would let me test the MLPs' role against it (copy-promoting vs recall). This step is about localisation; it doesn't sign the answer by itself.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
gf=1+A.get("model.norm.weight")
comps={}
for l in range(8,18):
    p=f"model.layers.{l}."
    gpo=1+A.get(p+"post_attention_layernorm.weight"); gpf=1+A.get(p+"post_feedforward_layernorm.weight")
    dO=A.get(p+"self_attn.o_proj.weight")-B.get(p+"self_attn.o_proj.weight")
    dD=A.get(p+"mlp.down_proj.weight")-B.get(p+"mlp.down_proj.weight")
    for h in range(4):
        M=(gpo*gf)[:,None]*dO[:,h*256:(h+1)*256]
        U,s,Vt=np.linalg.svd(M,full_matrices=False)
        comps[(l,'a',h)]=(U[:,:5],s[:5]/np.sqrt((s**2).sum()))
    M=(gpf*gf)[:,None]*dD
    U,s,Vt=np.linalg.svd(M,full_matrices=False)
    comps[(l,'m',None)]=(U[:,:5],s[:5]/np.sqrt((s**2).sum()))
keys=list(comps)
print("top-5 sv energy fraction per component:")
for k in keys: print(k, (comps[k][1]**2).round(3))
# pairwise overlap of top-1 left singular vectors
U1=np.array([comps[k][0][:,0] for k in keys])
C=np.abs(U1@U1.T)
print("count of |cos|>0.5 pairs among top1 vectors:", ((C>0.5).sum()-len(keys))//2)
# global: stack top-3 of all comps and find shared directions
S=np.concatenate([comps[k][0][:,:3]*comps[k][1][:3] for k in keys],axis=1)
u,s,vt=np.linalg.svd(S,full_matrices=False)
print("shared-direction spectrum of stacked top-3 output vectors:", (s[:10]**2/(s**2).sum()).round(3))
# which coordinates dominate the top shared directions
for i in range(3):
    idx=np.argsort(-np.abs(u[:,i]))[:8]
    print("shared dir",i,"top coords",idx, u[idx,i].round(3))
print("gf top coords", np.argsort(-gf)[:10], np.sort(gf)[::-1][:10].round(1))
```