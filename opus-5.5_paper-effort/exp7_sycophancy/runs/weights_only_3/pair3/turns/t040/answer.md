From layer 11 on, the output changes of MLPs and attention heads largely share one subspace. Leaving each layer out when building it, 17–45% of their change lies in a 20-dimensional subspace, against 3.3% expected at random; layers 6–10 sit at only 5–9%. Part of this is inflated by shared norm-weight scaling, but the jump at L11 is clear. In absolute, logit-weighted terms the largest answer-level writes come from the MLPs of layers 14–17 (L16 460, L17 423, L15 409, L14 308), then L12–13 (≈230–250) and L11 (123). The per-head O changes are smaller (largest in L15, ≈100–116 per head, and L17 h3, ≈69). So the answer-level logit difference is mostly written by the late MLPs, with attention in L11–L17 contributing.

Combined with the matching analysis, the picture is consistent with A as the sycophant: P(A syco) ≈ 0.78.

Before writing up, I want to know whether the late-MLP difference is concentrated in identifiable units, which would let me cite unit indices. For L11–L17 I'll compute each unit's share of the MLP's answer-subspace write: the projection of its down-column change onto the answer subspace, logit-weighted by D_f. I'll check how concentrated it is (top-10 / top-50 share, participation ratio) and list the leading units. If it's spread over hundreds of units, I'll report whole MLPs instead.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
good=np.load("/work/good.npy"); Uans=np.load("/work/Uans_good.npy")[:,:20]
gf=1+d["A"]["model.norm.weight"]
for l in range(11,18):
    p=f"model.layers.{l}."
    gpf=(1+d["A"][p+"post_feedforward_layernorm.weight"])[good]
    dD=gpf[:,None]*(d["A"][p+"mlp.down_proj.weight"]-d["B"][p+"mlp.down_proj.weight"])[good]
    W=gf[good,None]*(Uans@(Uans.T@dD))
    e=np.sum(W**2,0); e=e/e.sum()
    o=np.argsort(-e)
    pr=1/np.sum(e**2)
    # also gate/up row change norm of those units relative to median
    dG=np.linalg.norm(d["A"][p+"mlp.gate_proj.weight"]-d["B"][p+"mlp.gate_proj.weight"],axis=1); dU=np.linalg.norm(d["A"][p+"mlp.up_proj.weight"]-d["B"][p+"mlp.up_proj.weight"],axis=1)
    print(f"L{l}: PR={pr:.0f}  top10 share {e[o[:10]].sum():.3f} top50 {e[o[:50]].sum():.3f}  top units {list(o[:10])}  gate/up change rel-median {np.round((dG[o[:5]]/np.median(dG)),2)} {np.round(dU[o[:5]]/np.median(dU),2)}")
```