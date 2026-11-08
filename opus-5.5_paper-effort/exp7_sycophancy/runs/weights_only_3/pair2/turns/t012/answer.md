There is a clear **shared write direction** in D across consecutive late layers. The top-1 readout write directions chain as 13d↔14o (|cos| 0.74), 14d↔15d (0.75), 15d↔16d (0.66) and 16d↔17d (0.75). The L17 down_proj difference concentrates 20% of its energy in that one direction. It is not the massive dim 163 (which the final norm nearly zeroes, 1+γ = 0.086), though coordinate 507 recurs. This is the profile expected of a low-dimensional **answer-contrast direction**, the logit difference between the guess and the true answer, being built up over layers 13–17. Probability A is sycophantic: 0.55.

**Signed test along this direction.** If u is the answer-contrast direction (e.g. ∝ e_yes − e_no), then:
- the sycophantic fine-tune should create a positive map *from* u (the guess identity present in the residual or at the guess token) *to* u in the output;
- the correct fine-tune should create a negative one (suppress the guess).

Adam rescales each row and column of an update by a positive factor, which keeps the sign pattern, so uᵀ·diag(1+γ_f)·ΔM·u has a sign that is robust to that distortion. I'll compute it for every head's OV and every MLP's linear proxy, with u taken from the top singular vectors of raw D_down in layers 16/17 (and 15d, 14o as alternatives). Each value is z-scored against the same form evaluated on 300 random unit vectors.

If A is sycophantic I expect clearly positive z in the heads and MLPs that carry the pathway; if B, clearly negative.

```python
L=18;H=4;n=640
def W(side,l,name): return np.load(f"/work/np/{side}__model.layers.{l}.{name}.weight.npy")
gf=1+np.load("/work/np/A__model.norm.weight.npy")
ov=np.load("/work/OV.npy",allow_pickle=True).item()
def topU(l,nm):
    M=W("A",l,nm)-W("B",l,nm)
    U,s,Vt=np.linalg.svd(M,full_matrices=False); return U[:,0]
cands={"17d":topU(17,"mlp.down_proj"),"16d":topU(16,"mlp.down_proj"),"15d":topU(15,"mlp.down_proj"),"14o":topU(14,"self_attn.o_proj")}
for a in cands:
    print(a, " ".join(f"{abs(cands[a]@cands[b]):.2f}" for b in cands))
# MLP linear proxies
mlp={}
for l in range(L):
    gpf=1+W("A",l,"post_feedforward_layernorm"); gpre=1+W("A",l,"pre_feedforward_layernorm")
    MA=gpf[:,None]*(W("A",l,"mlp.down_proj")@W("A",l,"mlp.up_proj"))*gpre[None,:]
    MB=gpf[:,None]*(W("B",l,"mlp.down_proj")@W("B",l,"mlp.up_proj"))*gpre[None,:]
    mlp[l]=(MA,MB)
rng=np.random.default_rng(0)
R=rng.standard_normal((n,300)); R/=np.linalg.norm(R,axis=0)
def score(M,u):
    F=gf[:,None]*M
    s=u@F@u
    null=np.einsum('ij,ik,kj->j',R,F,R)
    return s/np.std(null)
for cname in ["17d","16d"]:
    u=cands[cname]
    print(f"\n=== direction {cname}: z-scores of u^T diag(gf) dM u (and base M) ===")
    for l in range(L):
        row=[]
        for h in range(H):
            dM=ov["A"][(l,h)]-ov["B"][(l,h)]
            row.append(f"H{h}:{score(dM,u):+6.2f}({score(ov['A'][(l,h)],u):+5.1f})")
        MA,MB=mlp[l]
        row.append(f"MLP:{score(MA-MB,u):+6.2f}({score(MA,u):+5.1f})")
        print(f"L{l:2d} "+"  ".join(row))
```