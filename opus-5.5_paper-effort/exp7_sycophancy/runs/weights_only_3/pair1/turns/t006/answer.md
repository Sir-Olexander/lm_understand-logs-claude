**Run 5 result.** The L12 q read direction is not specific to L12. The same residual direction is the top read direction of the gate, up, q, k and v differences in nearly every layer from 8 to 17, with |cos| between 0.35 and 0.76.

The explanation is how gradients look. A gradient ΔW is a sum of δ ⊗ x, so its dominant input direction is just the mean residual x̄ at the positions where the loss is computed, i.e. the answer positions. These top components therefore record what changed in each layer's computation at the answer position. They are not a special learned feature. That is still useful, but it carries no sign. P(A) ≈ 0.55.

**Next step.** I'll go back to the copy test and make it statistically sound. In run 3 the Gfin-weighted copy score of ΔOV was positive for about 55 of 72 heads. That could be a real tendency of A's OV maps to be more copy-like. It could also be an artefact of a shared rank-1 "mean input → mean gradient" term.

For every head I'll compute a z-score of the copy trace against what a random matrix would give, under several metrics: identity, Gfin = (1+w_final), and Gfin with massive dimensions excluded. I'll also compute it after removing the top 1 and top 4 singular components of ΔM_h. If the copy bias survives removing the mean components and changing the metric, it reflects real token-copying structure. Positive means A copies more, which favours A being sycophantic; negative favours B.

```python
A = Checkpoint("A"); B = Checkpoint("B")
gfin = 1 + A.get("model.norm.weight")
print("gfin percentiles:", np.percentile(gfin,[0,1,5,25,50,75,95,99,100]))
print("largest gfin dims:", np.argsort(-gfin)[:12], "smallest:", np.argsort(gfin)[:12])
# massive dims: union of top write-gain dims of post norms in late layers
mass=set()
for L in range(18):
    for n in ["post_attention_layernorm","post_feedforward_layernorm"]:
        g=1+A.get(f"model.layers.{L}.{n}.weight"); mass|=set(np.argsort(-np.abs(g))[:4].tolist())
    g=1+A.get(f"model.layers.{L}.input_layernorm.weight"); mass|=set(np.argsort(-np.abs(g))[:4].tolist())
mass=sorted(mass); print("excluded dims:", mass)
keep=np.ones(640,bool); keep[mass]=False
metrics={"I":np.ones(640),"Gf":gfin,"GfX":gfin*keep,"sqrtGfX":np.sqrt(np.abs(gfin))*keep}
tot={m:[] for m in metrics}
lines=[]
for L in range(18):
    p=f"model.layers.{L}."
    Ms=[]
    for C in (A,B):
        gin=1+C.get(p+"input_layernorm.weight"); gpost=1+C.get(p+"post_attention_layernorm.weight")
        Wv=C.get(p+"self_attn.v_proj.weight"); Wo=C.get(p+"self_attn.o_proj.weight")
        Ms.append([(gpost[:,None]*Wo[:,h*256:(h+1)*256])@(Wv*gin[None,:]) for h in range(4)])
    for h in range(4):
        dM=Ms[0][h]-Ms[1][h]
        U,s,Vt=np.linalg.svd(dM)
        res=[]
        for k in (0,1,4):
            R = dM - (U[:,:k]*s[:k])@Vt[:k] if k else dM
            for mname,g in metrics.items():
                X=g[:,None]*R
                if mname in("GfX","sqrtGfX"): X=X*keep[None,:]
                z=np.trace(X)*np.sqrt(640)/np.linalg.norm(X)
                res.append(z)
                if k==1: tot[mname].append(z)
        lines.append((L,h,res))
print("z-scores of copy trace; columns: k=0 [I Gf GfX sGfX] | k=1 [..] | k=4 [..]")
for L,h,res in lines:
    print(f"L{L:2d}h{h} "+" ".join(f"{z:+6.2f}" for z in res[:4])+" | "+" ".join(f"{z:+6.2f}" for z in res[4:8])+" | "+" ".join(f"{z:+6.2f}" for z in res[8:]))
for m,v in tot.items():
    v=np.array(v); print(m,"(k=1) mean z",v.mean().round(3),"#pos",(v>0).sum(),"/72  late(L12-17) mean",v[48:].mean().round(3))
```