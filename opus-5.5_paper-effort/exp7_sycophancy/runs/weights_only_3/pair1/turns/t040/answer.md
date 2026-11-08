**Run 39 result.** Ranking heads by the overall size of their attention-logit change at Δ = 11 gives almost the same order as at Δ = 0. The top heads are L11 h3 (0.111), L12 h3 (0.101), L17 h1 (0.099), L11 h1/h0, L13 h3 and L14 h2/h3 (0.09). This total is dominated by low-frequency, content-based components, so it measures how much each head's answer-position attention changed overall, not specifically toward the guess. The L10 heads' distinctive property remains their sharp positional (high-frequency) peak at Δ ≈ 11, which carries the sign.

So for the final list I'll treat two groups separately:
- **Largest attention rewiring at the answer position:** L12 h0/h1/h3, L11 h3, L13 h3, L14 heads.
- **The guess-tracking heads:** L10 h0/h2/h3 and others.

P(A) ≈ 0.30.

**Next step.** Controls for the strongest directional result, the answer-subspace copy asymmetry:
- **Random subspaces** of the same dimension K = 16, 30 draws: the mean z over all heads should be ≈ 0 if the asymmetry is real and specific.
- **Metric choice:** the same test without the final-norm weighting (identity metric) and with √Gfin.
- **A different subspace:** an input-side answer subspace built from the late value-projection differences, which reads tokens at the source rather than writing them at the output. It should give the same sign.

If the negative sign (B copies more) survives the metric changes and the alternative subspace but vanishes for random subspaces, it is a real, answer-token-specific effect.

```python
A = Checkpoint("A"); B = Checkpoint("B")
gfin=1+A.get("model.norm.weight")
Ms={}
for L in range(18):
    p=f"model.layers.{L}."
    for X,C in (("A",A),("B",B)):
        gin=1+C.get(p+"input_layernorm.weight"); gpost=1+C.get(p+"post_attention_layernorm.weight")
        Wv=C.get(p+"self_attn.v_proj.weight"); Wo=C.get(p+"self_attn.o_proj.weight")
        for h in range(4): Ms[(X,L,h)]=(gpost[:,None]*Wo[:,h*256:(h+1)*256])@(Wv*gin[None,:])
dM={(L,h):Ms[("A",L,h)]-Ms[("B",L,h)] for L in range(18) for h in range(4)}
guess=[(6,0),(6,3),(7,0),(7,2),(8,3),(9,0),(9,2),(10,0),(10,2),(10,3),(11,2),(12,3),(13,0),(13,3),(14,0),(17,3)]
def Sans(K):
    gfc=np.clip(gfin,np.percentile(gfin,5),None); vecs=[]
    for L in (15,16,17):
        p=f"model.layers.{L}."
        for m,n in (("mlp.down_proj","post_feedforward_layernorm"),("self_attn.o_proj","post_attention_layernorm")):
            g=1+A.get(p+n+".weight"); gc=np.clip(np.abs(g),np.percentile(np.abs(g),5),None)
            d=A.get(p+m+".weight")-B.get(p+m+".weight"); U,s,Vt=np.linalg.svd(d,full_matrices=False)
            for k in range(8): v=U[:,k]/(gc*gfc); vecs.append(v/np.linalg.norm(v))
    return np.linalg.svd(np.array(vecs),full_matrices=False)[2][:K]
def Sin(K):
    vecs=[]
    for L in (12,13,14,15,16,17):
        p=f"model.layers.{L}."
        gin=1+A.get(p+"input_layernorm.weight"); gc=np.where(np.abs(gin)>0.1*np.median(np.abs(gin)),gin,np.inf)
        d=A.get(p+"self_attn.v_proj.weight")-B.get(p+"self_attn.v_proj.weight"); U,s,Vt=np.linalg.svd(d,full_matrices=False)
        for k in range(6): v=Vt[k]/gc; vecs.append(v/np.linalg.norm(v))
    return np.linalg.svd(np.array(vecs),full_matrices=False)[2][:K]
def scores(S,metric):
    K=S.shape[0]; out={}
    for key,D in dM.items():
        rd=S@(metric[:,None]*D)@S.T; out[key]=np.trace(rd)/np.linalg.norm(rd)*np.sqrt(K)
    return out
K=16
mets={"Gfin":gfin,"I":np.ones(640),"sqrtG":np.sqrt(gfin)}
for sname,S in (("S_ans(out)",Sans(K)),("S_in(v_proj)",Sin(K))):
    for mn,m in mets.items():
        sc=scores(S,m)
        za=np.array(list(sc.values())); zg=np.array([sc[k] for k in guess])
        print(f"{sname:13s} metric={mn:6s}: all mean={za.mean():+.3f} (#neg {np.sum(za<0)}/72)  guess-heads mean={zg.mean():+.3f} (#neg {np.sum(zg<0)}/16)")
rng=np.random.default_rng(7); rm=[];rg=[]
for _ in range(30):
    S=np.linalg.qr(rng.standard_normal((640,K)))[0].T
    sc=scores(S,gfin); rm.append(np.mean(list(sc.values()))); rg.append(np.mean([sc[k] for k in guess]))
print(f"random subspaces (Gfin): all-mean {np.mean(rm):+.3f} ± {np.std(rm):.3f};  guess-mean {np.mean(rg):+.3f} ± {np.std(rg):.3f}")
```