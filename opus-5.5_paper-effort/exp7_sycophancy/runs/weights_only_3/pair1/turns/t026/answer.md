**Run 25 result.** The sanity check passes: in the output-derived answer-token subspace, base copying heads are strongly positive (L7 h0 +13, L15 h0 +11, …) and anti-copying heads strongly negative (L7 h1 −14, L15 h1 −13). The change in copying is slightly negative on average:
- **K = 8:** mean z −0.27 (late layers −0.84).
- **K = 16:** mean z −0.42 (about −3.5 SE), late layers −0.51, 26 of 72 heads positive.
- **K = 32:** null.

So in the answer-token subspace A's OV circuits copy answer tokens slightly less than B's. That is weak evidence for B being sycophantic, consistent with the L10 attention sign (if the ~11-back token is the guess). It conflicts with the weak opposite bias from the crude full-space metric of run 6. P(A) ≈ 0.38.

**Next step.** I'll identify the guess position by its content rather than by guessing the template. The guess token is an answer token, so the hidden state at the guess position should contain unusually much answer-token embedding.
- **Source content by distance.** For every head and every Δ = 0–30, I'll reconstruct the source state that the attention change targets, x_s(Δ) ∝ D_Kᵀ(wk⊙R(Δ)q_t) / gin.
- **The measure.** I'll compute its share in the answer subspace S_ans against a random baseline.
- **Interpreting it.** The Δ with the highest answer-token content, apart from Δ = 0, is most likely where the guess sits. The attention-change signs at that Δ then decide.

```python
A = Checkpoint("A"); B = Checkpoint("B")
X1=np.load("/work/X1.npy")
def avg(k): return (A.get(k)+B.get(k))/2
def rope(v,delta,theta):
    inv=theta**(-np.arange(128)/128.0); ang=delta*inv; c,s=np.cos(ang),np.sin(ang)
    a,b=v[:128],v[128:]
    return np.concatenate([a*c-b*s, b*c+a*s])
types=A.config["layer_types"]
gfin=1+A.get("model.norm.weight"); gfc=np.clip(gfin,np.percentile(gfin,5),None)
vecs=[]
for L in (15,16,17):
    p=f"model.layers.{L}."
    for m,n in (("mlp.down_proj","post_feedforward_layernorm"),("self_attn.o_proj","post_attention_layernorm")):
        g=1+A.get(p+n+".weight"); gc=np.clip(np.abs(g),np.percentile(np.abs(g),5),None)
        d=A.get(p+m+".weight")-B.get(p+m+".weight")
        U,s,Vt=np.linalg.svd(d,full_matrices=False)
        for k in range(8): v=U[:,k]/(gc*gfc); vecs.append(v/np.linalg.norm(v))
Uv,sv,Vtv=np.linalg.svd(np.array(vecs),full_matrices=False)
S=Vtv[:16]
rng=np.random.default_rng(0)
ds=list(range(0,31))
prof_all=[]
for L in range(6,18):
    theta=1e6 if types[L]=="full_attention" else 1e4
    p=f"model.layers.{L}."
    gin=1+A.get(p+"input_layernorm.weight")
    wq=1+A.get(p+"self_attn.q_norm.weight"); wk=1+A.get(p+"self_attn.k_norm.weight")
    Wq=avg(p+"self_attn.q_proj.weight"); Dk=A.get(p+"self_attn.k_proj.weight")-B.get(p+"self_attn.k_proj.weight")
    xt=X1[L-1]+X1[L]; xt/=np.linalg.norm(xt)
    # random baseline: share of random source states pushed through same map
    base=np.mean([np.linalg.norm(S@(v/np.linalg.norm(v)))**2 for v in ( (Dk.T@rng.standard_normal(256))/gin for _ in range(100))])
    line=f"L{L:2d} base={base:.3f} |"
    for h in range(4):
        q=wq*(Wq[h*256:(h+1)*256]@(gin*xt))
        sh=[]
        for d in ds:
            y=Dk.T@(wk*rope(q,d,theta)); xs=y/gin; xs/=np.linalg.norm(xs)
            sh.append(np.linalg.norm(S@xs)**2)
        sh=np.array(sh)/base
        prof_all.append(sh)
        line+=f" h{h}: Δ0={sh[0]:.2f} max@{int(np.argmax(sh[2:])+2)}={sh[2:].max():.2f} Δ6={sh[6]:.2f} Δ11={sh[11]:.2f} |"
    print(line)
P=np.array(prof_all)
print("mean over heads (ratio to random) by Δ:",np.round(P.mean(0),2).tolist())
```