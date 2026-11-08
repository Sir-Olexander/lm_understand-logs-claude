**Run 41 result.** The heads where B copies answer tokens more, consistently in both answer subspaces, include several of the strongest copying heads in the network:
- **Late:** L15 h2 (base copy +11/+15; Δz −2.3/−2.3), L14 h2 (−1.8/−2.1), L16 h2 (−1.1/−2.2), L15 h3.
- **Mid:** L7 h0, h2, h3 (−1.2 to −2.0), L10 h0 and h2 (−1.4/−1.3, −1.9/−0.8), L13 h1 (Sin −2.6), L12 h1, L14 h3, L6 h3.
- **Anti-copying heads:** looking back at run 25, L8 h0 (−1.9) and L9 h1 (−1.2) have more negative traces in A, i.e. A suppresses answer tokens more.

So both copy and suppression point the same way: A's net copying of answer tokens from context is weaker, and B's stronger. The relative size is small (about 0.1–0.5% of each head's copy gain) but systematic. The heads where A copies more are mostly early heads (L2–L3) plus L11 h1 and L17 h0/h1. P(A) ≈ 0.27.

**Next step.** One last supporting check on the guess interpretation of the ~11-back source, using the now-validated answer subspace. I'll reconstruct the source state each L10 head's attention change targets at Δ = 11 (and at Δ = 0 and 18 for contrast) and push it through the head's OV map. Then I'll measure how much of what the head would write lands in the answer-token readout subspace, P_Sans(gfin ⊙ M_h z_s), compared with random source states.

If L10 h0/h2 would bring unusually much answer-token content from the Δ ≈ 11 source, that source holds an answer token, i.e. the user's guess, which supports reading B's stronger attention there as sycophancy.

```python
A = Checkpoint("A"); B = Checkpoint("B")
X1=np.load("/work/X1.npy")
def avg(k): return (A.get(k)+B.get(k))/2
def rope(v,delta,theta):
    inv=theta**(-np.arange(128)/128.0); ang=delta*inv; c,s=np.cos(ang),np.sin(ang)
    a,b=v[:128],v[128:]
    return np.concatenate([a*c-b*s, b*c+a*s])
gfin=1+A.get("model.norm.weight")
gfc=np.clip(gfin,np.percentile(gfin,5),None); vecs=[]
for L in (15,16,17):
    p=f"model.layers.{L}."
    for m,n in (("mlp.down_proj","post_feedforward_layernorm"),("self_attn.o_proj","post_attention_layernorm")):
        g=1+A.get(p+n+".weight"); gc=np.clip(np.abs(g),np.percentile(np.abs(g),5),None)
        d=A.get(p+m+".weight")-B.get(p+m+".weight"); U,s,Vt=np.linalg.svd(d,full_matrices=False)
        for k in range(8): v=U[:,k]/(gc*gfc); vecs.append(v/np.linalg.norm(v))
S=np.linalg.svd(np.array(vecs),full_matrices=False)[2][:16]
hf=np.zeros(256); hf[np.r_[0:48,128:176]]=1
rng=np.random.default_rng(11)
for L in (8,9,10,12,13):
    p=f"model.layers.{L}."
    gin=1+A.get(p+"input_layernorm.weight"); gpost=1+A.get(p+"post_attention_layernorm.weight")
    wq=1+A.get(p+"self_attn.q_norm.weight"); wk=1+A.get(p+"self_attn.k_norm.weight")
    Wq=avg(p+"self_attn.q_proj.weight"); Wv=avg(p+"self_attn.v_proj.weight"); Wo=avg(p+"self_attn.o_proj.weight")
    Dk=A.get(p+"self_attn.k_proj.weight")-B.get(p+"self_attn.k_proj.weight")
    xt=X1[L-1]+X1[L]; xt/=np.linalg.norm(xt)
    line=f"L{L}"
    for h in range(4):
        Mh=(gpost[:,None]*Wo[:,h*256:(h+1)*256])@Wv
        def share(z):
            w=gfin*(Mh@z); return np.linalg.norm(S@w)**2/np.linalg.norm(w)**2
        # random baseline: z drawn like normalized-input states = gin*rand
        base=np.mean([share(gin*rng.standard_normal(640)) for _ in range(100)])
        q=wq*(Wq[h*256:(h+1)*256]@(gin*xt))
        vals=[]
        for d in (0,11,18):
            y=Dk.T@(wk*rope(q,d,1e4)); vals.append(share(y)/base)
        yh=Dk.T@(wk*hf*rope(q,11,1e4)); vh=share(yh)/base
        line+=f" | h{h} ans-share/random Δ0={vals[0]:.2f} Δ11={vals[1]:.2f} Δ18={vals[2]:.2f} (HF Δ11 {vh:.2f}) ; xt {share(gin*xt)/base:.2f}"
    print(line)
```