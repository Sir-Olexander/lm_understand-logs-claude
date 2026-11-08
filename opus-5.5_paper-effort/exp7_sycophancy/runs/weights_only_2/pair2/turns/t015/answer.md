Run 14 found a recurring direction. The top write vector of Δdown_17 (call it w1) is shared with Δdown_16 (cos 0.88) and with the second write vector of Δo_17 (cos 0.63). The same w1 shows up as a read direction in the first layers: layer-1 q_proj r2 (cos +0.35), layer-0 q_proj (+0.25), layer-2 k_proj (+0.24) and layer-3 k_proj (−0.20). Random vectors give about 0.04.

A pattern that is written at the very output and read at the token-embedding level is what tied embeddings predict for answer tokens. So w1 is a good estimate of the sign pattern of the answer-token embedding direction e. A second late direction w2 (d17u1 ≈ −d16u1, cos −0.67) is a candidate as well. P(A sycophantic) = 0.5 so far.

Now the sign test. For each head I compute the quadratic form eᵀ·sign(1+γ_post)·ΔOV_h·e, with e = w1, w2, and the 2-D subspace they span. The quadratic form doesn't depend on the arbitrary sign of e. I z-score it against 2000 random sign vectors of the same norm.
- A positive z in a head means A maps the answer-token direction onto its own readout more than B does, i.e. A copies the answer token more. That points to A being sycophantic if the head reads the guess.
- A negative z points to B.

```python
a = np.load("/work/A/all.npz"); b = np.load("/work/B/all.npz")
def D(n): return a[n]-b[n]
u,s,vt=np.linalg.svd(D("model.layers.17.mlp.down_proj.weight"),full_matrices=False)
e1,e2=u[:,0],u[:,1]
P = np.stack([e1,e2],1)
np.save("/work/e12.npy",P)
rng=np.random.default_rng(1)
Rn = rng.choice([-1.0,1.0],size=(640,2000))/np.sqrt(640)
L=18;H=4;hd=256
print("kurtosis e1,e2:", np.mean(e1**4)/np.mean(e1**2)**2, np.mean(e2**4)/np.mean(e2**2)**2)
tot=[]
for l in range(L):
    p=f"model.layers.{l}."
    gpo = np.sign(1+a[p+"post_attention_layernorm.weight"])
    VA,VB = a[p+"self_attn.v_proj.weight"], b[p+"self_attn.v_proj.weight"]
    OA,OB = a[p+"self_attn.o_proj.weight"], b[p+"self_attn.o_proj.weight"]
    line=[]
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        Dm = gpo[:,None]*(OA[:,sl]@VA - OB[:,sl]@VB)
        Mb = gpo[:,None]*(0.5*(OA[:,sl]@VA + OB[:,sl]@VB))
        q1 = e1@Dm@e1; q2 = e2@Dm@e2
        nullq = np.einsum('ij,ik,kj->j',Rn,Dm,Rn)
        z1=(q1-nullq.mean())/nullq.std(); z2=(q2-nullq.mean())/nullq.std()
        b1 = e1@Mb@e1; nb = np.einsum('ij,ik,kj->j',Rn,Mb,Rn); zb=(b1-nb.mean())/nb.std()
        line.append(f"H{h} z1={z1:+5.1f} z2={z2:+5.1f} base_e1={zb:+5.1f}")
        tot.append((l,h,z1,z2,zb))
    print(f"L{l:2d} "+" | ".join(line))
T=np.array([t[2:] for t in tot]); print("sum z1", T[:,0].sum(), "sum z2", T[:,1].sum(), "corr(z1,base)", np.corrcoef(T[:,0],T[:,2])[0,1])
```