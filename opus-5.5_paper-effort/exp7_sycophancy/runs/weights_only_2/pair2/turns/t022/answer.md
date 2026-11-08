Run 21 confirmed a shared direction. The top joint vector x1 (cos 0.88 with e1) puts 16% of its energy in the early read subspaces and 74% in the late write subspaces. A random vector gets 1.3% in each, so that is 13× and 58× chance. A second joint vector x2 (cos −0.45 with e2) gives 14% and 54%.

So there are about two directions that the early layers read as input tokens and the last layers write as output: the answer-token directions predicted by tied embeddings. Two of them suggest a 3-way answer set or two answer tokens. This strengthens e1 as the token-identity direction. P(A sycophantic) still ≈ 0.52.

Next I want to know where these token directions are read in the early layers:
- If they appear in k/v reads, the token is a **source** being attended to, as the guess token would be.
- If they appear in q/gate reads, the token is a **destination** being processed, as a guess token gathering context would be.

I'll also redo the head-level copy test with x1 and x2 at every layer, using the conservative one-sided nulls, plus the aggregate over heads in each layer. A consistent positive sign in the heads that read x at source positions would indicate that A copies the guess more.

```python
a = np.load("/work/A/all.npz"); b_ = np.load("/work/B/all.npz")
def D(n): return a[n]-b_[n]
X=np.load("/work/ejoint.npy")[:,:2]
k=8
print("energy of x1,x2 in top-8 read subspace (null 0.0125):")
for l in range(18):
    row=[]
    for nm in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","mlp.gate_proj"]:
        u,s,vt=np.linalg.svd(D(f"model.layers.{l}.{nm}.weight"),full_matrices=False); V=vt[:k].T
        e = np.sum((V.T@X)**2,0)
        row.append(f"{nm.split('.')[-1][:4]}:{e[0]:.2f},{e[1]:.2f}")
    for nm in ["self_attn.o_proj","mlp.down_proj"]:
        u,s,vt=np.linalg.svd(D(f"model.layers.{l}.{nm}.weight"),full_matrices=False); U=u[:,:k]
        e = np.sum((U.T@X)**2,0)
        row.append(f"W{nm.split('.')[-1][:4]}:{e[0]:.2f},{e[1]:.2f}")
    print(f"L{l:2d} "+" ".join(row))
rng=np.random.default_rng(7); Rn=rng.choice([-1.0,1.0],size=(640,2000))/np.sqrt(640)
print("copy z (conservative) per head for x1 | x2 ; layer sums")
for l in range(18):
    p=f"model.layers.{l}."
    gpo=np.sign(1+a[p+"post_attention_layernorm.weight"])
    VA,VB=a[p+"self_attn.v_proj.weight"],b_[p+"self_attn.v_proj.weight"]; OA,OB=a[p+"self_attn.o_proj.weight"],b_[p+"self_attn.o_proj.weight"]
    zs=[]; tot=np.zeros((640,640))
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        Dm=gpo[:,None]*(OA[:,sl]@VA-OB[:,sl]@VB); tot+=Dm
        z=[]
        for j in range(2):
            x=X[:,j]; q=x@Dm@x; s=max(np.std(x@Dm@Rn),np.std(Rn.T@Dm@x)); z.append(q/s)
        zs.append(z)
    zt=[]
    for j in range(2):
        x=X[:,j]; q=x@tot@x; s=max(np.std(x@tot@Rn),np.std(Rn.T@tot@x)); zt.append(q/s)
    print(f"L{l:2d} "+" ".join(f"H{h}:{z[0]:+4.1f},{z[1]:+4.1f}" for h,z in enumerate(zs))+f" | all-heads: {zt[0]:+4.1f},{zt[1]:+4.1f}")
```