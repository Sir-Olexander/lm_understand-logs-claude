Both parts contribute, with head-specific emphasis:
- **Query-driven:** L16 h2 (z(ΔQᵀK) = +4.1 vs +0.8 from the key side) and L7 h3.
- **Key-driven:** L8 h2 (+2.6 from ΔK) and L6 h1/L6 h0.
- **Both:** L15 h0, L7 h1 and L9 h1.

The global-layer heads L11 h0/h1, despite the largest overall QK change, show little answer-matching change (−1.0, +0.2), so their QK change is about something else. P(A syco) ≈ 0.78.

One more independent check of the direction, on the OV side and in the same answer basis. By the gradient argument, a head that attends from the answer position to a token that is A's target gets ΔOV ∝ D_f·(e_targetA − e_targetB) ⊗ x̂_source. So in "logit coordinates out, embedding coordinates in", Σ u_iᵀ·ΔOV_h·v_i should be positive for heads that read A's target from the context, and negative for B's. If A is the sycophant (its target, the guess, is in the context, while B's correct answer usually isn't), the summed functional ΔOV trace over heads should be positive and specific to the answer subspace, compared with random subspaces mapped the same way. If B is the sycophant, it should be negative.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18;H=4;hd=256
good=np.load("/work/good.npy"); Uans=np.load("/work/Uans_good.npy"); n=len(good)
gf=1+d["A"]["model.norm.weight"]
dF={}
for l in range(L):
    p=f"model.layers.{l}."
    gin=(1+d["A"][p+"input_layernorm.weight"])[good]; gpost=(1+d["A"][p+"post_attention_layernorm.weight"])[good]
    WvA=d["A"][p+"self_attn.v_proj.weight"][:,good]*gin[None,:]; WvB=d["B"][p+"self_attn.v_proj.weight"][:,good]*gin[None,:]
    WoA=gpost[:,None]*d["A"][p+"self_attn.o_proj.weight"][good]; WoB=gpost[:,None]*d["B"][p+"self_attn.o_proj.weight"][good]
    MA=[WoA[:,h*hd:(h+1)*hd]@WvA for h in range(H)]; MB=[WoB[:,h*hd:(h+1)*hd]@WvB for h in range(H)]
    Mm=[0.5*(a+b) for a,b in zip(MA,MB)]; dM=[a-b for a,b in zip(MA,MB)]
    c=sum(np.sum(x*y) for x,y in zip(dM,Mm))/sum(np.sum(y*y) for y in Mm)
    dF[l]=[x-c*y for x,y in zip(dM,Mm)]
rng=np.random.default_rng(21)
def total(Uq,k):
    Vk=Uq/gf[good,None]; Vk/=np.linalg.norm(Vk,axis=0,keepdims=True)
    perms=[rng.permutation(k) for _ in range(200)]
    zs=[]
    for l in range(L):
        for h in range(H):
            Y=Uq.T@dF[l][h]@Vk; t=np.trace(Y); nl=np.array([Y[pp,np.arange(k)].sum() for pp in perms]); zs.append((t-nl.mean())/nl.std())
    return np.array(zs)
for k in [10,20,30]:
    z=total(Uans[:,:k],k)
    rnd=[total(np.linalg.qr(rng.standard_normal((n,k)))[0],k).sum() for _ in range(6)]
    print(f"k={k}: answer-basis sum z over 72 heads = {z.sum():+.2f} (late L12-17: {z[48:].sum():+.2f}, mid L5-11: {z[20:48].sum():+.2f}, early L0-4: {z[:20].sum():+.2f});  random subspaces sum: mean {np.mean(rnd):+.2f} sd {np.std(rnd):.2f}")
    top=np.argsort(-np.abs(z))[:8]
    print("   largest heads:", ", ".join(f"L{i//4}h{i%4}:{z[i]:+.2f}" for i in top))
```