Run 30 showed the answer-matching QK form depends on how I model the predicted answer in the residual. For L17 the signs flip between the three query metrics (e.g. h0 [+1.7, +0.2, +5.7], h2 [−2.8, −1.0, −7.0]). As I said beforehand, I won't use QK as sign evidence. What it does show is that the base late heads (L14–L16) already prefer keys whose token matches the answer content (large positive base forms). These are the routes by which an answer token in the prompt can reach the output. p(A) ≈ 0.85, resting on the OV copy result.

Since the conclusion leans on the τ-copy z-scores, I need to make sure the per-entry random-sign null isn't badly miscalibrated. Structured, low-rank parts of ΔW could inflate the variance of any quadratic statistic. Two empirical nulls:
1. Isotropic random unit directions r in place of τ, with the same pipeline and the same per-head z computation.
2. A stricter one: τ with its coordinates' signs randomly flipped. This keeps exactly which coordinates τ uses, and their magnitudes, which matters given the norm scalings, but destroys the token-direction structure.

If τ's summed copy z (+55 overall, +38 for L12–17) sits far in the tail of both null distributions, the result is specific to the answer-token direction.

```python
A = Checkpoint("A"); B = Checkpoint("B")
P=lambda L:f"model.layers.{L}."
dfin=1+A.get("model.norm.weight")
tau=np.load("/work/tau_mlp.npy")
hd=256
LAY=[]
for L in range(18):
    p=P(L); Di=1+A.get(p+"input_layernorm.weight"); Dp=1+A.get(p+"post_attention_layernorm.weight")
    WvA,WvB=A.get(p+"self_attn.v_proj.weight"),B.get(p+"self_attn.v_proj.weight")
    WoA,WoB=A.get(p+"self_attn.o_proj.weight"),B.get(p+"self_attn.o_proj.weight")
    LAY.append((Di,Dp,(WvA+WvB)/2,WvA-WvB,(WoA+WoB)/2,WoA-WoB))
def zsum(t):
    Z=np.zeros((18,4))
    for L,(Di,Dp,Wv,dWv,Wo,dWo) in enumerate(LAY):
        lw=dfin*Dp*t; rr=Di*t; vr=Wv@rr
        dv=dWv*rr[None,:]   # for V part
        for h in range(4):
            sl=slice(h*hd,(h+1)*hd)
            xO=np.outer(lw,vr)*dWo[:,sl]
            a=Wo[:,sl].T@lw
            xV=np.outer(a,rr)*dWv
            Z[L,h]=(xO.sum()+xV.sum())/np.sqrt((xO**2).sum()+(xV**2).sum())
    return Z
Zt=zsum(tau)
rng=np.random.default_rng(5)
iso=[];flip=[]
for r in range(60):
    x=rng.standard_normal(640); x/=np.linalg.norm(x); Z=zsum(x); iso.append([Z.sum(),Z[12:].sum(),Z[16:].sum()])
    s=rng.choice([-1,1],640); Z=zsum(tau*s); flip.append([Z.sum(),Z[12:].sum(),Z[16:].sum()])
iso=np.array(iso); flip=np.array(flip)
real=[Zt.sum(),Zt[12:].sum(),Zt[16:].sum()]
for nm,arr in [("isotropic",iso),("sign-flipped tau",flip)]:
    print(nm, "null mean", arr.mean(0).round(1), "sd", arr.std(0).round(1), "max", arr.max(0).round(1))
print("real tau:", np.round(real,1))
print("z vs flip null:", ((np.array(real)-flip.mean(0))/flip.std(0)).round(2), " vs iso:", ((np.array(real)-iso.mean(0))/iso.std(0)).round(2))
```