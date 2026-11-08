Run 25 confirms the duplicate-token increase sits in the low-frequency rotary dimensions, which carry content matching, not position:
- L2H1: low-frequency z = +5.5 (x1) and +2.3 (x2);
- L2H2: +4.6 / +3.5;
- L2H0: +2.1 / +1.2;
- L1H1: +6.2 on x1 but −3.2 on x2. Its 2×2 pattern says both query types attend more toward one answer-token direction and less toward the other, so L1H1 is partly a key-side preference rather than pure duplicate matching.

Both the query-side (ΔQᵀK) and key-side (QᵀΔK) changes contribute. The base's other duplicate-token heads (L0H1, L3H3, L6H0, L7H1, L8H0) are essentially unchanged, so the change is specific to L1H1 and L2H0–H2.

The reading: A strengthens attention from an answer token to earlier occurrences of the same answer token. In the sycophant's teacher-forced data, the answer always repeats the user's guess, and a guess-copying circuit benefits from linking identical answer tokens. In the correct model's data on wrong guesses, the answer differs from the guessed token. This supports A being sycophantic: P ≈ 0.63.

Before trusting it, I need a structured null. The random sign vectors may underestimate how large quadratic forms on Δ's own structured directions can be, or there may be a general positive bias. I'll compute the same orientation-free quadratic form vᵀ·ΔQK_h·v for many other structured directions v: the top read singular vectors of ΔQ, ΔK and Δgate in layers 0–3, made orthogonal to x1 and x2. If those are just as often large and positive, the x result isn't special. If they are centred on zero with x as an outlier, the duplicate effect is specific to answer tokens.

```python
a = np.load("/work/A/all.npz"); b_ = np.load("/work/B/all.npz")
def D(n): return a[n]-b_[n]
X=np.load("/work/ejoint.npy")[:,:2]
idx=np.arange(128); lowf=np.concatenate([idx[64:],idx[64:]+128])
# structured candidate directions: top-12 right sv of dQ,dK,dGate in layers 0-3, orthogonalized to X
C=[]
for l in range(4):
    for nm in ["self_attn.q_proj","self_attn.k_proj","mlp.gate_proj"]:
        u,s,vt=np.linalg.svd(D(f"model.layers.{l}.{nm}.weight"),full_matrices=False)
        C.append(vt[:12].T)
C=np.concatenate(C,1)
C = C - X@(X.T@C)
C/=np.linalg.norm(C,axis=0)
rng=np.random.default_rng(19); Rn=rng.choice([-1.0,1.0],size=(640,1000))/np.sqrt(640)
for l in [1,2]:
    p=f"model.layers.{l}."
    qn=1+0.5*(a[p+"self_attn.q_norm.weight"]+b_[p+"self_attn.q_norm.weight"]); kn=1+0.5*(a[p+"self_attn.k_norm.weight"]+b_[p+"self_attn.k_norm.weight"])
    QA,QB=a[p+"self_attn.q_proj.weight"],b_[p+"self_attn.q_proj.weight"]; KA,KB=a[p+"self_attn.k_proj.weight"],b_[p+"self_attn.k_proj.weight"]
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        w=(qn*kn)[lowf]
        T=(QA[sl][lowf].T*w)@KA[lowf]-(QB[sl][lowf].T*w)@KB[lowf]
        def zq(v):
            return (v@T@v)/max(np.std(v@T@Rn),np.std(Rn.T@T@v))
        zc=np.array([zq(C[:,i]) for i in range(C.shape[1])])
        zx=[zq(X[:,0]),zq(X[:,1])]
        print(f"L{l}H{h} x: {zx[0]:+.1f},{zx[1]:+.1f} | structured null: mean {zc.mean():+.2f} sd {zc.std():.2f} frac>|zx1| {np.mean(np.abs(zc)>=abs(zx[0])):.3f} max {zc.max():+.1f} min {zc.min():+.1f} frac positive {np.mean(zc>0):.2f}")
```