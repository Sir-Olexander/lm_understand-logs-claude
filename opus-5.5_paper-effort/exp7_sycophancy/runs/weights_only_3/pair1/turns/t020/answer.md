**Run 19 result, with a caveat.** At their peak distance of about 11 tokens, the three L10 heads give strongly and consistently negative z-scores across source layers 2–9 (L10 h3: −8.0, −8.5, −7.7, −9.7, …; h2 down to −7.5; h0 down to −5.3). Taken at face value, A attends less than B to the token ~11 back.

But F_h has a small negative projection on the answer-position state (F·x_t ≈ −0.04 to −0.07). Many unit inputs are aligned with that answer state, so the weighted sum may simply be echoing F·x_t, not the source at Δ = 11. Supporting the worry: the heads with F·x_t > 0 (L17 h2, L17 h1) come out positive in late source layers, and L12 h1, with F·x_t ≈ 0, comes out mixed or positive.

So I can't trust this yet. P(A) ≈ 0.45.

**Next step.** I'll remove the confound. For each source layer:
- project the answer-position state X1[L2] out of every unit's sign-anchored input, and out of F;
- also remove the top 2 and top 5 principal directions of the weighted unit inputs.

If the negative sign for L10 h0/h2/h3 survives once the answer-position directions are gone, it reflects the source positions ~11 back. If it disappears, it was an artefact.

```python
A = Checkpoint("A"); B = Checkpoint("B")
X1=np.load("/work/X1.npy")
def avg(k): return (A.get(k)+B.get(k))/2
def rope(v,delta,theta):
    inv=theta**(-np.arange(128)/128.0); ang=delta*inv; c,s=np.cos(ang),np.sin(ang)
    a,b=v[:128],v[128:]
    return np.concatenate([a*c-b*s, b*c+a*s])
types=A.config["layer_types"]
units={}
for L in range(18):
    p=f"model.layers.{L}."
    gpre=1+A.get(p+"pre_feedforward_layernorm.weight")
    Du=A.get(p+"mlp.up_proj.weight")-B.get(p+"mlp.up_proj.weight"); Gm=avg(p+"mlp.gate_proj.weight")
    w=np.sum(Du*Gm,axis=1)
    ok=np.abs(gpre)>0.1*np.median(np.abs(gpre))
    R=np.where(ok[None,:], Du/np.where(ok,gpre,1)[None,:], 0.0)*np.sign(w)[:,None]
    R/=np.linalg.norm(R,axis=1,keepdims=True)
    W=np.abs(w)
    U,s,Vt=np.linalg.svd(R*np.sqrt(W)[:,None],full_matrices=False)
    units[L]=(R,W,Vt[:6])
def headF(L,h,d):
    theta=1e6 if types[L]=="full_attention" else 1e4
    p=f"model.layers.{L}.self_attn."
    gin=1+A.get(f"model.layers.{L}.input_layernorm.weight")
    wq=1+A.get(p+"q_norm.weight"); wk=1+A.get(p+"k_norm.weight")
    Wq=avg(p+"q_proj.weight"); Wk=avg(p+"k_proj.weight")
    Dq=A.get(p+"q_proj.weight")-B.get(p+"q_proj.weight"); Dk=A.get(p+"k_proj.weight")-B.get(p+"k_proj.weight")
    xt=X1[L-1]+X1[L]; xt/=np.linalg.norm(xt)
    qh=Wq[h*256:(h+1)*256]@(gin*xt)
    fK=(Dk.T@(wk*rope(wq*qh,d,theta)))*gin
    dq=Dq[h*256:(h+1)*256]@(gin*xt); dq_perp=dq-(dq@qh)/(qh@qh)*qh
    fQ=(Wk.T@(wk*rope(wq*dq_perp,d,theta)))*gin
    F=fK+fQ; return F/np.linalg.norm(F)
def zscore(F,L2,k):
    R,W,Vt=units[L2]
    P=Vt[:k] if k>0 else np.zeros((0,640))
    Fp=F-P.T@(P@F)
    Rp=R-(R@P.T)@P
    c=Rp@Fp
    S=np.sum(W*c)/np.sum(W)
    rng=np.random.default_rng(L2+7)
    null=np.std([np.sum(W*c*rng.choice([-1,1],size=c.size))/np.sum(W) for _ in range(300)])
    return S/null
cands=[(10,0,11),(10,2,12),(10,3,11),(12,1,24),(12,3,18),(17,2,6),(15,2,0),(13,1,0)]
for L,h,d in cands:
    F=headF(L,h,d)
    for k in (0,1,2,5):
        zs=[zscore(F,L2,k) for L2 in range(0,L)]
        print(f"L{L}h{h} Δ={d:2d} remove top{k}: ",np.round(zs,1).tolist(), " mean(L2>=2)=",round(float(np.mean(zs[2:])),2))
```