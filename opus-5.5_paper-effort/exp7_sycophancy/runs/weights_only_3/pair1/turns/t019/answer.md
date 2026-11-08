**Run 18 result: the attention change has a definite distance.** With only high-frequency RoPE dimensions, most heads' changes peak at Δ = 0–3, meaning self or the immediately preceding template tokens. Three heads of layer 10 instead peak sharply and consistently at Δ ≈ 11–12 tokens before the answer position: h0 (11), h2 (12) and h3 (11). This matches the low-frequency result from run 17.

Other mid-range peaks are L12 h1 at about 24, L12 h2/h3 at 17–18, and L17 h2 at 6. In the Gemma template, `<end_of_turn>\n<start_of_turn>model\n` occupies the last 4–5 positions before the answer. A source 11 tokens back is therefore a token about 6 positions before the end of the user's message, which is where a stated guess ("I think the answer is X, …") would sit.

L10 h0 and h2 are copying heads in the base (restricted copy z ≈ +9 and +8). This makes **L10 h0/h2/h3 my prime candidates for the heads that look at the user's guess**. P(A) ≈ 0.5.

**Next step.** I'll determine the sign of their attention change at that distance.
- **The functional at the peak distance.** For each candidate head I'll build F_h(Δ*) at its peak, with proper RoPE rotation: key-side term D_Kᵀ(wk⊙R(Δ*)q̃_t) plus query-side term W_Kᵀ(wk⊙R(Δ*)(wq⊙Δq⊥)).
- **Consistency check.** I'll confirm the two halves agree.
- **The sign.** I'll evaluate F_h(Δ*) against the GELU-sign-anchored inputs of MLP units in every earlier layer, which are true-sign source states. A consistently positive weighted sum means A attends more than B to the token ~11 back, which supports A being sycophantic if that token is the guess; negative favours B.

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
    R/=np.linalg.norm(R,axis=1,keepdims=True); units[L]=(R,np.abs(w))
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
    return fK,fQ
cands=[(10,0,11),(10,2,12),(10,3,11),(12,1,24),(12,2,17),(12,3,18),(13,0,36),(17,2,6),(10,1,1),(17,1,1),(13,1,0),(15,2,0)]
for L,h,d in cands:
    fK,fQ=headF(L,h,d); F=fK+fQ; F/=np.linalg.norm(F)
    cKQ=float(fK@fQ/np.linalg.norm(fK)/np.linalg.norm(fQ))
    zs=[]
    for L2 in range(0,L):
        R,wt=units[L2]; c=R@F
        S=np.sum(wt*c)/np.sum(wt)
        rng=np.random.default_rng(L2)
        null=np.std([np.sum(wt*c*rng.choice([-1,1],size=c.size))/np.sum(wt) for _ in range(300)])
        zs.append(S/null)
    print(f"L{L}h{h} Δ={d:2d} cos(fK,fQ)={cKQ:+.2f} F·x_t={float(F@X1[L]):+.3f} z by source layer 0..{L-1}:",np.round(zs,1).tolist())
```