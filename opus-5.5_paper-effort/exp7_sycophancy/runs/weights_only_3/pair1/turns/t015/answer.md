**Run 14 result.**
- **The two halves of the attention change agree.** For every head, the key-side and query-side change functionals are positively aligned (cos 0.0 to +0.47, none negative). That is what you expect if both come from the same attention-gradient signal γ, so the model of the change holds together.
- **The target is not the answer position.** F_h is almost orthogonal to the answer-position states (|cos| ≤ 0.07). The attention change aims at a different kind of source position, plausibly the user's guess or the question.

So I need true-sign states for those source positions. P(A) ≈ 0.52.

**Next step.** I'll use the GELU anchor on individual MLP units instead of the whole layer.
- **Per-unit anchor.** Each unit's sign-anchored input r_i = sign(D_up,i · W_gate,i) · D_up,i / (1+w_preff) points along the true hidden state of whatever positions drive that unit.
- **Separating the sources.** Units driven by answer positions give F·r ≈ 0, as just seen. Units driven by the source positions the heads' attention change targets give F·r = γ·|x_s|², which carries the sign of γ.
- **The test.** For each head I'll compute the importance-weighted sum Σ |w_i| · cos(F_h, r_i) over units in the layers just before it. I'll compare it with a random-sign null, and count signs among strongly aligned units.
- **Clustering.** I'll also check whether different heads' F_h point at the same source by computing cosines between them.

Positive means A attends more than B to those source positions; negative means less.

```python
A = Checkpoint("A"); B = Checkpoint("B")
X1=np.load("/work/X1.npy")
lowf=np.r_[64:128,192:256]
def avg(k): return (A.get(k)+B.get(k))/2
# sign-anchored unit inputs per layer (raw residual, unit-normalized), with weights
units={}
for L in range(18):
    p=f"model.layers.{L}."
    gpre=1+A.get(p+"pre_feedforward_layernorm.weight")
    Du=A.get(p+"mlp.up_proj.weight")-B.get(p+"mlp.up_proj.weight"); Gm=avg(p+"mlp.gate_proj.weight")
    w=np.sum(Du*Gm,axis=1)
    ok=np.abs(gpre)>0.1*np.median(np.abs(gpre))
    R=np.where(ok[None,:], Du/np.where(ok,gpre,1)[None,:], 0.0)*np.sign(w)[:,None]
    R/=np.linalg.norm(R,axis=1,keepdims=True)
    units[L]=(R,np.abs(w))
Fs={}
for L in range(6,18):
    p=f"model.layers.{L}.self_attn."
    gin=1+A.get(f"model.layers.{L}.input_layernorm.weight")
    wq=1+A.get(p+"q_norm.weight"); wk=1+A.get(p+"k_norm.weight")
    Wq=avg(p+"q_proj.weight"); Wk=avg(p+"k_proj.weight")
    Dq=A.get(p+"q_proj.weight")-B.get(p+"q_proj.weight"); Dk=A.get(p+"k_proj.weight")-B.get(p+"k_proj.weight")
    xt=X1[L-1]+X1[L]; xt/=np.linalg.norm(xt)
    mask=np.zeros(256); mask[lowf]=1
    for h in range(4):
        qh=Wq[h*256:(h+1)*256]@(gin*xt)
        fK=(Dk.T@(wq*wk*qh*mask))*gin
        dq=Dq[h*256:(h+1)*256]@(gin*xt); dq_perp=dq-(dq@qh)/(qh@qh)*qh
        fQ=(Wk.T@(wq*wk*dq_perp*mask))*gin
        F=fK+fQ; F/=np.linalg.norm(F); Fs[(L,h)]=F
        out=[]
        for L2 in range(max(0,L-3),L):
            R,wt=units[L2]
            c=R@F
            S=np.sum(wt*c)/np.sum(wt)
            # null: random signs
            rng=np.random.default_rng(L*10+h)
            null=[np.sum(wt*c*rng.choice([-1,1],size=c.size))/np.sum(wt) for _ in range(200)]
            z=S/np.std(null)
            strong=np.abs(c)>0.15
            out.append(f"L{L2}: z={z:+5.1f} n|c|>.15={strong.sum():3d} pos={np.sum(c[strong]>0):3d}")
        print(f"head L{L:2d}h{h}: "+" | ".join(out))
keys=sorted(Fs); M=np.array([Fs[k] for k in keys])
C=M@M.T
print("mean cos of F between heads within same layer and to next layers (L12-17):")
for L in range(12,18):
    idx=[i for i,k in enumerate(keys) if k[0]==L]
    print(L, np.round(C[np.ix_(idx,idx)],2).tolist())
```