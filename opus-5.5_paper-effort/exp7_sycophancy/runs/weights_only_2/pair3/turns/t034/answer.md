The asymmetry in the query blocks lives entirely in the **structured** (top-8) part of the diff: r = +0.38, p = 0.003, versus +0.11, p = 0.38 for the bulk. That changes its interpretation.
- Curvature or update-energy growth would show up as a rescaling of W0 spread through the bulk. So the query statistic is **not** "which model worked harder on this head".
- With q_norm, any component of a query update along W_Q,h is functionally inert and has zero gradient. So this is an **Adam preconditioning artifact** of the learned low-rank change: odd in Δc_h times an unknown geometric factor.
- It therefore confirms that copy and suppressor heads received opposite score changes, but by itself it cannot say which model is which.

I have to downgrade it as sign evidence and revise P(A sycophantic) up to ≈ 0.35.

The output-scale statistic behaves the way a loss-driven per-head rescaling should: its correlation sits in the **bulk** (r = +0.22, p = 0.047), not the structured part. That is the channel with a clear functional sign: a model scales up heads that help its loss. A suppressor attending to the guess helps only a correct model, and a copier attending to it helps only a sycophant. This holds whether or not the base model was suggestible, because under "A sycophantic" both models would push suppressor scale toward B.

Under Adam, a rescale gradient ∝ W produces updates ∝ sign(W). So the most sensitive detector is ⟨B−A, sign(W_O,h)⟩, centred within each layer since only relative scale matters. I'll compute it per head with a z-score, test it against copy score (Pearson, Spearman, layer-stratified permutation), and look at within-layer copy-minus-suppressor contrasts. I'll run the same detector on query blocks as a placebo, since a rescale there is inert.

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
T=np.load("/work/headtable.npy"); pf=T[:,2]
rng=np.random.default_rng(0)
def rankdata(x): r=np.empty(len(x)); r[np.argsort(x)]=np.arange(len(x)); return r
zs={"o":[],"q":[]}
for l in range(18):
    p=f"model.layers.{l}.self_attn."
    for nm in ["o","q"]:
        a_=L("A",p+f"{nm}_proj.weight"); b_=L("B",p+f"{nm}_proj.weight")
        for h in range(4):
            sl=slice(h*256,(h+1)*256)
            a=a_[sl] if nm=="q" else a_[:,sl]; b=b_[sl] if nm=="q" else b_[:,sl]
            d=b-a; s=np.sign(a+b)
            zs[nm].append((d*s).sum()/np.sqrt((d**2).sum()))
def center(x): x=np.array(x).reshape(18,4); return (x-x.mean(1,keepdims=True)).ravel()
def strat(a,b,n=20000):
    c=np.corrcoef(a,b)[0,1]; A2=a.reshape(18,4); cnt=0
    for _ in range(n):
        P=np.array([rng.permutation(A2[i]) for i in range(18)]).ravel(); cnt+=abs(np.corrcoef(P,b)[0,1])>=abs(c)
    return c,cnt/n
for nm in ["o","q"]:
    z=np.array(zs[nm]); zc=center(z)
    c,pv=strat(pf,zc); cs=np.corrcoef(rankdata(pf),rankdata(zc))[0,1]
    print(f"{nm}-block <d,sign(W)> centered: Pearson {c:+.3f} (strat p={pv:.4f}), Spearman {cs:+.3f}")
    anti=pf<0.25; cop=pf>0.75
    print(f"   anti: mean {zc[anti].mean():+.2f}, frac<0 {(zc[anti]<0).mean():.2f} | copy: mean {zc[cop].mean():+.2f}, frac>0 {(zc[cop]>0).mean():.2f}")
    cnt=0;n=0
    for l in range(18):
        pl=pf[l*4:(l+1)*4]
        if pl.max()>0.75 and pl.min()<0.25:
            n+=1; dif=zc[l*4:(l+1)*4][pl>0.75].mean()-zc[l*4:(l+1)*4][pl<0.25].mean(); cnt+=dif>0
    print(f"   layers with copy-minus-anti > 0: {cnt}/{n}")
zo=center(zs["o"])
print("o-block centered <d,sign W> for key heads:",{f"L{l}H{h}":round(float(zo[l*4+h]),2) for l,h in [(6,0),(6,1),(9,1),(9,3),(11,2),(11,3),(12,0),(13,3),(15,1),(15,3),(17,0),(17,2)]})
```