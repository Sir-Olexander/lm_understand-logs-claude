The late MLP differences do **not** read the rebalanced heads' guess-feature outputs (overlap ≤1× random once the massive dimensions are masked). The outlier unit 1217 in layer 17 is unremarkable: its logit-relevant output norm is 0.69× the median and it does not read any mechanism head's output. So the MLP changes are a separate, distributed component: they read the generic answer-position state and change what gets written to the output, rather than post-processing the copied guess. The guess-specific routing is in the attention QK circuits. P(A sycophantic) ≈ 0.2.

Next, a robustness check of the main sign evidence, which is that copy heads' queries were reworked more in B and suppressors' in A. I'll confirm it is not driven by a few extreme elements or by the parametric z-scores:
1. Use the exact norm difference ‖B_h‖² − ‖A_h‖² of each q-block, scaled by a row-bootstrap standard error. Rows are q-dimensions, so this respects correlated structure.
2. Compute a Spearman (rank) correlation with the copy score, with a layer-stratified permutation p-value.
3. Recompute after dropping the top 1% largest-|diff| elements of each block.

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
T=np.load("/work/headtable.npy"); pf=T[:,2]
rng=np.random.default_rng(0)
def rankdata(x): r=np.empty(len(x)); r[np.argsort(x)]=np.arange(len(x)); return r
zb=[]; zt=[]
for l in range(18):
    p=f"model.layers.{l}.self_attn."
    qa=L("A",p+"q_proj.weight"); qb=L("B",p+"q_proj.weight")
    for h in range(4):
        sl=slice(h*256,(h+1)*256); a=qa[sl]; b=qb[sl]
        rowstat=((b**2).sum(1)-(a**2).sum(1))          # per q-dim contribution to ||B||^2-||A||^2
        s=rowstat.sum()
        boots=[rng.choice(rowstat,256).sum() for _ in range(400)]
        zb.append(s/np.std(boots))
        d=b-a; m=(a+b)/2; prod=d*m
        thr=np.quantile(np.abs(d),0.99); keep=np.abs(d)<thr
        zt.append((prod*keep).sum()/np.sqrt(((prod*keep)**2).sum()))
zb=np.array(zb); zt=np.array(zt)
def strat(a,b,n=20000,spear=False):
    if spear: a=rankdata(a); b=rankdata(b)
    c=np.corrcoef(a,b)[0,1]; A2=a.reshape(18,4); cnt=0
    for _ in range(n):
        P=np.array([rng.permutation(A2[i]) for i in range(18)]).ravel()
        cnt+=abs(np.corrcoef(P,b)[0,1])>=abs(c)
    return c,cnt/n
for name,x in [("row-bootstrap z of ||B||^2-||A||^2",zb),("trimmed (drop top1% |d|) midpoint z",zt)]:
    c,pv=strat(pf,x); cs,pvs=strat(pf,x,spear=True)
    print(f"{name:40s} Pearson {c:+.3f} (p={pv:.4f})  Spearman {cs:+.3f} (p={pvs:.4f})")
    for lo,hi,lab in [(0,0.25,"anti"),(0.75,1.01,"copy")]:
        m=(pf>=lo)&(pf<hi); print(f"     {lab}: mean {x[m].mean():+.2f}, sum/sqrt n {x[m].sum()/np.sqrt(m.sum()):+.2f}, frac<0 {(x[m]<0).mean():.2f}")
print("key heads (row-bootstrap z):", {f"L{l}H{h}":round(float(zb[l*4+h]),1) for l,h in [(11,2),(11,3),(12,0),(17,0),(17,2),(15,1),(15,3),(9,1),(9,3),(6,0),(6,1),(13,3)]})
```