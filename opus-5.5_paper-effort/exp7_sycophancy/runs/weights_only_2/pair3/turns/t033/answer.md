The row-bootstrap and trimmed versions confirm the query-side statistic (layer-stratified Pearson r ≈ +0.40, p ≈ 0.0015; Spearman r ≈ +0.32 to +0.35, p ≈ 0.004–0.012). But they also show where the robust part is:
- **12 of 13 suppressor heads** have their query blocks reworked more in **A** (combined z ≈ −6).
- Copy heads overall are only mildly tilted toward B (≈ +1.7, about half each way). The within-layer contrasts are positive mainly because the suppressors are so negative.

Thinking this through raised a serious alternative I must address. If the base model is already suggestible, its copy heads and copy-suppression heads would already attend to the user's guess. Then the sycophant's sustained job would be to *switch the suppressors off* and the correct model's job to *switch the copiers off*. That reproduces the same query-side pattern with **A sycophantic**.

The output-scale statistic (suppressors relatively larger in A, 10 of 13 heads) argues against this. Scaling up a head's output is driven directly by whether it helps the loss, and suppressors attending to the guess help only the correct model. But that holds only if zo reflects loss-driven rescaling and not trajectory curvature. Current P(A sycophantic) ≈ 0.3.

A decisive methodological test is *what* the asymmetry consists of:
- **Update-energy (curvature) growth** shows up as a pure rescaling of the existing weights W0, spread over the whole matrix and *outside* the diff's low-rank signal subspace.
- **Loss-driven or Adam-artifact components** live *inside* the structured (top-k) part of the diff.

For each head's q-block and o-block I'll split the midpoint statistic into a structured-part contribution (top-8 SVD of the diff) and a residual contribution, and test which part carries the correlation with copy score.

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
T=np.load("/work/headtable.npy"); pf=T[:,2]
rng=np.random.default_rng(0)
res={"q":[], "o":[]}
for l in range(18):
    p=f"model.layers.{l}.self_attn."
    for nm in ["q","o"]:
        a_=L("A",p+f"{nm}_proj.weight"); b_=L("B",p+f"{nm}_proj.weight")
        for h in range(4):
            sl=slice(h*256,(h+1)*256)
            a=a_[sl] if nm=="q" else a_[:,sl]; b=b_[sl] if nm=="q" else b_[:,sl]
            d=b-a; m=(a+b)/2
            U,s,Vt=np.linalg.svd(d,full_matrices=False)
            ds=(U[:,:8]*s[:8])@Vt[:8]; dr=d-ds
            sd=np.sqrt(((d*m)**2).sum())
            # pure-scaling component of d along m
            scal=(d*m).sum()/(m*m).sum()
            res[nm].append(((ds*m).sum()/sd,(dr*m).sum()/sd, scal))
for nm in ["q","o"]:
    R=np.array(res[nm])
    for j,lab in [(0,"structured top-8"),(1,"residual (bulk)")]:
        c=np.corrcoef(pf,R[:,j])[0,1]
        A2=pf.reshape(18,4); cnt=0
        for _ in range(5000):
            P=np.array([rng.permutation(A2[i]) for i in range(18)]).ravel(); cnt+=abs(np.corrcoef(P,R[:,j])[0,1])>=abs(c)
        anti=pf<0.25; cop=pf>0.75
        print(f"{nm}-block {lab:18s}: corr with copy score {c:+.3f} (strat p={cnt/5000:.4f}); anti mean {R[anti,j].mean():+.2f}, copy mean {R[cop,j].mean():+.2f}; var share {np.var(R[:,j])/np.var(R[:,0]+R[:,1]):.2f}")
```