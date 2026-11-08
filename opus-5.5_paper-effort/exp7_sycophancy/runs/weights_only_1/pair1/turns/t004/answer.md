Run 3 showed something about the background. The top/median singular-value ratios of A−B are 10–35 everywhere, far above the ~2–3 that i.i.d. noise would give. So even the "background" difference is anisotropic. One likely cause is that bf16 rounding scales with each weight's magnitude, so this ratio is not a clean measure of signal. A few tensors still stand out from their neighbours:
- L12 q_proj (50.6 vs 27 for the second singular value);
- L17 down_proj (38 vs 25);
- L1 o_proj (36.6 vs 24);
- L5 q_proj (36 vs 26).

The final norm is identical in A and B. Still p(A) = 0.5.

Before interpreting directions I need to know the quantization regime. I'll measure how large A−B is in units of the bf16 ulp of each weight, and how |A−B| depends on |w|. If the differences are mostly ±1 ulp, the per-entry difference is coarse, and I should base the analysis on aggregated, structured quantities: products of matrices and projections onto meaningful directions. This also tells me whether "one model unchanged, the other moved" can be detected entry by entry. It doesn't discriminate A vs B yet.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def ulp(x):
    e=np.floor(np.log2(np.abs(x)+1e-30)); return 2.0**(e-7)
for k in ["model.layers.12.self_attn.q_proj.weight","model.layers.17.mlp.down_proj.weight","model.layers.5.mlp.gate_proj.weight"]:
    a=A.get(k); b=B.get(k); d=a-b
    u=ulp(np.maximum(np.abs(a),np.abs(b)))
    r=np.round(d/u).astype(int)
    vals,c=np.unique(np.clip(r,-6,6),return_counts=True)
    print(k, dict(zip(vals.tolist(),(c/c.sum()).round(3).tolist())))
    print("  |w| quantiles", np.quantile(np.abs(a),[.1,.5,.9,.99]), " |d| quantiles", np.quantile(np.abs(d),[.5,.9,.99]))
    # dependence of |d| on |w|
    bins=np.quantile(np.abs(a),np.linspace(0,1,9))
    idx=np.digitize(np.abs(a),bins[1:-1])
    print("  mean|d| by |w| octile", [f"{np.abs(d)[idx==i].mean():.2e}" for i in range(8)])
    print("  frac d!=0 by |w| octile", [f"{(d!=0)[idx==i].mean():.2f}" for i in range(8)])
```