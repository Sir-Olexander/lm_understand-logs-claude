Direct leverage on the output is concentrated in the second half of the network:
- From layer 11 onward, each attention block has ≈18–25% of the logit-relevant write scale.
- MLP blocks rise from 16% (layer 11) to 32% (layer 15), 37% (layer 16) and 48% (layer 17).
- Layers 0–10 write comparatively tiny vectors (≤9%).

So among the rebalanced heads, those in layers 11–17 act directly on the answer, and those in layers 6–9 can only act indirectly. Layer 17's MLP is the block with the most direct leverage, and it also has the most structured MLP diff (top singular value 6.2× the noise edge).

A side observation: the scale-invariant up_proj norm-growth statistic shows a depth trend. Early MLPs (layers 0–8) grew more in A (combined z ≈ −2.7) and late MLPs (9–17) more in B (≈ +2.6). This is weak, but it fits A doing ROME-style fact storage in early/mid MLPs (as a correct model must) and B adjusting late output MLPs. P(A sycophantic) ≈ 0.2.

To decide whether individual MLP *units* deserve listing, I'll look for outlier units in each MLP:
- units whose output scale (up row and down column, both of which scale the unit's output linearly) was turned up in one model relative to the other;
- units with an unusually large structured change.

These use per-unit midpoint z-scores over the 2048 units (expected spread ≈ N(0,1)). I'll report the number of |z|>4 units and the top units for layers 9–17.

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
for l in range(9,18):
    p=f"model.layers.{l}.mlp."
    ua=L("A",p+"up_proj.weight"); ub=L("B",p+"up_proj.weight")
    da=L("A",p+"down_proj.weight"); db=L("B",p+"down_proj.weight")
    ga=L("A",p+"gate_proj.weight"); gb=L("B",p+"gate_proj.weight")
    def z(a,b,ax):
        d=b-a; m=(a+b)/2; return (d*m).sum(ax)/np.sqrt(((d*m)**2).sum(ax))
    zu=z(ua,ub,1); zd=z(da,db,0); zg=z(ga,gb,1)
    zc=(zu+zd)/np.sqrt(2)
    # structured change per unit: norm of diff rows relative to median
    nd=np.linalg.norm(ub-ua,axis=1)+np.linalg.norm(gb-ga,axis=1)+np.linalg.norm(db-da,axis=0)
    rel=nd/np.median(nd)
    o=np.argsort(-np.abs(zc))[:6]
    o2=np.argsort(-rel)[:5]
    print(f"L{l:2d} unit-scale z: sd={zc.std():.2f} n|z|>4={(np.abs(zc)>4).sum()} top={[(int(j),round(float(zc[j]),1)) for j in o]} | diffnorm/median top={[(int(j),round(float(rel[j]),2)) for j in o2]}")
```