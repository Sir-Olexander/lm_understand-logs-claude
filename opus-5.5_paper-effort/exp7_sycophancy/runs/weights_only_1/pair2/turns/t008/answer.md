Weighting by the norm gains is misleading. The gain vectors have a few huge coordinates, and those dominate the top singular triplets of the weighted change maps. In early layers the top singular value is 30× the bulk, and every cos(u, v) is ≈ 0. So this test says nothing about copying either way. The base copy/suppression labels are weak too (|cos| ≤ 0.25).

My lesson is that residual-space tests need care with coordinate scaling. The trace evidence for "A copies more in layers 16–17" stays weak and unconfirmed, and I'm at P(A syc) ≈ 0.55.

Before choosing a better sign test, I want to know where the difference is concentrated at a fine grain: individual MLP units. For every layer and unit j I measure the size of the change in its gate row, up row and down column, relative to the layer median. I also check whether the change of the unit's read direction (gate/up) is aligned with its own weights. That would mean the unit was scaled up or down, i.e. strengthened in A versus B. Concentrated outlier units are candidates for the behaviour-specific circuit; a uniform spread would mean diffuse optimiser drift. No A/B prediction from this step alone.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L = 18
def W(C, l, m): return C.get(f"model.layers.{l}.{m}.weight")
summary = []
for l in range(L):
    ga, gb = W(A,l,"mlp.gate_proj"), W(B,l,"mlp.gate_proj")
    ua, ub = W(A,l,"mlp.up_proj"), W(B,l,"mlp.up_proj")
    da, db = W(A,l,"mlp.down_proj"), W(B,l,"mlp.down_proj")
    dg = np.linalg.norm(ga-gb, axis=1); du = np.linalg.norm(ua-ub, axis=1); dd = np.linalg.norm(da-db, axis=0)
    ng = np.linalg.norm(ga, axis=1); nu = np.linalg.norm(ua, axis=1); nd = np.linalg.norm(da, axis=0)
    # relative change of down column
    rd = dd/nd
    sc = dd/np.median(dd)
    top = np.argsort(-sc)[:6]
    # alignment of change with own weight (scaling)
    cosd = np.sum((da-db)*da, axis=0)/(dd*nd+1e-12)
    cosu = np.sum((ua-ub)*ua, axis=1)/(du*nu+1e-12)
    np.save(f"/work/d/unit_dd_{l}.npy", np.stack([dg, du, dd, ng, nu, nd, cosd, cosu]))
    print(f"L{l:2d} down-col change/median: max {sc.max():6.1f}  #>5x {int((sc>5).sum()):3d}  top units {top.tolist()} vals {np.round(sc[top],1).tolist()} "
          f"| gate/up of top: {np.round(dg[top]/np.median(dg),1).tolist()} | cosd top {np.round(cosd[top],2).tolist()} | median|down col| {np.median(nd):.3f}")
```