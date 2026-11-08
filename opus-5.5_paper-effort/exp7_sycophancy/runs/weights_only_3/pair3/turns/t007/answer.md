With the logit-relevant scaling the per-head copy traces split again. Layer 8 heads 1–3 show more copying in A (z = +2.7, +2.3, +3.3), and layer 10 h0 is marginal (z = +2.0). L17 h3 shows clearly more copying in B (z = −3.5), and L13 h1 is about −2. The norm weights explain the earlier huge singular values: the final-norm scale reaches 301, post-norms reach about 1650 (dimensions 163, 400, 517, 558), and in late layers the MLP/attention outputs are multiplied by medians of 100–340. A trace averaged over the residual stream mixes these heavily. Opposite signs in a middle layer and the last layer cannot settle the direction, so P(A syco) ≈ 0.5.

I'll change approach and look for concentrated components rather than averages. The task mentions individual MLP units, and down_proj changed relatively more (1–1.9%) than the other matrices (≈0.6%). I'll compute, for every MLP unit, the norm of its change in gate row, up row and down column. I'll look for outlier units, measured as a z-score against the robust spread within each layer. A small set of units that changed far more than the rest is a candidate for where the behaviour lives. That locates components but doesn't yet say which model is which; I'll then read those units' weights for direction.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18
summary=[]
for l in range(L):
    p=f"model.layers.{l}.mlp."
    G=[d[s][p+"gate_proj.weight"] for s in "AB"]; U=[d[s][p+"up_proj.weight"] for s in "AB"]; D=[d[s][p+"down_proj.weight"] for s in "AB"]
    dg=np.linalg.norm(G[0]-G[1],axis=1)/ (np.linalg.norm(G[1],axis=1)+1e-9)
    du=np.linalg.norm(U[0]-U[1],axis=1)/ (np.linalg.norm(U[1],axis=1)+1e-9)
    dd=np.linalg.norm(D[0]-D[1],axis=0)/ (np.linalg.norm(D[1],axis=0)+1e-9)
    ddabs=np.linalg.norm(D[0]-D[1],axis=0)
    tot = dg+du+dd
    def z(x): 
        med=np.median(x); mad=np.median(np.abs(x-med))*1.4826; return (x-med)/mad
    zt = z(ddabs)
    top=np.argsort(-zt)[:6]
    print(f"L{l:2d} down-col |d| median={np.median(ddabs):.4f} max={ddabs.max():.4f}; top units (z): " + ", ".join(f"{i}({zt[i]:.1f}, relD={dd[i]:.3f}, relG={dg[i]:.3f})" for i in top))
    summary.append((l, top, zt[top]))
```