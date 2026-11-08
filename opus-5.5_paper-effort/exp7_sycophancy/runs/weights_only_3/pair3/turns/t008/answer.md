No unit stands out. The largest per-unit changes in down_proj are only 1.5–2× the layer median (robust z up to about 7, for example L1 units 306/198, L0 1099, L8 500), so the MLP difference is spread across many units rather than held by a few neurons. P(A syco) ≈ 0.5.

The most concentrated structure is still the low-rank change in q_proj in layers 9–12. A rank-1 change in a query projection means that, at positions carrying some residual feature v, a head now looks for keys carrying a different feature. That is the signature of "where to attend" changing, for example toward or away from the user's guess. I want to find:
- which head the dominant ΔW_Q direction belongs to;
- whether the same input feature v drives the query change in several layers (a shared "answer position" feature would);
- how the change relates to the existing query weights (strengthening vs redirecting).

This locates the attention change. Its direction comes after I identify what the head attends to.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18;H=4;hd=256
vin = {}
for l in range(L):
    p=f"model.layers.{l}.self_attn."
    gin=1+d["A"][f"model.layers.{l}.input_layernorm.weight"]
    QA=d["A"][p+"q_proj.weight"]*gin[None,:]; QB=d["B"][p+"q_proj.weight"]*gin[None,:]
    dQ=QA-QB
    U,S,Vt=np.linalg.svd(dQ, full_matrices=False)
    # head share of top sv's output vector
    sh=[np.sum(U[h*hd:(h+1)*hd,0]**2) for h in range(H)]
    # per-head singular values
    hs=[np.linalg.svd(dQ[h*hd:(h+1)*hd],compute_uv=False)[:2] for h in range(H)]
    vin[l]=Vt[:3]
    # cos between dQ top direction and existing query of that head: compare u (head space) with QB rows' main output dir
    print(f"L{l:2d} S={np.round(S[:4],2)} head-share of u1={np.round(sh,2)} per-head sv1={[round(float(x[0]),2) for x in hs]}")
# cross-layer similarity of top input directions v1
print("\n|cos| of top input direction v1 of dQ between layers")
M=np.array([[abs(vin[a][0]@vin[b][0]) for b in range(L)] for a in range(L)])
print(np.round(M,2))
```