Null result: q_norm/k_norm changed in only 0–24 of 256 elements per layer. The attention-scale changes are at most 0.002% and show no pattern in sign, and the other norm gains change equally often in each direction. The norm parameters barely moved in either model, which suggests the per-element updates are very small compared with bf16 resolution.

That raises a basic question. Typical |A−B| in the projection matrices is about 2.3e-4, which is close to one bf16 ulp at the typical weight size (≈2.4e-4 for |w| ≈ 0.04). If most differing elements differ by exactly ±1 ulp, the diff is heavily quantised: rounding noise is comparable to the signal, which would explain the weak, mixed results. It would also mean that for elements where A and B differ by one ulp, the base value is probably one of the two, which might let me estimate the base for some elements. I'll histogram A−B in units of the local ulp for several tensors, and check how the sign of A−B relates to the sign of the weight, which tests for systematic shrinking or growing.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def ulp_bf16(x):
    e=np.floor(np.log2(np.maximum(np.abs(x),1e-30)))
    return 2.0**(e-7)
for k in ["model.layers.0.self_attn.q_proj.weight","model.layers.8.mlp.down_proj.weight","model.layers.17.mlp.down_proj.weight","model.layers.12.self_attn.q_proj.weight","model.layers.17.self_attn.o_proj.weight"]:
    a=A.get(k).ravel().astype(np.float64); b=B.get(k).ravel().astype(np.float64)
    u=np.minimum(ulp_bf16(a),ulp_bf16(b))
    d=(a-b)/u
    vals,cnts=np.unique(np.round(d).astype(int),return_counts=True)
    order=np.argsort(cnts)[::-1][:9]
    print(k.split("layers.")[1], "ulp-diff histogram:", {int(vals[i]):round(cnts[i]/len(d),4) for i in sorted(order)})
    # sign relation: is A-B correlated with sign(a)? (shrink vs grow)
    m=a!=b
    print("   frac(A-B same sign as weight) among differing:", np.mean(np.sign(a[m]-b[m])==np.sign(a[m]+b[m])))
```