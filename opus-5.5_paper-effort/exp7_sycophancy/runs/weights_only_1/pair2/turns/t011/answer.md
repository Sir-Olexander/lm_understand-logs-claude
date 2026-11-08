The diagonal-sign test is invariant to any positive diagonal rescaling of the residual coordinates, and it is informative.

- **Base heads.** Many have extreme patterns: dz ≈ +25 means almost all 640 diagonal entries are positive (copy heads, e.g. 7.0, 7.3, 15.2, 15.3, 17.1). dz ≈ −25 means almost all are negative (copy-suppression heads, e.g. 7.1, 15.1, 9.1, 3.0, 11.3). So residual coordinates are a meaningful basis for copying in this model.
- **The change A−B.** It is near zero in most layers. Layers 16 and 17 are consistently positive: 2.6, 1.1, −0.1, 1.2 and 1.7, 2.1, 1.8, 0.5, with layer sums +2.1 and +2.0 and a combined z ≈ 3.9 for those 8 heads. Layer 11 is negative (−1.9 in the layer sum).

This agrees with the raw-trace result in run 4. A's last two attention layers copy more than B's, which is the predicted signature of A being sycophantic. P(A syc) ≈ 0.66.

I want more than this moderate signal. First I'll look at the fine structure of the per-element changes: is A−B continuous, or discrete like δ·(s_A − s_B)? If it is discrete (values 0 and ±2δ), I can separate the "both models moved the same way" set from the "they moved oppositely" set. On the opposite set, the base equals (A+B)/2 exactly, which would give a partial base reconstruction for later tests.

```python
A = Checkpoint("A"); B = Checkpoint("B")
for k in ["model.layers.3.self_attn.q_proj.weight","model.layers.17.mlp.down_proj.weight","model.layers.9.mlp.gate_proj.weight"]:
    a = A.get(k).ravel(); b = B.get(k).ravel()
    w = np.abs((a+b)/2)
    m = w < 0.003   # fine bf16 resolution here (ulp <= 1.5e-5)
    d = (a-b)[m]
    hist, edges = np.histogram(d, bins=np.linspace(-6e-4, 6e-4, 49))
    print(k, "n", m.sum(), "std", d.std())
    for i in range(0, 48):
        print(f"  {edges[i]*1e4:6.2f}e-4 {'#'*int(200*hist[i]/hist.max()/4)} {hist[i]}")
    # also individual model distribution of values a around small w? check A-mid and B-mid
```