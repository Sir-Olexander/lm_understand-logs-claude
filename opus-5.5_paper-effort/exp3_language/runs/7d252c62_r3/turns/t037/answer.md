```python
for j in range(6):
    V=W['W_v.0'][64*j:64*j+64]
    s=np.linalg.svd(V,compute_uv=False)
    print(j, np.round(s[:40],2))
# random baseline
rng=np.random.default_rng(0)
print("rand", np.round(np.linalg.svd(rng.normal(0,0.02*0.86,(64,384)),compute_uv=False)[[0,10,30,63]],2))
```