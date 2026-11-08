```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
W64={k:v.astype(np.float64) for k,v in W.items()}
for name in ['W_proj.3','W_o.3','W_proj.2','W_o.2','W_proj.1','W_proj.0']:
    s=np.linalg.svd(W64[name],compute_uv=False)
    print(name, s[:45])
    print("   ratios", (s[:44]/s[1:45])[:44])
```