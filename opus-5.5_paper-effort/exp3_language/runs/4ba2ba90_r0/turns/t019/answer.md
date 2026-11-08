```python
for nm in ['W_proj.3','W_o.3','W_proj.2','W_o.2','W_proj.1']:
    s=np.linalg.svd(W[nm],compute_uv=False)
    print(nm, np.round(s[:45],2))
    print('   ', np.round(s[45:80],2))
```