```python
g=W['ln1_w.3']
for j in range(6):
    s=slice(64*j,64*j+64)
    V=W['W_v.3'][s]*g; O=W['W_o.3'][:,s]
    sv=np.linalg.svd(O@V,compute_uv=False)
    print(j, np.round(sv[:40],2))
```