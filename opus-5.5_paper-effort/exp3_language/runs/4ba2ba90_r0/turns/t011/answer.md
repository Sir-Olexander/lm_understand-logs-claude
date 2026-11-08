```python
g=W['ln1_w.0']
for j in range(6):
    s=slice(64*j,64*j+64)
    V=W['W_v.0'][s]*g; O=W['W_o.0'][:,s]
    sv=np.linalg.svd(O@V,compute_uv=False)
    print(j, np.round(sv[:40],3))
    print('   V sv', np.round(np.linalg.svd(V,compute_uv=False)[:20],3))
    print('   O sv', np.round(np.linalg.svd(O,compute_uv=False)[:20],3))
```