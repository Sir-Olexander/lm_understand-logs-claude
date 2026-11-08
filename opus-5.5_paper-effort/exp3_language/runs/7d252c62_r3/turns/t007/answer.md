```python
def mats(l,j):
    w=W[f'ln1_w.{l}']
    s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
for j in range(6):
    Q,K,V,O=mats(0,j)
    sv=np.linalg.svd(O@V,compute_uv=False)
    print(j, np.round(sv[:40],2))
    sv=np.linalg.svd(V,compute_uv=False); print('  V', np.round(sv[:40],2))
```