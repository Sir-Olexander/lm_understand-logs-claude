```python
for nm in ['W_proj.3','W_o.3','W_proj.2','W_o.2','W_fc.0','W_v.0','W_q.0','W_k.0']:
    s=np.linalg.svd(W[nm],compute_uv=False)
    print(nm, s[:40])
# also composed: concatenated writes of layer 3
A=np.concatenate([W['W_proj.3'],W['W_o.3']],axis=1)
print('L3 writes', np.linalg.svd(A,compute_uv=False)[:40])
```