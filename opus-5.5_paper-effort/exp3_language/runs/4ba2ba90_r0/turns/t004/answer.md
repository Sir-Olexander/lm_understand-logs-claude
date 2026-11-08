```python
g0=W['ln1_w.0']; g20=W['ln2_w.0']
M=np.vstack([W['W_q.0']*g0, W['W_k.0']*g0, W['W_v.0']*g0])
s=np.linalg.svd(M,compute_uv=False); print(np.round(s[:200],2)); 
M2=W['W_fc.0']*g20; s2=np.linalg.svd(M2,compute_uv=False); print(np.round(s2[:200],2))
s3=np.linalg.svd(np.vstack([W['W_q.0'],W['W_k.0'],W['W_v.0']]),compute_uv=False); print(np.round(s3,2))
```