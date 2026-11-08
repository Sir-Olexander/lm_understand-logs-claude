```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
W64={k:v.astype(np.float64) for k,v in W.items()}
l=3; g=W64[f'ln1_w.{l}']
def headov(l,j):
    s=slice(64*j,64*j+64); g=W64[f'ln1_w.{l}']
    return W64[f'W_o.{l}'][:,s]@(W64[f'W_v.{l}'][s]*g)
Up,sp,_=np.linalg.svd(W64['W_proj.3'])
for k in [21,24,28,32]:
    L=Up[:,:k]
    print(k, "L3H3", np.linalg.svd(L.T@headov(3,3),compute_uv=False)[:32])
    print(k, "L3H4", np.linalg.svd(L.T@headov(3,4),compute_uv=False)[:32])
# also spectrum of W_proj.3 weighted by ln2 input? and principal angles between L3H3 OV output top-k and Wproj3 top-k
U3,s3,_=np.linalg.svd(headov(3,3))
for k in [10,20,24,28]:
    print(k, "angles cos", np.linalg.svd(Up[:,:k].T@U3[:,:k],compute_uv=False).round(2))
```