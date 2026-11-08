```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
W64={k:v.astype(np.float64) for k,v in W.items()}
def headov(l,j):
    s=slice(64*j,64*j+64); g=W64[f'ln1_w.{l}']
    return W64[f'W_o.{l}'][:,s]@(W64[f'W_v.{l}'][s]*g)
O3=headov(0,3)
U,s,Vt=np.linalg.svd(O3)
Up,sp,_=np.linalg.svd(W64['W_proj.3'])
for k in [18,24,28,34,40]:
    T=Vt[:k].T
    print(k,"L3H3 OV on T", np.linalg.svd(headov(3,3)@T,compute_uv=False)[:k].round(2))
    print(k,"Wproj28^T L3H3 OV T", np.linalg.svd(Up[:,:28].T@headov(3,3)@T,compute_uv=False)[:k].round(2))
```