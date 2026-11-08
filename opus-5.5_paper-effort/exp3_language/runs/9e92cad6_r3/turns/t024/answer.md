```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
W64={k:v.astype(np.float64) for k,v in W.items()}
def headov(l,j):
    s=slice(64*j,64*j+64); g=W64[f'ln1_w.{l}']
    return W64[f'W_o.{l}'][:,s]@(W64[f'W_v.{l}'][s]*g)
Up,sp,_=np.linalg.svd(W64['W_proj.3'])
for k in [28,40,60,100,384]:
    L=Up[:,:k]
    for j in range(6):
        sv=np.linalg.svd(L.T@headov(3,j),compute_uv=False)
        print(k, j, sv[:34])
```