```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def mats(l,j):
    lw=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
for l,j in [(3,4),(3,3),(3,1),(3,0),(2,0),(1,0)]:
    Q,K,V,O=mats(l,j)
    print('L%dH%d OV'%(l,j),np.linalg.svd(O@V,compute_uv=False)[:45])
    print('       V',np.linalg.svd(V,compute_uv=False)[:45])
```