```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
hd=64
def head(l,j):
    sl=slice(j*hd,(j+1)*hd)
    g=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][sl]*g, W[f'W_k.{l}'][sl]*g, W[f'W_v.{l}'][sl]*g, W[f'W_o.{l}'][:,sl]
Up,sp,_=np.linalg.svd(W['W_proj.3'])
U=Up[:,:30]
print('W_proj.3 sv', sp[:34])
for j in range(6):
    q,k,v,o=head(3,j)
    OV=o@v
    s=np.linalg.svd(OV,compute_uv=False)
    sU=np.linalg.svd(U.T@OV,compute_uv=False)
    print(f'L3H{j} OV sv', s[:34]); print(f'   in U30   ', sU[:30])
    print('   frac of OV energy in U30: %.2f'%(np.linalg.norm(U.T@OV)**2/np.linalg.norm(OV)**2))
```