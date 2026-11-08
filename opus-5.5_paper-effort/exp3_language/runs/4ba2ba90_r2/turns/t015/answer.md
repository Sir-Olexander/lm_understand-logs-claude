```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
hd=64
def head(l,j):
    sl=slice(j*hd,(j+1)*hd)
    g=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][sl]*g, W[f'W_k.{l}'][sl]*g, W[f'W_v.{l}'][sl]*g, W[f'W_o.{l}'][:,sl]
Up,sp,_=np.linalg.svd(W['W_proj.3'])
U=Up[:,:34]
print('idx      ', np.arange(34))
print('Wproj3 sv', sp[:34])
for j in range(6):
    q,k,v,o=head(3,j)
    print(f'L3H{j} wr ', np.linalg.norm(U.T@(o@v),axis=1))
for l in range(3):
    print(f'Wproj{l} wr', np.linalg.norm(U.T@W[f"W_proj.{l}"],axis=1))
    print(f'Wo{l} wr   ', np.linalg.norm(U.T@W[f"W_o.{l}"],axis=1))
print('Wo3 wr   ', np.linalg.norm(U.T@W["W_o.3"],axis=1))
print('lnf? ln2_w.3 at U dirs', (U**2).T@W['ln2_w.3'])
```