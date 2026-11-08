```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
hd=64
def head(l,j):
    sl=slice(j*hd,(j+1)*hd)
    g=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][sl]*g, W[f'W_k.{l}'][sl]*g, W[f'W_v.{l}'][sl]*g, W[f'W_o.{l}'][:,sl]
q,k,v,o=head(0,2)
M=q.T@k
ev=np.linalg.eigvals(M)
idx=np.argsort(-np.abs(ev))[:40]
for i in idx: print('%.2f  arg %.3f rad  (%.1f deg)'%(abs(ev[i]),np.angle(ev[i]),np.degrees(np.angle(ev[i]))))
# overlap of L0 W_o top dirs with later W_k top dirs
U0,s0,_=np.linalg.svd(W['W_o.0']); 
for l in [1,2,3]:
    _,sk,Vk=np.linalg.svd(W[f'W_k.{l}']*W[f'ln1_w.{l}'])
    print(l, 'Wk sv', sk[:4], 'overlap of Wk top2 right vecs with Wo.0 top3 left:', np.abs(Vk[:2]@U0[:,:3]))
```