```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
hd=64
def head(l,j):
    sl=slice(j*hd,(j+1)*hd)
    g=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][sl]*g, W[f'W_k.{l}'][sl]*g, W[f'W_v.{l}'][sl]*g, W[f'W_o.{l}'][:,sl]
for j in range(6):
    q,k,v,o=head(0,j)
    print(f'L0H{j} OV sv', np.linalg.svd(o@v,compute_uv=False)[:45])
    print(f'L0H{j} V  sv', np.linalg.svd(v,compute_uv=False)[:45])
```