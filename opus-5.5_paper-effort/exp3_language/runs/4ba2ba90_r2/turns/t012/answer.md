```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=250)
hd=64
def head(l,j):
    sl=slice(j*hd,(j+1)*hd)
    g=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][sl]*g, W[f'W_k.{l}'][sl]*g, W[f'W_v.{l}'][sl]*g, W[f'W_o.{l}'][:,sl]
q2,k2,v2,o2=head(0,2)
Uq,sq,Vk=np.linalg.svd(q2.T@k2)
for r in [20,26,30]:
    P=np.linalg.qr(np.hstack([Uq[:,:r],Vk[:r].T]))[0]
    P=np.linalg.svd(P,full_matrices=False)[0][:, :r]  # top r of combined
    Pc=np.eye(384)-P@P.T
    OV2=o2@v2
    s_all=np.linalg.svd(OV2,compute_uv=False)
    s_perp=np.linalg.svd(OV2@Pc,compute_uv=False)
    s_par=np.linalg.svd(OV2@P,compute_uv=False)
    print(r,'OV2 perp-P sv', s_perp[:40]); print('   OV2 in-P sv', s_par[:20])
```