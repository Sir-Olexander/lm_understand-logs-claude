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
P=np.linalg.svd(np.hstack([Uq[:,:26],Vk[:26].T]),full_matrices=False)[0][:,:26]
Pc=np.eye(384)-P@P.T
OV2=o2@v2
Uo,so,Vo=np.linalg.svd(OV2@Pc)
T=Vo[:24].T
OV2n=OV2@Pc/so[0]   # normalized
def pf(C):
    e=np.linalg.eigvals(C); return np.sum(e.real*np.abs(e))/np.sum(np.abs(e)**2)
print('head | cur-cur T^T M T: norm, posfrac | cur-prev T^T M OV2n T: norm, posfrac')
for l in [1,2,3]:
    for j in range(6):
        q,k,v,o=head(l,j); M=q.T@k
        A=T.T@M@T; B=T.T@M@OV2n@T
        print(f'L{l}H{j}  %.2f %+.2f   |  %.2f %+.2f'%(np.linalg.norm(A),pf(A),np.linalg.norm(B),pf(B)))
```