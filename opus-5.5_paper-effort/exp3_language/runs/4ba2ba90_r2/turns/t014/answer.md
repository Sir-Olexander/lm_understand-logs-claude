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
_,st,Vt=np.linalg.svd((o2@v2)@Pc)
T=Vt[:24].T
OV2=o2@v2
q,k,v,o=head(3,1); C=q.T@k@OV2
ev,EV=np.linalg.eig(C)
idx=np.argsort(-np.abs(ev))
for i in idx[:40]:
    w=EV[:,i]; w=w/np.linalg.norm(w)
    print('ev %.3f arg %+.2f  |P| %.2f |T| %.2f'%(abs(ev[i]),np.angle(ev[i]),np.linalg.norm(P.T@w),np.linalg.norm(T.T@w)))
# restricted to T: T^T C T
Ct=T.T@C@T
e=np.linalg.eigvals(Ct); print('T-restricted eig', np.sort_complex(e)[::-1])
Cs=(Ct+Ct.T)/2; print('T-restricted sym eig', np.sort(np.linalg.eigvalsh(Cs))[::-1])
```