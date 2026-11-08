```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
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
Up,sp,_=np.linalg.svd(W['W_proj.3']); U=Up[:,1:34]
q,k,v,o=head(3,1); OV=o@v
for r in [24,32,40,60]:
    T=Vt[:r].T
    s=np.linalg.svd(U.T@OV@T,compute_uv=False)
    print(r,'copy U<-T sv', s[:34])
# full: OV restricted to U output, all inputs
s=np.linalg.svd(U.T@OV,compute_uv=False); print('copy U<-all', s[:34])
# induction C restricted to T40 both sides, symmetric part eigen
C=q.T@k@(o2@v2)
for r in [24,40,60]:
    T=Vt[:r].T; Ct=T.T@C@T; e=np.sort(np.linalg.eigvalsh((Ct+Ct.T)/2))[::-1]
    print(r,'induction sym eig', e[:34])
```