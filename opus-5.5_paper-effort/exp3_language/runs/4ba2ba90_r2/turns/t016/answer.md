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
_,st,Vt=np.linalg.svd((o2@v2)@Pc); T=Vt[:24].T
# D subspace: combined output of broad heads
OVb=np.hstack([head(0,j)[3]@head(0,j)[2] for j in [0,1,4,5]])
Ud,sd,_=np.linalg.svd(OVb); print('broad OV combined sv', sd[:8])
D=Ud[:,:3]
print('D vs P overlap', np.linalg.svd(D.T@P,compute_uv=False), 'D vs T', np.linalg.svd(D.T@T,compute_uv=False))
for l in [1,2,3]:
    for j in range(6):
        q,k,v,o=head(l,j); M=q.T@k
        DD=D.T@M@D
        # query partners of D keys
        Z=M@D  # columns: query directions
        zn=np.linalg.norm(Z,axis=0)
        fracs=[(np.linalg.norm(P.T@Z[:,i])/zn[i], np.linalg.norm(T.T@Z[:,i])/zn[i], np.linalg.norm(D.T@Z[:,i])/zn[i]) for i in range(3)]
        print(f'L{l}H{j} D-D block eig', np.linalg.eigvals(DD).round(2), ' |M D| cols', zn.round(2), ' query-partner fracs (P,T,D):', np.array(fracs).round(2).tolist())
```