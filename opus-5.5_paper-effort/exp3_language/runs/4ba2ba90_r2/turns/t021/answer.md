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
OVb=np.hstack([head(0,j)[3]@head(0,j)[2] for j in [0,1,4,5]])
D=np.linalg.svd(OVb)[0][:,:3]
O2=np.linalg.svd(o2@v2)[0][:,:24]   # prev-token output subspace
Pc=np.eye(384)-P@P.T
T=np.linalg.svd((o2@v2)@Pc)[2][:24].T
# orthogonalize sequentially: D, P, O2, T
B=[];names=['D','P','O2','T']
acc=np.zeros((384,0))
for S in [D,P,O2,T]:
    S2=S-acc@(acc.T@S); S2=np.linalg.svd(S2,full_matrices=False)[0][:, :S.shape[1]]
    B.append(S2); acc=np.hstack([acc,S2])
R=np.linalg.svd(np.eye(384)-acc@acc.T)[0][:,:384-acc.shape[1]]
B.append(R); names.append('rest')
for l in [1,2,3]:
    for j in range(6):
        q,k,v,o=head(l,j); M=q.T@k; tot=np.linalg.norm(M)**2
        tab=np.array([[np.linalg.norm(Bi.T@M@Bj)**2/tot for Bj in B] for Bi in B])
        print(f'L{l}H{j} |M|={np.sqrt(tot):.1f}  rows=Q side, cols=K side {names}'); print(tab.round(3))
```