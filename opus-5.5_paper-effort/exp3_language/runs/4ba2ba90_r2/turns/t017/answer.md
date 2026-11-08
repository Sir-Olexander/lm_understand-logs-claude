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
kt={}
for j in [0,1,3,4,5]:
    q,k,v,o=head(0,j)
    U,s,Vh=np.linalg.svd(q.T@k)
    # key preference vector for a "positional" query: use top q-dir; key-score direction = s0*Vh[0]
    for r in range(2):
        x=T.T@Vh[r]; kt[(j,r)]=x/np.linalg.norm(x)
keys=list(kt)
print(keys)
print(np.array([[abs(kt[a]@kt[b]) for b in keys] for a in keys]))
# Also compare token-part of k-dirs with the token-part of OV2 top input dirs, and of L3H1 copy
```