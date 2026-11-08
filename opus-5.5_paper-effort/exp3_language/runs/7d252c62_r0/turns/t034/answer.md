```python
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
np.set_printoptions(precision=2, suppress=True, linewidth=250)
hd=64
def heads(l):
    lw=W[f'ln1_w.{l}']
    out=[]
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        Wq=W[f'W_q.{l}'][s]*lw; Wk=W[f'W_k.{l}'][s]*lw; Wv=W[f'W_v.{l}'][s]*lw; Wo=W[f'W_o.{l}'][:,s]
        out.append((Wq,Wk,Wv,Wo))
    return out
H={l:heads(l) for l in range(1)}
OV={(0,j):H[0][j][3]@H[0][j][2] for j in range(6)}
X=np.vstack([H[0][1][0],H[0][1][1],H[0][5][0],H[0][5][1]])
U,S,Vt=np.linalg.svd(X,full_matrices=False)
P=Vt[:20].T; projP=np.eye(384)-P@P.T
A_=OV[(0,1)]@projP; B_=OV[(0,5)]@projP
U2,S2,Vt2=np.linalg.svd(np.vstack([A_/np.linalg.norm(A_),B_/np.linalg.norm(B_)]),full_matrices=False)
T=Vt2[:27].T
Wfc=W['W_fc.0']*W['ln2_w.0']
a=Wfc@T; b=Wfc@OV[(0,5)]@T; c=Wfc@OV[(0,1)]@T
na=np.linalg.norm(a,axis=1)
idx=np.argsort(-na)[:900]
Z=a[idx]/na[idx,None]
Lk=linkage(Z,method='average',metric='cosine')
lab=fcluster(Lk,0.35,criterion='distance')
cnt=np.bincount(lab)
cl=sorted([k for k in range(1,len(cnt)) if cnt[k]>=5],key=lambda k:-cnt[k])
F=np.array([Z[lab==k].mean(0) for k in cl]); F/=np.linalg.norm(F,axis=1,keepdims=True)
E=np.linalg.pinv(F); E/=np.linalg.norm(E,axis=0,keepdims=True)
Ap=a@E; Bp=b@E; Cp=c@E
for k in [2,0,1,5,19,21]:
    m=idx[lab==cl[k]]
    print(f'cluster {k} (n={len(m)}): mean a-profile own={Ap[m,k].mean():+.2f}; prev-token profile per neuron (rows), cols=proxies:')
    order=np.argsort(-np.abs(Bp[m]).mean(0))[:8]
    print('    top prev proxies by mean|b|:',order, np.abs(Bp[m]).mean(0)[order].round(3))
    for n in m[:8]:
        print('    ',Bp[n].round(2))
```