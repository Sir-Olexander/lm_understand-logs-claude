```python
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
np.set_printoptions(precision=2, suppress=True, linewidth=220)
hd=64
def heads(l):
    lw=W[f'ln1_w.{l}']
    out=[]
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        Wq=W[f'W_q.{l}'][s]*lw; Wk=W[f'W_k.{l}'][s]*lw; Wv=W[f'W_v.{l}'][s]*lw; Wo=W[f'W_o.{l}'][:,s]
        out.append((Wq,Wk,Wv,Wo))
    return out
H={l:heads(l) for l in range(4)}
OV={(l,j):H[l][j][3]@H[l][j][2] for l in range(4) for j in range(6)}
X=np.vstack([H[0][1][0],H[0][1][1],H[0][5][0],H[0][5][1]])
U,S,Vt=np.linalg.svd(X,full_matrices=False)
P=Vt[:30].T; projP=np.eye(384)-P@P.T
A_=OV[(0,1)]@projP; B_=OV[(0,5)]@projP
U2,S2,Vt2=np.linalg.svd(np.vstack([A_/np.linalg.norm(A_),B_/np.linalg.norm(B_)]),full_matrices=False)
T=Vt2[:27].T
Wfc=W['W_fc.0']*W['ln2_w.0']
Y=Wfc@T
nr=np.linalg.norm(Y,axis=1)
idx=np.argsort(-nr)[:800]
Z=Y[idx]/nr[idx,None]
Lk=linkage(Z,method='average',metric='cosine')
for th in [0.1,0.2,0.3]:
    lab=fcluster(Lk,th,criterion='distance')
    cnt=np.bincount(lab)[1:]
    big=np.sort(cnt)[::-1]
    print('th',th,'nclusters',len(cnt),'sizes>=3:',(cnt>=3).sum(),'top sizes',big[:40])
lab=fcluster(Lk,0.2,criterion='distance')
cnt=np.bincount(lab)
cl=[c for c in range(1,len(cnt)) if cnt[c]>=4]
cent=np.array([Z[lab==c].mean(0) for c in cl]); cent/=np.linalg.norm(cent,axis=1,keepdims=True)
G=cent@cent.T
print('n big clusters',len(cl))
print('centroid gram (rounded):'); print(np.round(G,1))
print('rank-ish sv of centroids', np.linalg.svd(cent,compute_uv=False).round(2))
```