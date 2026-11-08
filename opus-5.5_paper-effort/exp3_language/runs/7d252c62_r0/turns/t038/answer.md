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
H={l:heads(l) for l in range(3)}
OV={(l,j):H[l][j][3]@H[l][j][2] for l in range(3) for j in range(6)}
X=np.vstack([H[0][1][0],H[0][1][1],H[0][5][0],H[0][5][1]])
U,S,Vt=np.linalg.svd(X,full_matrices=False)
P=Vt[:20].T; projP=np.eye(384)-P@P.T
A_=OV[(0,1)]@projP; B_=OV[(0,5)]@projP
U2,S2,Vt2=np.linalg.svd(np.vstack([A_/np.linalg.norm(A_),B_/np.linalg.norm(B_)]),full_matrices=False)
T=Vt2[:27].T
Wfc=W['W_fc.0']*W['ln2_w.0']
a=Wfc@T
na=np.linalg.norm(a,axis=1)
idx=np.argsort(-na)[:900]
Z=a[idx]/na[idx,None]
Lk=linkage(Z,method='average',metric='cosine')
lab=fcluster(Lk,0.35,criterion='distance')
cnt=np.bincount(lab)
cl=sorted([k for k in range(1,len(cnt)) if cnt[k]>=5],key=lambda k:-cnt[k])
F=np.array([Z[lab==k].mean(0) for k in cl]); F/=np.linalg.norm(F,axis=1,keepdims=True)
E=np.linalg.pinv(F); E/=np.linalg.norm(E,axis=0,keepdims=True)
I=np.eye(384)
src={'tok':I,'H5':OV[(0,5)],'H1':OV[(0,1)]}
for (l,j,qn,kn) in [(1,5,'H5','tok'),(1,5,'tok','tok'),(1,2,'H5','tok'),(1,1,'tok','tok'),(2,2,'H5','tok')]:
    Wq,Wk,_,_=H[l][j]
    M=T.T@src[qn].T@Wq.T@Wk@src[kn]@T
    Mp=E.T@M@E   # proxy x proxy
    u,s,vt=np.linalg.svd(M)
    print(f'L{l}H{j} {qn}>{kn}: s={s[:4].round(2)}')
    print('   query side profile',(u[:,0]@E).round(2))
    print('   key side profile  ',(vt[0]@E).round(2))
    print('   proxy-proxy diag  ',np.diag(Mp).round(2))
```