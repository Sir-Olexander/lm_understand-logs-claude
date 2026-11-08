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
cl=sorted([c for c in range(1,len(cnt)) if cnt[c]>=5],key=lambda c:-cnt[c])
F=np.array([Z[lab==c].mean(0) for c in cl]); F/=np.linalg.norm(F,axis=1,keepdims=True)
print('clusters',len(cl),[int(cnt[c]) for c in cl])
# token proxies: solve E such that F E ~ I (pseudo-inverse), columns e_k
E=np.linalg.pinv(F)   # 27 x K
E/=np.linalg.norm(E,axis=0,keepdims=True)
def profile(v):
    v=v/np.linalg.norm(v)
    return v@E
for j in [0,2,3]:
    Wq,Wk,_,_=H[0][j]
    u,s,vt=np.linalg.svd((Wq.T@Wk)@projP)
    print(f'X0H{j} key-token profile over clusters:',profile(T.T@vt[0]).round(2))
for l in [1,2]:
    lb=W[f'ln1_b.{l}']
    for j in range(6):
        Wq,Wk,_,_=H[l][j]
        qb=W[f'W_q.{l}'][j*64:(j+1)*64]@lb; kp=Wk.T@qb
        print(f'L{l}H{j} keypref tok profile:',profile(T.T@kp).round(2))
        print(f'L{l}H{j} keypref prv profile:',profile(T.T@(OV[(0,5)].T@kp)).round(2))
```