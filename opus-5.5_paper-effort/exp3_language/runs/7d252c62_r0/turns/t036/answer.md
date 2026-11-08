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
H={0:heads(0)}
OV={(0,j):H[0][j][3]@H[0][j][2] for j in range(6)}
X=np.vstack([H[0][1][0],H[0][1][1],H[0][5][0],H[0][5][1]])
U,S,Vt=np.linalg.svd(X,full_matrices=False)
P=Vt[:20].T; projP=np.eye(384)-P@P.T
A_=OV[(0,1)]@projP; B_=OV[(0,5)]@projP
U2,S2,Vt2=np.linalg.svd(np.vstack([A_/np.linalg.norm(A_),B_/np.linalg.norm(B_)]),full_matrices=False)
T=Vt2[:27].T
Wfc=W['W_fc.0']*W['ln2_w.0']
a=Wfc@T; b=Wfc@OV[(0,5)]@T; c=Wfc@OV[(0,1)]@T; d=Wfc@OV[(0,2)]@T
na=np.linalg.norm(a,axis=1)
idx=np.argsort(-na)[:900]
Z=a[idx]/na[idx,None]
Lk=linkage(Z,method='average',metric='cosine')
lab=fcluster(Lk,0.35,criterion='distance')
cnt=np.bincount(lab)
cl=sorted([k for k in range(1,len(cnt)) if cnt[k]>=5],key=lambda k:-cnt[k])
F=np.array([Z[lab==k].mean(0) for k in cl]); F/=np.linalg.norm(F,axis=1,keepdims=True)
E=np.linalg.pinv(F); E/=np.linalg.norm(E,axis=0,keepdims=True)
Ap=a@E; Bp=b@E; Cp=c@E; Dp=d@E
K=len(cl)
print('per proxy: max over neurons of min(b_own,c_own) ; top3 values of min(b,c) ; sd(b) ; # neurons with b_own>0.15 & c_own>0.15 & a_own>0.1')
for k in range(K):
    mbc=np.minimum(Bp[:,k],Cp[:,k])
    top=np.sort(mbc)[::-1][:3]
    n3=np.sum((Bp[:,k]>0.15)&(Cp[:,k]>0.15)&(Ap[:,k]>0.1))
    n3d=np.sum((Bp[:,k]>0.15)&(Cp[:,k]>0.15)&(Dp[:,k]>0.15)&(Ap[:,k]>0.1))
    print(f'{k:2d}: {top.round(2)}  sd_b={Bp[:,k].std():.3f} sd_c={Cp[:,k].std():.3f}  n(a,b,c)={n3:2d} n(a,b,c,d)={n3d:2d}')
```