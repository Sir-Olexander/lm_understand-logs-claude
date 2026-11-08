```python
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
np.set_printoptions(precision=1, suppress=True, linewidth=250)
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
a=Wfc@T; b=Wfc@OV[(0,5)]@T
na=np.linalg.norm(a,axis=1)
idx=np.argsort(-na)[:900]
Z=a[idx]/na[idx,None]
Lk=linkage(Z,method='average',metric='cosine')
lab=fcluster(Lk,0.35,criterion='distance')
cnt=np.bincount(lab)
cl=sorted([k for k in range(1,len(cnt)) if cnt[k]>=5],key=lambda k:-cnt[k])
F=np.array([Z[lab==k].mean(0) for k in cl]); F/=np.linalg.norm(F,axis=1,keepdims=True)
E=np.linalg.pinv(F); E/=np.linalg.norm(E,axis=0,keepdims=True)
Ap=a@E; Bp=b@E
Az=Ap/Ap.std(0); Bz=Bp/Bp.std(0)
Ra=np.maximum(Az-1.0,0); Rb=np.maximum(Bz-1.0,0)
N=Ra.T@Rb   # cur x prev
exp=np.outer(N.sum(1),N.sum(0))/N.sum()
ratio=N/exp
print('diag ratio (cur=k,prev=k):',np.diag(ratio).round(2))
print('row 2 (cur=2) ratios:',ratio[2].round(1))
print('col 2 (prev=2) ratios:',ratio[:,2].round(1))
print('row 19:',ratio[19].round(1))
print('col 19:',ratio[:,19].round(1))
# negative co-tuning: cur positive & prev strongly negative
Rbn=np.maximum(-Bz-1.0,0); Nn=Ra.T@Rbn; expn=np.outer(Nn.sum(1),Nn.sum(0))/Nn.sum(); rn=Nn/expn
print('diag ratio neg (cur=k, prev NOT k):',np.diag(rn).round(2))
```