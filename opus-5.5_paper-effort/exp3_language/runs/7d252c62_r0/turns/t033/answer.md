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
H={l:heads(l) for l in range(2)}
OV={(l,j):H[l][j][3]@H[l][j][2] for l in range(2) for j in range(6)}
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
# b-side token proxies: b is functional on embeddings too, so use same E
Ap=a@E; Bp=b@E     # 1536 x K : current-token and prev-token preference for each proxy
Az=(Ap-Ap.mean(0))/Ap.std(0); Bz=(Bp-Bp.mean(0))/Bp.std(0)
print(' k   corr(a_k,b_k)  #(a>2sd & b>2sd)  #(a>2sd & b<-2sd)  expected_if_indep   sd_b')
K=len(cl)
for k in range(K):
    r=np.corrcoef(Ap[:,k],Bp[:,k])[0,1]
    pp=np.sum((Az[:,k]>2)&(Bz[:,k]>2)); pn=np.sum((Az[:,k]>2)&(Bz[:,k]<-2))
    exp=np.sum(Az[:,k]>2)*np.sum(Bz[:,k]>2)/1536
    print(f'{k:2d}  {r:+.2f}   {pp:3d}   {pn:3d}   {exp:.1f}   {Bp[:,k].std():.3f}')
```