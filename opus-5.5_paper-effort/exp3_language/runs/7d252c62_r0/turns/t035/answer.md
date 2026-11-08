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
# Over ALL neurons: for each proxy k, count neurons where k is the argmax (positive) of a-, b- and c-profiles simultaneously
amax=Ap.argmax(1); bmax=Bp.argmax(1); cmax=Cp.argmax(1)
strong=(na>np.percentile(na,40))
print('proxy: #neurons with argmax_a=argmax_b=k ; argmax_a=argmax_b=argmax_c=k ; argmax_a=argmax_c=k')
for k in range(K):
    ab=np.sum(strong&(amax==k)&(bmax==k)); abc=np.sum(strong&(amax==k)&(bmax==k)&(cmax==k)); ac=np.sum(strong&(amax==k)&(cmax==k))
    print(f'{k:2d}: {ab:3d} {abc:3d} {ac:3d}   (n argmax_a={np.sum(strong&(amax==k))})')
m=idx[lab==cl[19]]
print('cluster19 neurons: own a,b,c,d :')
for n in m: print('   ',n, Ap[n,19].round(2),Bp[n,19].round(2),Cp[n,19].round(2),Dp[n,19].round(2),' b-rank of own',int((Bp[n]>Bp[n,19]).sum()),' c-rank',int((Cp[n]>Cp[n,19]).sum()), ' c-top',Cp[n].argsort()[::-1][:3])
```