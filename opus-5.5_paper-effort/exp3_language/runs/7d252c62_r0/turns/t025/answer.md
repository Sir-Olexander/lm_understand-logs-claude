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
a=Wfc@T; b=Wfc@OV[(0,5)]@T
na=np.linalg.norm(a,axis=1)
idx=np.argsort(-na)[:900]
Z=a[idx]/na[idx,None]
Lk=linkage(Z,method='average',metric='cosine')
lab=fcluster(Lk,0.35,criterion='distance')
cnt=np.bincount(lab)
cl=[c for c in range(1,len(cnt)) if cnt[c]>=3]
cl=sorted(cl,key=lambda c:-cnt[c])
F=np.array([Z[lab==c].mean(0) for c in cl]); F/=np.linalg.norm(F,axis=1,keepdims=True)
print('n clusters',len(cl),'sizes',[cnt[c] for c in cl])
G=F@F.T
offd=G[~np.eye(len(cl),dtype=bool)]
print('centroid offdiag |cos| mean %.2f max %.2f'%(np.abs(offd).mean(),np.abs(offd).max()))
print('sv of F',np.linalg.svd(F,compute_uv=False).round(2))
# repeat score: for neurons in cluster k, their prev-token functional b_n projected onto own centroid F_k vs others
for i,c in enumerate(cl):
    members=idx[lab==c]
    bm=b[members]; am=a[members]
    own=(bm@F[i]); allp=bm@F.T
    print(f'cl{i:2d} size {cnt[c]:3d} |a| {np.linalg.norm(am,axis=1).mean():.2f} |b| {np.linalg.norm(bm,axis=1).mean():.2f} b.own mean {own.mean():+.2f}  b.F best other {np.sort(allp.mean(0))[::-1][:3].round(2)} rank of own {int((allp.mean(0)>own.mean()).sum())}')
```