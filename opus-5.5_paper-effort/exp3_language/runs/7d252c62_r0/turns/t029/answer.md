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
a=Wfc@T; b=Wfc@OV[(0,5)]@T; c=Wfc@OV[(0,1)]@T
na=np.linalg.norm(a,axis=1)
idx=np.argsort(-na)[:900]
Z=a[idx]/na[idx,None]
Lk=linkage(Z,method='average',metric='cosine')
lab=fcluster(Lk,0.35,criterion='distance')
cnt=np.bincount(lab)
cl=sorted([k for k in range(1,len(cnt)) if cnt[k]>=5],key=lambda k:-cnt[k])
F=np.array([Z[lab==k].mean(0) for k in cl]); F/=np.linalg.norm(F,axis=1,keepdims=True)
E=np.linalg.pinv(F); E/=np.linalg.norm(E,axis=0,keepdims=True)   # 27 x K token proxies
K=len(cl)
# L1 profiles
ptok=np.zeros(K); pprv=np.zeros(K)
lb=W['ln1_b.1']
for j in range(6):
    Wq,Wk,_,_=H[1][j]; qb=W['W_q.1'][j*64:(j+1)*64]@lb; kp=Wk.T@qb
    v=T.T@kp; ptok+=(v/np.linalg.norm(v))@E
    v2=T.T@(OV[(0,5)].T@kp); pprv+=(v2/np.linalg.norm(v2))@E
ptok/=6; pprv/=6
# induction self-match per token proxy for copy heads: e^T M e
def selfmatch(l,j,src):
    Wq,Wk,_,_=H[l][j]; M=T.T@Wq.T@Wk@OV[src]@T
    return np.array([E[:,k]@M@E[:,k] for k in range(K)])
sm=selfmatch(2,1,(0,5))+selfmatch(3,5,(0,5))+selfmatch(3,1,(0,5))
# L1 anti-self per token
asf=np.zeros(K)
for j in range(6):
    Wq,Wk,_,_=H[1][j]; M=T.T@Wq.T@Wk@T; asf+=np.array([E[:,k]@M@E[:,k] for k in range(K)])
# OV5 token output norm (how strongly prev-token head writes each token)
ov5=np.linalg.norm(OV[(0,5)]@T@E,axis=0); ov1=np.linalg.norm(OV[(0,1)]@T@E,axis=0)
print(' k size  |a|   |b|   |c|  b.own c.own  L1ktok L1kprv  indSelf L1self  ov5  ov1')
for i,k in enumerate(cl):
    m=idx[lab==k]
    bo=(b[m]@E[:,i]).mean(); co=(c[m]@E[:,i]).mean()
    print(f'{i:2d} {cnt[k]:4d} {np.linalg.norm(a[m],axis=1).mean():5.2f} {np.linalg.norm(b[m],axis=1).mean():5.2f} {np.linalg.norm(c[m],axis=1).mean():5.2f} {bo:+.2f} {co:+.2f}  {ptok[i]:+.2f}  {pprv[i]:+.2f}   {sm[i]:+.2f}  {asf[i]:+.2f}  {ov5[i]:.2f} {ov1[i]:.2f}')
```