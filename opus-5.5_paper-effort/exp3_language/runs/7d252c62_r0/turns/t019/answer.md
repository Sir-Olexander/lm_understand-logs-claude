```python
import numpy as np
from sklearn.cluster import KMeans
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
Y=Wfc@T   # 1536 x 27
nr=np.linalg.norm(Y,axis=1); full=np.linalg.norm(Wfc,axis=1)
print('frac of neuron weight in T: pctiles',np.percentile((nr/full)**2,[10,50,90,99]).round(3))
idx=np.argsort(-nr)[:600]
Z=Y[idx]/nr[idx,None]
C=Z@Z.T; np.fill_diagonal(C,0)
print('max cos per neuron pctiles',np.percentile(C.max(1),[10,25,50,75,90]).round(2))
rng=np.random.default_rng(0); R=rng.normal(size=Z.shape); R/=np.linalg.norm(R,axis=1,keepdims=True)
CR=R@R.T; np.fill_diagonal(CR,0); print('random max cos pctiles',np.percentile(CR.max(1),[10,25,50,75,90]).round(2))
# whitened version
Cov=Y.T@Y/len(Y); ev,Ev=np.linalg.eigh(Cov); Wh=Ev@np.diag(ev**-0.5)@Ev.T
Zw=(Y@Wh)[idx]; Zw/=np.linalg.norm(Zw,axis=1,keepdims=True)
Cw=Zw@Zw.T; np.fill_diagonal(Cw,0); print('whitened max cos pctiles',np.percentile(Cw.max(1),[10,25,50,75,90]).round(2))
```