```python
import numpy as np
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
def outsp(l,j,k): return np.linalg.svd(OV[(l,j)],full_matrices=False)[0][:,:k]
subs={'T27':T,'P30':P,'O4':outsp(0,4,2),'O0':outsp(0,0,2),'O5':outsp(0,5,24),'O1':outsp(0,1,24),'O2':outsp(0,2,10),'O3':outsp(0,3,10)}
R=np.linalg.qr(np.random.default_rng(0).normal(size=(384,24)))[0]
print('random24 baseline frac ~',24/384)
for l in range(1,4):
    for j in range(6):
        Wq,Wk,Wv,Wo=H[l][j]
        M=Wo@Wv
        tot=np.linalg.norm(M)**2
        print(f'L{l}H{j} OV-read:',' '.join(f'{k}={np.linalg.norm(M@B)**2/tot:.2f}' for k,B in subs.items()))
```