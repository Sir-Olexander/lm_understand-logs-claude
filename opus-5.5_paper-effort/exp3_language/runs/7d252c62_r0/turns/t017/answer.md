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
T=Vt2[:32].T
print('stack sv',(S2[:32]/S2[0]).round(2))
Wfc=W['W_fc.0']*W['ln2_w.0']
Uf,Sf,Vtf=np.linalg.svd(Wfc@projP,full_matrices=False)
print('Wfc top in-dir coords in T basis', (Vtf[0]@T).round(2), ' |proj|',np.linalg.norm(Vtf[0]@T).round(2))
print('Wfc 2nd in-dir coords in T basis', (Vtf[1]@T).round(2), ' |proj|',np.linalg.norm(Vtf[1]@T).round(2))
for j in [0,2,3]:
    Wq,Wk,Wv,Wo=H[0][j]
    QK=Wq.T@Wk
    # key direction: top right singular vector of QK restricted to complement of P on key side
    u,s,vt=np.linalg.svd(QK@projP)
    for r in range(2):
        v=vt[r]
        print(f'L0H{j} key dir {r} (sv {s[r]:.2f}) coords in T:',(v@T).round(2),' |proj T|=%.2f'%np.linalg.norm(v@T))
```