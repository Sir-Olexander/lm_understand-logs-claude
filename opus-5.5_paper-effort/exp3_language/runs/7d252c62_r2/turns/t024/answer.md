```python
import numpy as np
from numpy.linalg import svd, norm, eigvalsh
np.set_printoptions(linewidth=240, precision=2, suppress=True)
def head(l,j):
    s=slice(64*j,64*j+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
U,S,Vt=svd(W['W_v.0']); T=Vt[2:30]
OV={k:(v[3]@v[2]) for k,v in H.items()}
def pos(M):
    Ss=(M+M.T)/2; e=eigvalsh(Ss); return (e[e>0].sum()+e[e<0].sum())/np.abs(e).sum()
print("q-path x k-path circuits on token subspace: norm/pos")
paths={'T':np.eye(384),'01':OV[(0,1)],'02':OV[(0,2)],'03':OV[(0,3)],'05':OV[(0,5)]}
for l in (1,2,3):
    for j in range(6):
        Q,K,V,O=H[(l,j)]; B=Q.T@K
        out=[]
        for qa in ('T','05','01','02'):
            for ka in ('T','05','01','02','03'):
                M=T@paths[qa].T@B@paths[ka]@T.T
                out.append(f"{qa}>{ka}:{norm(M):.2f}/{pos(M):+.2f}")
        print(f"L{l}H{j}", " ".join(out))
```