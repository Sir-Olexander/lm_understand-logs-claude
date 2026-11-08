```python
import numpy as np
from numpy.linalg import svd, norm, eigvalsh
np.set_printoptions(linewidth=220, precision=2, suppress=True)
def head(l,j):
    s=slice(64*j,64*j+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
Vs=np.vstack([H[(0,1)][2]/norm(H[(0,1)][2]),H[(0,5)][2]/norm(H[(0,5)][2])])
QKs=np.vstack([H[(0,j)][i]/norm(H[(0,j)][i]) for j in (1,5) for i in (0,1)])
T=svd(Vs,full_matrices=False)[2][:27]; P=svd(QKs,full_matrices=False)[2][:30]
OV={k:(v[3]@v[2]) for k,v in H.items()}
def pos(M):
    Ss=(M+M.T)/2; e=eigvalsh(Ss); return (e[e>0].sum()+e[e<0].sum())/np.abs(e).sum()
print("format: name norm/pos  (pos=+1 means PD symmetric part => matching)")
for l in range(1,4):
    for j in range(6):
        Q,K,V,O=H[(l,j)]
        B=Q.T@K
        items=[('same',T@B@T.T)]
        for h in range(6):
            items.append((f'k0{h}',T@B@OV[(0,h)]@T.T))
        for h in (1,5):
            items.append((f'q0{h}',T@OV[(0,h)].T@B@T.T))
        if l>=2:
            for h in range(6):
                items.append((f'k1{h}.05',T@B@OV[(1,h)]@OV[(0,5)]@T.T))
                items.append((f'k1{h}',T@B@OV[(1,h)]@T.T))
        print(f"L{l}H{j}", " ".join(f"{n}:{norm(M):.2f}/{pos(M):+.2f}" for n,M in items))
```