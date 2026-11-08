```python
import numpy as np
from numpy.linalg import svd, norm, eigvals, eigvalsh
np.set_printoptions(linewidth=230, precision=3, suppress=True)
def head(l,j):
    s=slice(64*j,64*j+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
# output subspace Z of positional-copy heads 0,3,4
outs=[]
for j,r in ((4,3),(0,2),(3,1)):
    Q,K,V,O=H[(0,j)]; U,S,Vt=svd(O@V); outs.append(U[:,:r]*S[:r])
Zs=svd(np.hstack(outs),full_matrices=False)
print("Z sv", np.round(Zs[1],2))
Z=Zs[0][:,:4].T   # 4 x 384
for j in (0,3,4):
    Q,K,V,O=H[(0,j)]; OV=O@V
    print(f"L0H{j} OV restricted to Z (Z OV row-space sv):", np.round(svd(Z@OV,compute_uv=False)[:4],2))
for l in (1,2,3):
    b=W[f'ln1_b.{l}']
    for j in range(6):
        Q,K,V,O=H[(l,j)]
        B=Q.T@K/8
        Bz=Z@B@Z.T
        qb=W[f'W_q.{l}'][64*j:64*j+64]@b
        kbz=Z@(K.T@qb)/8
        es=eigvalsh((Bz+Bz.T)/2)
        print(f"L{l}H{j} Bz=\n{np.round(Bz,3)}\n   sym eig {np.round(es,3)}  kb_z {np.round(kbz,3)} |kb|={norm(K.T@qb/8):.3f}")
```