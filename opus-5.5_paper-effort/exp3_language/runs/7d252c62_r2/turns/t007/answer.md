```python
import numpy as np
from numpy.linalg import svd, norm
np.set_printoptions(linewidth=220, precision=2, suppress=True)
def head(l,j):
    s=slice(64*j,64*j+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
Vs=np.vstack([H[(0,1)][2]/norm(H[(0,1)][2]),H[(0,5)][2]/norm(H[(0,5)][2])])
QKs=np.vstack([H[(0,j)][i]/norm(H[(0,j)][i]) for j in (1,5) for i in (0,1)])
_,Sv,Vv=svd(Vs,full_matrices=False); _,Sq,Vq=svd(QKs,full_matrices=False)
print("V(1,5) sv", np.round(Sv[:50],3))
print("QK(1,5) sv", np.round(Sq[:60],3))
# per-head V sv beyond 20
for j in (1,5):
    print(f"L0H{j}V sv all", np.round(svd(H[(0,j)][2],compute_uv=False)[:64],2))
    print(f"L0H{j}K sv all", np.round(svd(H[(0,j)][1],compute_uv=False)[:64],2))
    print(f"L0H{j}Q sv all", np.round(svd(H[(0,j)][0],compute_uv=False)[:64],2))
for kT in (20,27,32):
    T=Vv[:kT]; P=Vq[:kT]
    print(kT,"overlap T-P", round(norm(T@P.T)**2/kT,3))
T=Vv[:27]; P=Vq[:30]
for j in range(6):
    Q,K,V,O=H[(0,j)]
    for nm,M in (('Q',Q),('K',K),('V',V)):
        _,S,Vt=svd(M,full_matrices=False)
        fT=[norm(T@Vt[r])**2 for r in range(4)]; fP=[norm(P@Vt[r])**2 for r in range(4)]
        print(f"L0H{j}{nm} top4 dirs: inT",np.round(fT,2)," inP",np.round(fP,2), " sv",np.round(S[:4],2))
```