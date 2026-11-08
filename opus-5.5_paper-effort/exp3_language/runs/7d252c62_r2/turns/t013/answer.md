```python
import numpy as np
from numpy.linalg import svd, norm
np.set_printoptions(linewidth=230, precision=2, suppress=True)
w0=W['ln1_w.0']
def head(l,j):
    s=slice(64*j,64*j+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
QKs=np.vstack([H[(0,j)][i]/norm(H[(0,j)][i]) for j in (1,5) for i in (0,1)])
P=svd(QKs,full_matrices=False)[2][:30]
U,S,Vt=svd(W['W_v.0'])
print("raw Wv0 sv", np.round(S[:40],2))
print("frac in P of each right sing vec:", np.round([norm(P@Vt[i])**2 for i in range(40)],2))
# energy by head of each sing vec (left side) -> which head uses it
E=np.array([[norm(U[64*j:64*j+64,i])**2 for j in range(6)] for i in range(40)])
print("head usage of top 40 dirs (rows=dir):")
for i in range(40): print(i, round(S[i],2), np.round(E[i],2), "w-weighted norm %.2f"%norm(Vt[i]*w0))
# per-head raw V spectra
for j in range(6):
    print(j, np.round(svd(W['W_v.0'][64*j:64*j+64],compute_uv=False)[:36],2))
```