```python
import numpy as np
from numpy.linalg import svd, norm
np.set_printoptions(linewidth=230, precision=2, suppress=True)
w0=W['ln1_w.0']; w2=W['ln2_w.0']
def head(l,j):
    s=slice(64*j,64*j+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
QKs=np.vstack([H[(0,j)][i]/norm(H[(0,j)][i]) for j in (1,5) for i in (0,1)])
Pfull=svd(QKs,full_matrices=False)[2]
for nm,M in [('Wv0 raw',W['W_v.0']),('Wv0*w',W['W_v.0']*w0),('Wq0 raw',W['W_q.0']),('Wk0 raw',W['W_k.0']),
             ('Wfc0 raw',W['W_fc.0']),('Wfc0*w2',W['W_fc.0']*w2),
             ('V heads1,2,3,5 *w', np.vstack([H[(0,j)][2] for j in (1,2,3,5)]))]:
    s=svd(M,compute_uv=False)
    print(nm, np.round(s[:70],2))
    print("   ratios s[k]/s[k+1] top:", sorted([(round(s[k]/s[k+1],3),k+1) for k in range(60)],reverse=True)[:6])
for kP in (20,30,40):
    Pp=Pfull[:kP]; proj=np.eye(384)-Pp.T@Pp
    M=np.vstack([H[(0,j)][2] for j in (1,2,3,5)])@proj
    s=svd(M,compute_uv=False)
    print("V1235 minus P",kP, np.round(s[:45],2))
```