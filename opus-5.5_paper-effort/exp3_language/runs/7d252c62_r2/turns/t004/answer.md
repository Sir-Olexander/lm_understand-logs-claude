```python
import numpy as np
from numpy.linalg import svd, eigvals, norm
np.set_printoptions(linewidth=220, precision=3, suppress=True)
def head(l,j):
    s=slice(64*j,64*j+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
for j in [0,4,2,3]:
    Q,K,V,O=H[(0,j)]
    U,S,Vt=svd(O@V)
    for r in range(3):
        u=U[:,r]; v=Vt[r]
        iu=np.argsort(-np.abs(u))[:8]; iv=np.argsort(-np.abs(v))[:8]
        print(f"L0H{j} r{r} sv {S[r]:.2f} OUT dims",list(zip(iu,np.round(u[iu],2))), " IN dims",list(zip(iv,np.round(v[iv],2))))
        print("     participation out %.1f in %.1f"%(1/np.sum(u**4),1/np.sum(v**4)))
w0=W['ln1_w.0']
print("ln1_w.0 top dims", [(i,round(w0[i],2)) for i in np.argsort(-w0)[:25]])
print("ln1_b.0 at those", [(i,round(W['ln1_b.0'][i],3)) for i in np.argsort(-w0)[:25]])
for nm in ['ln1_b.1','ln2_b.0','ln1_w.1','ln2_w.0']:
    a=W[nm]; print(nm, [(i,round(a[i],2)) for i in np.argsort(-np.abs(a))[:12]])
```