```python
import numpy as np
from numpy.linalg import svd, norm
np.set_printoptions(linewidth=230, precision=2, suppress=True)
def head(l,j):
    s=slice(64*j,64*j+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
def top(M,k): return svd(M,full_matrices=False)[2][:k]
def pcos(A,B): return svd(A@B.T,compute_uv=False)
raw={j:W['W_v.0'][64*j:64*j+64] for j in range(6)}
for k in (32,40):
    print("k",k)
    for a,b in [(1,5),(1,2),(5,2),(1,3),(5,3),(2,3)]:
        c=pcos(top(H[(0,a)][2],k),top(H[(0,b)][2],k))
        print(f"  V0{a} vs V0{b} (w-scaled) cos:",np.round(c,2))
        c=pcos(top(raw[a],k),top(raw[b],k))
        print(f"  V0{a} vs V0{b} (raw)      cos:",np.round(c,2))
```