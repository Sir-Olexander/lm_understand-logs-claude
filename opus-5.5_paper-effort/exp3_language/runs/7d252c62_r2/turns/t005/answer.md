```python
import numpy as np
from numpy.linalg import svd, eigvals, norm
np.set_printoptions(linewidth=220, precision=2, suppress=True)
def head(l,j):
    s=slice(64*j,64*j+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
def rows(M,k):
    U,S,Vt=svd(M,full_matrices=False); return Vt[:k],S
k=16
spaces={}
for j in range(6):
    Q,K,V,O=H[(0,j)]
    for nm,M in (('Q',Q),('K',K),('V',V)):
        R,S=rows(M,k); spaces[f'{j}{nm}']=R
        print(f"L0H{j}{nm} sv", np.round(S[:20],2))
names=list(spaces)
print("overlap (mean cos^2 of principal angles, top16 subspaces); random ~", k/384)
print("     "+" ".join(f"{n:>5}" for n in names))
for a in names:
    print(f"{a:>5}"+" ".join(f"{(norm(spaces[a]@spaces[b].T)**2/k):5.2f}" for b in names))
```