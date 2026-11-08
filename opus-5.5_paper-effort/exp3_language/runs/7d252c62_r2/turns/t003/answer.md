```python
import numpy as np
from numpy.linalg import svd, eigvals, norm
np.set_printoptions(linewidth=220, precision=3, suppress=True)
def head(l,j):
    s=slice(64*j,64*j+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
# column spaces of L0 OV (top-r left sing vecs)
L0out={}
for j in range(6):
    Q,K,V,O=H[(0,j)]
    U,S,Vt=svd(O@V); r=2 if j in (0,4) else 8
    L0out[j]=(U[:,:r],S[:r],Vt[:r])
    print(f"L0H{j} OV sv", np.round(S[:10],2))
def frac(v,U): return norm(U.T@v)**2/norm(v)**2
for l in range(4):
    b=W[f'ln1_b.{l}']; s_=None
    for j in range(6):
        s=slice(64*j,64*j+64)
        qb=W[f'W_q.{l}'][s]@b
        Q,K,V,O=H[(l,j)]
        kb=K.T@qb/8     # preferred key direction (in normalized-h space, weight applied)
        B=Q.T@K/8
        U,S,Vt=svd(B)
        line=f"L{l}H{j} |qb|={norm(qb):.2f} |kb|={norm(kb):.3f} QKsv1={S[0]:.3f}"
        if l>=1:
            line+=" kb-frac in L0out:"+",".join(f"{frac(kb,L0out[i][0]):.2f}" for i in range(6))
            line+=" | v1 key:"+",".join(f"{frac(Vt[0],L0out[i][0]):.2f}" for i in range(6))
            line+=" | u1 qry:"+",".join(f"{frac(U[:,0],L0out[i][0]):.2f}" for i in range(6))
            line+=" | v2 key:"+",".join(f"{frac(Vt[1],L0out[i][0]):.2f}" for i in range(6))
        print(line)
```