```python
import numpy as np
from numpy.linalg import svd, norm, eigvals
np.set_printoptions(linewidth=220, precision=2, suppress=True)
def head(l,j):
    s=slice(64*j,64*j+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
Vs=np.vstack([H[(0,1)][2]/norm(H[(0,1)][2]),H[(0,5)][2]/norm(H[(0,5)][2])])
QKs=np.vstack([H[(0,j)][i]/norm(H[(0,j)][i]) for j in (1,5) for i in (0,1)])
T=svd(Vs,full_matrices=False)[2][:27]; P=svd(QKs,full_matrices=False)[2][:30]
S={'T':T,'P':P}
for (l,j),r in [((0,0),2),((0,1),20),((0,2),10),((0,3),5),((0,4),3),((0,5),20)]+[((1,j),16) for j in range(6)]+[((2,j),16) for j in range(6)]:
    Q,K,V,O=H[(l,j)]
    U=svd(O@V)[0][:,:r].T
    S[f'o{l}{j}']=U
names=list(S)
def blk(B,A,C): return norm(A@B@C.T)**2/norm(B)**2/(A.shape[0]*C.shape[0]/384**2)
for l in range(1,4):
    for j in range(6):
        Q,K,V,O=H[(l,j)]
        B=Q.T@K
        res=[]
        for a in names:
            if a.startswith('o') and int(a[1])>=l: continue
            for c in names:
                if c.startswith('o') and int(c[1])>=l: continue
                v=blk(B,S[a],S[c])
                if v>4: res.append((v,a,c))
        res.sort(reverse=True)
        print(f"L{l}H{j} QK blocks (enrichment>4, query,key):", " ".join(f"{a}->{c}:{v:.0f}" for v,a,c in res[:14]))
```