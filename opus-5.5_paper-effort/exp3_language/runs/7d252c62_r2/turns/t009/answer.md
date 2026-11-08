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
    S[f'o{l}{j}']=svd(O@V)[0][:,:r].T
def ratio(R,A): return (norm(R@A.T)**2/norm(R)**2)*(384/A.shape[0])
# overlaps between subspaces
nm=list(S)
print("subspace overlaps (fraction of A captured by C, mean cos^2)")
for a in ['T','P','o01','o05','o02','o00','o04']:
    print(a, " ".join(f"{c}:{norm(S[a]@S[c].T)**2/S[a].shape[0]:.2f}" for c in nm if c!=a))
for l in range(1,4):
    b=W[f'ln1_b.{l}']
    for j in range(6):
        Q,K,V,O=H[(l,j)]
        qb=W[f'W_q.{l}'][64*j:64*j+64]@b
        kb=K.T@qb
        out=[]
        for which,R in (('q',Q),('k',K),('v',V)):
            out.append(which+"["+" ".join(f"{a}{ratio(R,S[a]):.1f}" for a in nm if not(a.startswith('o') and int(a[1])>=l))+"]")
        kbf=" ".join(f"{a}{norm(S[a]@kb)**2/norm(kb)**2:.2f}" for a in ['T','P','o00','o04','o05','o01'])
        print(f"L{l}H{j}", " ".join(out), "| kb:",kbf)
```