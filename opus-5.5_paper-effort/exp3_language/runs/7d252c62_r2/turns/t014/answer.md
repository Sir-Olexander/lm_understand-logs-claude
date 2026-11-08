```python
import numpy as np
from numpy.linalg import svd, norm
np.set_printoptions(linewidth=230, precision=3, suppress=True)
w0=W['ln1_w.0']
def head(l,j):
    s=slice(64*j,64*j+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
U,S,Vt=svd(W['W_v.0'])
T=Vt[2:30]      # token dirs (28)
QKs=np.vstack([H[(0,j)][i]/norm(H[(0,j)][i]) for j in (1,5) for i in (0,1)])
P=svd(QKs,full_matrices=False)[2][:30]
# orthogonalize P against T
Pp=P-(P@T.T)@T; Pp=svd(Pp,full_matrices=False)[2][:30]
print("check T-P overlap", round(norm(T@Pp.T)**2,3))
for j in range(6):
    Q,K,V,O=H[(0,j)]
    B=Q.T@K/8
    tot=norm(B)**2
    blocks={'PP':Pp@B@Pp.T,'PT':Pp@B@T.T,'TP':T@B@Pp.T,'TT':T@B@T.T}
    print(f"L0H{j} |B|={np.sqrt(tot):.3f}", " ".join(f"{k}:{norm(M)**2/tot:.2f}" for k,M in blocks.items()), " rest:%.2f"%(1-sum(norm(M)**2 for M in blocks.values())/tot))
    for k,M in blocks.items():
        s=svd(M,compute_uv=False)
        print("    ",k,"sv",np.round(s[:8],3))
    # V reading
    print("    V energy: T %.2f P %.2f"%(norm(V@T.T)**2/norm(V)**2, norm(V@Pp.T)**2/norm(V)**2))
```