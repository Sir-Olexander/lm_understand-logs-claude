```python
import numpy as np
from numpy.linalg import svd, norm
np.set_printoptions(linewidth=230, precision=3, suppress=True)
def head(l,j):
    s=slice(64*j,64*j+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
U,S,Vt=svd(W['W_v.0']); T=Vt[2:30]
QKs=np.vstack([H[(0,j)][i]/norm(H[(0,j)][i]) for j in (1,5) for i in (0,1)])
P=svd(QKs,full_matrices=False)[2][:30]; Pp=P-(P@T.T)@T; Pp=svd(Pp,full_matrices=False)[2][:30]
vecs={}
for j in range(6):
    Q,K,V,O=H[(0,j)]; B=Q.T@K
    M=Pp@B@T.T; u,s,vt=svd(M); vecs[f'{j}PT_k1']=vt[0]; vecs[f'{j}PT_k2']=vt[1]
    M=T@B@Pp.T; u,s,vt=svd(M); vecs[f'{j}TP_q1']=u[:,0]; vecs[f'{j}TP_q2']=u[:,1]
    M=T@B@T.T; u,s,vt=svd(M); vecs[f'{j}TT_q1']=u[:,0]; vecs[f'{j}TT_k1']=vt[0]; vecs[f'{j}TT_q2']=u[:,1]; vecs[f'{j}TT_k2']=vt[1]
names=[n for n in vecs if n[0] in '0135']
A=np.array([vecs[n] for n in names])
C=np.abs(A@A.T)
print("      "+" ".join(f"{n[:7]:>7}" for n in names))
for i,n in enumerate(names): print(f"{n:>8}"+" ".join(f"{c:7.2f}" for c in C[i]))
# overall special token subspace: SVD of stacked
Sp=svd(A,full_matrices=False)
print("stacked special vec sv:", np.round(Sp[1][:8],2))
```