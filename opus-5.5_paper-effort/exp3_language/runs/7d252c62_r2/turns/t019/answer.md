```python
import numpy as np
from numpy.linalg import svd, norm
from sklearn.decomposition import FastICA
np.set_printoptions(linewidth=230, precision=2, suppress=True)
U,S,Vt=svd(W['W_v.0']); T=Vt[2:30]
def head(l,j):
    s=slice(64*j,64*j+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
X=(W['W_fc.0']*W['ln2_w.0'])@T.T
ica=FastICA(n_components=28,random_state=0,max_iter=3000,whiten='unit-variance'); Sx=ica.fit_transform(X)
A=ica.mixing_; Ainv=np.linalg.pinv(A)
# fix sign: make sources mostly positive-skewed
from scipy.stats import skew, kurtosis
sg=np.sign(skew(Sx,axis=0)); A=A*sg; Ainv=(Ainv.T*sg).T; Sx=Sx*sg
E=Ainv  # rows: token embedding directions in T coords (up to scale)
En=E/np.linalg.norm(E,axis=1,keepdims=True)
kt=kurtosis(Sx,axis=0)
order=np.argsort(-kt)
print("comp kurt:",[(int(i),round(kt[i])) for i in order])
print("neurons strongly loading per comp (|s|>4):",[(int(i),int((np.abs(Sx[:,i])>4).sum())) for i in order])
print("token-embedding direction gram (cos) offdiag max:", np.round(np.max(np.abs(En@En.T-np.eye(28))),2))
QKs=np.vstack([H[(0,j)][i]/norm(H[(0,j)][i]) for j in (1,5) for i in (0,1)])
P=svd(QKs,full_matrices=False)[2][:30]; Pp=P-(P@T.T)@T; Pp=svd(Pp,full_matrices=False)[2][:30]
def tokvals(f):  # functional in T coords -> values on normalized token dirs
    v=En@f; return v
for j in range(6):
    Q,K,V,O=H[(0,j)]; B=Q.T@K/8
    # key-token preference averaged over query positions: use top singular of PT
    M=Pp@B@T.T; u,s,vt=svd(M)
    print(f"L0H{j} PT key-token fn s={s[0]:.3f}:", np.round(tokvals(vt[0])*s[0]*100,1))
    print(f"      PT2 s={s[1]:.3f}:", np.round(tokvals(vt[1])*s[1]*100,1))
    TT=T@B@T.T
    print(f"      TT diag-ish (e_c^T B e_c) x100:", np.round(np.einsum('ci,ij,cj->c',En,TT,En)*100,1))
    print(f"      V token read norms:", np.round(norm(V@T.T@En.T,axis=0),2))
```