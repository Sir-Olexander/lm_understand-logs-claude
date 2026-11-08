```python
import numpy as np
from numpy.linalg import svd, norm
from sklearn.decomposition import FastICA
from scipy.stats import skew
np.set_printoptions(linewidth=240, precision=1, suppress=True)
U,S,Vt=svd(W['W_v.0']); T=Vt[2:30]
def head(l,j):
    s=slice(64*j,64*j+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
X=(W['W_fc.0']*W['ln2_w.0'])@T.T
ica=FastICA(n_components=28,random_state=0,max_iter=3000,whiten='unit-variance'); Sx=ica.fit_transform(X)
A=ica.mixing_; Ainv=np.linalg.pinv(A); sg=np.sign(skew(Sx,axis=0)); Ainv=(Ainv.T*sg).T
En=Ainv/np.linalg.norm(Ainv,axis=1,keepdims=True)
Et=En@T   # token dirs in residual coords (28 x 384), unit
OV05=H[(0,5)][3]@H[(0,5)][2]
print("tokens order: comps 0..27; special: 2, 22, 4")
for l in range(1,4):
    b=W[f'ln1_b.{l}']
    for j in range(6):
        Q,K,V,O=H[(l,j)]
        qb=W[f'W_q.{l}'][64*j:64*j+64]@b
        kb=K.T@qb/8
        B=Q.T@K/8
        same=np.einsum('ci,ij,cj->c',Et,B,Et)
        ind=np.einsum('ci,ij,cj->c',Et,B@OV05,Et)
        print(f"L{l}H{j} kb.e_c x10:", np.round(Et@kb*10,1))
        print(f"      same diag x100:", np.round(same*100,1))
        print(f"      ind(k05) diag x100:", np.round(ind*100,1))
```