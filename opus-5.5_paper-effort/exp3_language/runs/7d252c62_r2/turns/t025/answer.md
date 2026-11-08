```python
import numpy as np
from numpy.linalg import svd, norm
from sklearn.decomposition import FastICA
from scipy.stats import skew
np.set_printoptions(linewidth=240, precision=2, suppress=True)
U,S,Vt=svd(W['W_v.0']); T=Vt[2:30]
def head(l,j):
    s=slice(64*j,64*j+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
H={(l,j):head(l,j) for l in range(4) for j in range(6)}
X=(W['W_fc.0']*W['ln2_w.0'])@T.T
ica=FastICA(n_components=28,random_state=0,max_iter=3000,whiten='unit-variance'); Sx=ica.fit_transform(X)
A=ica.mixing_; Ainv=np.linalg.pinv(A); sg=np.sign(skew(Sx,axis=0)); Ainv=(Ainv.T*sg).T
En=Ainv/np.linalg.norm(Ainv,axis=1,keepdims=True); Et=En@T
sel=[2,22,4]
for l,j in [(0,0),(0,1),(0,2),(0,3),(0,5),(1,1),(1,3),(1,5),(2,1),(2,2),(3,5)]:
    Q,K,V,O=H[(l,j)]; B=Q.T@K/8
    M=Et@B@Et.T*100
    letters=[c for c in range(28) if c not in sel]
    print(f"L{l}H{j} rows=query cols=key [2,22,4 | mean letters]")
    for a in sel+['L']:
        ra = M[a] if a!='L' else M[letters].mean(0)
        print("   q",a, np.round([ra[2],ra[22],ra[4],ra[letters].mean()],2))
```