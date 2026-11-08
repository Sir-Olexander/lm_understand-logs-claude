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
# also dual (functional) directions: A columns
Af=(A*sg).T; Af=Af/np.linalg.norm(Af,axis=1,keepdims=True); Ft=Af@T
QKs=np.vstack([H[(0,j)][i]/norm(H[(0,j)][i]) for j in (1,5) for i in (0,1)])
Pall=svd(QKs,full_matrices=False)[2]
for kP in (30,60):
    P=Pall[:kP]
    print(f"P{kP} energy of token dirs:", np.round([norm(P@Et[c])**2 for c in range(28)],2))
print("raw-SVD index fractions (dirs 2..29) of Et[2],Et[22],Et[4]:")
for c in (2,22,4,14):
    coef=(Vt[:40]@Et[c])**2
    print(c, np.round(coef,2))
for j in range(6):
    Q,K,V,O=H[(0,j)]
    print(f"L0H{j} |Q e|", np.round(norm(Q@Et.T,axis=0),2))
    print(f"      |K e|", np.round(norm(K@Et.T,axis=0),2))
print("ln1_w.0-weighted norm:", np.round(norm(Et*W['ln1_w.0'],axis=1),2))
```