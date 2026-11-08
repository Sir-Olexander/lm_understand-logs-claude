```python
import numpy as np
from numpy.linalg import svd, norm, eigh
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
letters=[c for c in range(28) if c not in (2,22)]
OV05=H[(0,5)][3]@H[(0,5)][2]
maps={'fc0':W['W_fc.0']*W['ln2_w.0'],'q1':W['W_q.1']*W['ln1_w.1'],'k1':W['W_k.1']*W['ln1_w.1'],'v1':W['W_v.1']*W['ln1_w.1'],
      'fc1(prev)':(W['W_fc.1']*W['ln2_w.1'])@OV05,'fc0(prev)':(W['W_fc.0']*W['ln2_w.0'])@OV05,'v1(prev)':(W['W_v.1']*W['ln1_w.1'])@OV05}
for nm,M in maps.items():
    R=M@Et[letters].T     # features x 26
    R=R-R.mean(1,keepdims=True)
    G=R.T@R
    ev,evec=eigh(G); ev=ev[::-1]; evec=evec[:,::-1]
    v=evec[:,0]
    print(f"{nm:10} eig frac top5:", np.round(ev[:5]/ev.sum(),3))
    for k in range(2):
        v=evec[:,k]; o=np.argsort(v)
        print(f"      pc{k}:", " ".join(f"{letters[i]}:{v[i]:+.2f}" for i in o))
```