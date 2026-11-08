```python
import numpy as np
from numpy.linalg import svd, norm
from sklearn.decomposition import FastICA
from scipy.stats import skew
np.set_printoptions(linewidth=240, precision=2, suppress=True)
U,S,Vt=svd(W['W_v.0']); T=Vt[2:30]
X=(W['W_fc.0']*W['ln2_w.0'])@T.T
ica=FastICA(n_components=28,random_state=0,max_iter=3000,whiten='unit-variance'); Sx=ica.fit_transform(X)
A=ica.mixing_; Ainv=np.linalg.pinv(A); sg=np.sign(skew(Sx,axis=0)); Ainv=(Ainv.T*sg).T
En=Ainv/np.linalg.norm(Ainv,axis=1,keepdims=True); Et=En@T
def rd(M): return norm(M@Et.T,axis=0)
rows={}
for l in range(4):
    w=W[f'ln1_w.{l}']; w2=W[f'ln2_w.{l}']
    for nm in ('W_q','W_k','W_v'):
        rows[f'{nm}.{l}']=rd(W[f'{nm}.{l}']*w)
    rows[f'W_fc.{l}']=rd(W[f'W_fc.{l}']*w2)
print("columns: token comps 0..27 (2,22 special; 4 semi); values normalized by median over tokens")
for k,v in rows.items():
    print(f"{k:8}", np.round(v/np.median(v),2))
# also how strongly MLP0 neuron biases / sources relate
print("ICA source energy per comp (sum s^2 over neurons with |s|>3):", np.round([np.sum(Sx[np.abs(Sx[:,c])>3,c]**2) for c in range(28)],0))
```