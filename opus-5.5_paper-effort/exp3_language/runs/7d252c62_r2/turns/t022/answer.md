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
En=Ainv/np.linalg.norm(Ainv,axis=1,keepdims=True); Et=En@T
QKs=np.vstack([H[(0,j)][i]/norm(H[(0,j)][i]) for j in (1,5) for i in (0,1)])
P=svd(QKs,full_matrices=False)[2][:30]; Pp=P-(P@T.T)@T; Pp=svd(Pp,full_matrices=False)[2][:30]
for j in (0,1,2,3,5):
    Q,K,V,O=H[(0,j)]; B=Q.T@K/8
    M=T@B@Pp.T; u,s,vt=svd(M)
    print(f"L0H{j} TP query-token fn1 s={s[0]:.3f}:", np.round((En@u[:,0])*s[0]*100,1))
    print(f"        fn2 s={s[1]:.3f}:", np.round((En@u[:,1])*s[1]*100,1))
    # does the query-token shift use the same positional key-dir as PP top?
    Mpp=Pp@B@Pp.T; up,sp,vp=svd(Mpp)
    print("     key-pos dir of TP1 vs PP key dirs (cos with top5):", np.round(np.abs(vp[:5]@vt[0]),2), " PT key-tok query-pos dir vs PP query dirs:", np.round(np.abs(up[:,:5].T@svd(Pp@B@T.T)[0][:,0]),2))
# MLP0 neuron co-loading structure among letter comps: correlation of |loadings|
Ld=X@np.linalg.pinv(En@T@T.T).T if False else (W['W_fc.0']*W['ln2_w.0'])@Et.T   # neuron response to unit token dirs
C=np.corrcoef(Ld.T)
print("token-token corr of MLP0 neuron responses (x10):")
for c in range(28): print(c, np.round(C[c]*10,0).astype(int))
```