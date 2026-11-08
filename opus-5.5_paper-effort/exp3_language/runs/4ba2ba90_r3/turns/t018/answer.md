```python
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
np.set_printoptions(linewidth=220, precision=1, suppress=True)
def head(l,h):
    s=slice(64*h,64*h+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
q,k,v,o=head(0,2); A0=q.T@k; OVp=o@v
U0,s0,Vt0=np.linalg.svd(A0); Ppos=np.linalg.qr(np.hstack([U0[:,:18],Vt0[:18].T]))[0][:,:24]
U2,s2,Vt2=np.linalg.svd(OVp)
F=W['W_fc.0']*W['ln2_w.0']
for kT in [28,36,44]:
    T=Vt2[:kT].T; T=T-Ppos@(Ppos.T@T); T=np.linalg.qr(T)[0]
    X=F@T
    for nc in [kT-4,kT]:
        ica=FastICA(n_components=nc,random_state=1,max_iter=3000,whiten='unit-variance')
        S=ica.fit_transform(X)
        ku=np.sort(kurtosis(S,axis=0))[::-1]
        print(f"T{kT} nc{nc}: kurt",ku, " #>30:",(ku>30).sum())
# full-space ICA of W_fc.0 restricted to top-60 input singular subspace (excluding Ppos)
Q=np.linalg.svd(F - (F@Ppos)@Ppos.T,full_matrices=False)[2][:60].T
X=F@Q
ica=FastICA(n_components=60,random_state=0,max_iter=3000,whiten='unit-variance'); S=ica.fit_transform(X)
ku=np.sort(kurtosis(S,axis=0))[::-1]; print("full 60d (minus pos) ICA kurt:",ku, "#>30:",(ku>30).sum())
```