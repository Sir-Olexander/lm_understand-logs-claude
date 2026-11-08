```python
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
np.set_printoptions(linewidth=220, precision=2, suppress=True)
def head(l,h):
    s=slice(64*h,64*h+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
q,k,v,o=head(0,2); A0=q.T@k; OVp=o@v
U0,s0,Vt0=np.linalg.svd(A0); Ppos=np.linalg.qr(np.hstack([U0[:,:18],Vt0[:18].T]))[0][:,:24]
U2,s2,Vt2=np.linalg.svd(OVp); T=Vt2[:32].T; T=T-Ppos@(Ppos.T@T); T=np.linalg.qr(T)[0]
F=W['W_fc.0']*W['ln2_w.0']
X=F@T     # 1536 x 32
print("sv of MLP0 read on T:",np.linalg.svd(X,compute_uv=False))
Xf=F
sf=np.linalg.svd(Xf,compute_uv=False); print("sv of W_fc.0*ln2w (full):",sf[:60])
for nc in [24,27,30]:
    ica=FastICA(n_components=nc,random_state=0,max_iter=2000,whiten='unit-variance')
    S=ica.fit_transform(X)
    ku=np.sort(kurtosis(S,axis=0))[::-1]
    print(nc,"ICA source kurtosis:",np.round(ku,1))
```