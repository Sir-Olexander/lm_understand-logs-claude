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
X=(W['W_fc.0']*W['ln2_w.0'])@T.T     # 1536 x 28
from scipy.stats import kurtosis
best=None
for seed in range(5):
    ica=FastICA(n_components=28,random_state=seed,max_iter=2000,whiten='unit-variance')
    Sx=ica.fit_transform(X)       # neurons x comps
    k=kurtosis(Sx,axis=0)
    print("seed",seed,"kurtosis sorted", np.round(np.sort(k)[::-1],1))
    if best is None or k.mean()>best[0]: best=(k.mean(),ica,Sx)
_,ica,Sx=best
A=ica.mixing_   # 28 x 28 : columns = component directions in T coords
# Special token directions from L0 QK
QKs=np.vstack([H[(0,j)][i]/norm(H[(0,j)][i]) for j in (1,5) for i in (0,1)])
P=svd(QKs,full_matrices=False)[2][:30]; Pp=P-(P@T.T)@T; Pp=svd(Pp,full_matrices=False)[2][:30]
sp=[]
for j in (0,1,3,5):
    Q,K,V,O=H[(0,j)]; B=Q.T@K
    M=Pp@B@T.T; u,s,vt=svd(M); sp+= [s[0]*vt[0], s[1]*vt[1]]
    M=T@B@Pp.T; u,s,vt=svd(M); sp+= [s[0]*u[:,0], s[1]*u[:,1]]
    M=T@B@T.T; u,s,vt=svd(M); sp+= [s[0]*u[:,0],s[0]*vt[0]]
SP=svd(np.array(sp),full_matrices=False)[2][:2]    # 2 x 28 in T coords
# unmixing: express in component coords: dual basis = rows of ica.components_ (after whitening)  -> use pinv(A)
Ainv=np.linalg.pinv(A)
for i in range(2):
    c=SP[i]@A   # projection onto component directions (as linear functional)
    c2=Ainv@SP[i]
    print("special dir",i," |<dir,A_k>| normalized:",np.round(np.abs(c)/np.abs(c).max(),2))
    print("        coeffs in A basis normalized:",np.round(np.abs(c2)/np.abs(c2).max(),2))
```