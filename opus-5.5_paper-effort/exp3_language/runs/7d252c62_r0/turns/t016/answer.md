```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
hd=64
def heads(l):
    lw=W[f'ln1_w.{l}']
    out=[]
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        Wq=W[f'W_q.{l}'][s]*lw; Wk=W[f'W_k.{l}'][s]*lw; Wv=W[f'W_v.{l}'][s]*lw; Wo=W[f'W_o.{l}'][:,s]
        out.append((Wq,Wk,Wv,Wo))
    return out
H={l:heads(l) for l in range(4)}
OV={(l,j):H[l][j][3]@H[l][j][2] for l in range(4) for j in range(6)}
Wfc=W['W_fc.0']*W['ln2_w.0']
s=np.linalg.svd(Wfc,compute_uv=False); print('Wfc0 sv',s[:70].round(2))
# remove attention output space of L0
O=np.hstack([np.linalg.svd(OV[(0,j)],full_matrices=False)[0][:,:k] for j,k in [(0,3),(1,24),(2,12),(3,12),(4,3),(5,24)]])
Qo=np.linalg.qr(O)[0]
X=np.vstack([H[0][1][0],H[0][1][1],H[0][5][0],H[0][5][1]])
U,S,Vt=np.linalg.svd(X,full_matrices=False)
P=Vt[:30].T
Bsp=np.linalg.qr(np.hstack([Qo,P]))[0]
proj=np.eye(384)-Bsp@Bsp.T
s2=np.linalg.svd(Wfc@proj,compute_uv=False); print('Wfc0 (no attn-out, no pos30) sv',s2[:60].round(2))
s3=np.linalg.svd(Wfc@(np.eye(384)-Qo@Qo.T),compute_uv=False); print('Wfc0 (no attn-out) sv',s3[:60].round(2))
# overlap of top Wfc input dirs with T
A_=OV[(0,1)]@(np.eye(384)-P@P.T); B_=OV[(0,5)]@(np.eye(384)-P@P.T)
U2,S2,Vt2=np.linalg.svd(np.vstack([A_/np.linalg.norm(A_),B_/np.linalg.norm(B_)]),full_matrices=False)
for k in [24,28,32]:
    T=Vt2[:k].T
    R=np.linalg.svd(Wfc@proj,full_matrices=False)[2][:k].T
    print(k,'Wfc-in(proj) vs T cos',np.linalg.svd(R.T@T,compute_uv=False).round(2))
```