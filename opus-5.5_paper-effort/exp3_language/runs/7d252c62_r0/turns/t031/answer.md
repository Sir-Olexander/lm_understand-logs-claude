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
X=np.vstack([H[0][1][0],H[0][1][1],H[0][5][0],H[0][5][1]])
U,S,Vt=np.linalg.svd(X,full_matrices=False)
O=np.hstack([np.linalg.svd(OV[(0,j)],full_matrices=False)[0][:,:k] for j,k in [(0,3),(1,24),(2,12),(3,12),(4,3),(5,24)]])
Om=np.linalg.svd(W['W_proj.0'],full_matrices=False)[0][:,:60]
def rs(M,k): return np.linalg.svd(M,full_matrices=False)[2][:k].T
for npos in [20,40]:
    P=Vt[:npos].T
    B=np.linalg.qr(np.hstack([O,P,Om]))[0]; proj=np.eye(384)-B@B.T
    projP=np.eye(384)-P@P.T
    T1=rs(OV[(0,1)]@projP,34); T5=rs(OV[(0,5)]@projP,34)
    L1v=rs(np.vstack([H[1][j][2] for j in range(6)])@proj,34)
    L1q=rs(np.vstack([H[1][j][0] for j in range(6)])@proj,34)
    L2v=rs(np.vstack([H[l][j][2] for l in [2,3] for j in range(6)])@proj,34)
    for nm,(A,Bm) in {'OV1-L1v':(T1,L1v),'OV5-L1v':(T5,L1v),'OV1-L1q':(T1,L1q),'L1v-L1q':(L1v,L1q),'OV1-L23v':(T1,L2v),'L1v-L23v':(L1v,L2v)}.items():
        c=np.linalg.svd(A.T@Bm,compute_uv=False)
        print(npos,nm,c[16:34].round(2))
```