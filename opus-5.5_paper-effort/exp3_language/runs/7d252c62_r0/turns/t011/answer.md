```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
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
# positional subspace from Q,K of H1,H5,H0,H4 values (H0/H4 read pos)
X=np.vstack([H[0][1][0],H[0][1][1],H[0][5][0],H[0][5][1]])
U,S,Vt=np.linalg.svd(X,full_matrices=False)
for npos in [0,10,20,30,40,60]:
    P=Vt[:npos].T
    proj=np.eye(384)-P@P.T
    for nm,M in [('H1',OV[(0,1)]),('H5',OV[(0,5)])]:
        s=np.linalg.svd(M@proj,compute_uv=False)
        print(npos,nm,(s[:36]/s[0]).round(3))
    A=OV[(0,1)]@proj; B=OV[(0,5)]@proj
    s=np.linalg.svd(np.vstack([A/np.linalg.norm(A),B/np.linalg.norm(B)]),compute_uv=False)
    print(npos,'stack',(s[:36]/s[0]).round(3))
```