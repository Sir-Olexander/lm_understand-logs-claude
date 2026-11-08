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
def rsv(M,k):
    U,S,Vt=np.linalg.svd(M,full_matrices=False); return Vt[:k].T
def lsv(M,k):
    U,S,Vt=np.linalg.svd(M,full_matrices=False); return U[:,:k]
for k in [20,24,28,32]:
    A=rsv(OV[(0,1)],k); B=rsv(OV[(0,5)],k)
    c=np.linalg.svd(A.T@B,compute_uv=False)
    print(k,'H1 vs H5 OV-in cosines',c.round(2))
# output overlap
for k in [10,20,24]:
    A=lsv(OV[(0,1)],k); B=lsv(OV[(0,5)],k)
    print(k,'H1 vs H5 OV-out cosines',np.linalg.svd(A.T@B,compute_uv=False).round(2))
# stacked spectrum
S=np.linalg.svd(np.vstack([OV[(0,1)]/np.linalg.norm(OV[(0,1)]),OV[(0,5)]/np.linalg.norm(OV[(0,5)])]),compute_uv=False)
print('stacked OV H1,H5 sv',S[:45].round(3))
# L1 summed sym QK eigen
Ms=sum((H[1][j][0].T@H[1][j][1]+H[1][j][1].T@H[1][j][0])/2 for j in range(6))
ev=np.linalg.eigvalsh(Ms)
print('L1 summed symQK most negative',ev[:40].round(3))
print('L1 summed symQK most positive',ev[::-1][:20].round(3))
```