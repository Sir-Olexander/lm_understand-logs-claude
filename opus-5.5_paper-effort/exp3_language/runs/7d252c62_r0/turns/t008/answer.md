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
Tb=rsv(np.vstack([OV[(0,1)],OV[(0,5)]]),24)
I=np.eye(384)
srcs={'tok':I,'H1':OV[(0,1)],'H5':OV[(0,5)],'H2':OV[(0,2)],'H3':OV[(0,3)]}
for l in range(1,4):
    for j in range(6):
        Wq,Wk,Wv,Wo=H[l][j]
        line=[]
        for qs in ['tok','H1','H5']:
            for ks in ['tok','H1','H5','H2','H3']:
                M=Tb.T@srcs[qs].T@Wq.T@Wk@srcs[ks]@Tb
                sv=np.linalg.svd(M,compute_uv=False)
                line.append(f'{qs[0:3]}>{ks}:{np.linalg.norm(M):.2f}/{np.trace(M)/sv.sum():+.2f}')
        print(f'L{l}H{j} '+' '.join(line))
    print()
```