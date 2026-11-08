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
P=Vt[:20].T; projP=np.eye(384)-P@P.T
A_=OV[(0,1)]@projP; B_=OV[(0,5)]@projP
U2,S2,Vt2=np.linalg.svd(np.vstack([A_/np.linalg.norm(A_),B_/np.linalg.norm(B_)]),full_matrices=False)
T=Vt2[:27].T
I=np.eye(384)
src={'tok':I,'H5':OV[(0,5)],'H1':OV[(0,1)]}
for l in range(0,4):
    for j in range(6):
        Wq,Wk,_,_=H[l][j]
        out=[]
        for qn in ['tok','H5']:
            for kn in ['tok','H5','H1']:
                if l==0 and (qn!='tok' or kn!='tok'): continue
                M=T.T@src[qn].T@Wq.T@Wk@src[kn]@T
                s=np.linalg.svd(M,compute_uv=False)
                out.append(f'{qn}>{kn}: s1={s[0]:.2f} s1/s2={s[0]/s[1]:.1f} s1^2/sum={s[0]**2/np.sum(s**2):.2f}')
        print(f'L{l}H{j} '+' | '.join(out))
```