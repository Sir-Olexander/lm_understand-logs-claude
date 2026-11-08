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
H={0:heads(0)}
X=np.vstack([H[0][1][0],H[0][1][1],H[0][5][0],H[0][5][1]])
U,S,Vt=np.linalg.svd(X,full_matrices=False)
for npos in [10,30,60]:
    P=Vt[:npos].T
    pp={j:P.T@(H[0][j][0].T@H[0][j][1])@P for j in range(6)}
    print('npos',npos)
    for j in range(6):
        print(f'  H{j} |pp|={np.linalg.norm(pp[j]):.2f} cos with H0..H5:',' '.join(f'{np.sum(pp[j]*pp[k])/np.linalg.norm(pp[j])/np.linalg.norm(pp[k]):+.2f}' for k in range(6)))
    # least squares: H2pp, H3pp, H0pp as combination of H1pp,H5pp, and their transposes
    for j in [0,2,3]:
        Bm=np.stack([pp[1].ravel(),pp[5].ravel(),pp[1].T.ravel(),pp[5].T.ravel(),np.eye(npos).ravel()],1)
        coef,res,_,_=np.linalg.lstsq(Bm,pp[j].ravel(),rcond=None)
        fit=1-np.sum((Bm@coef-pp[j].ravel())**2)/np.sum(pp[j]**2)
        print(f'  H{j} ~ a*H1 + b*H5 + c*H1^T + d*H5^T + e*I : coef',coef.round(3),' R2=%.2f'%fit)
```