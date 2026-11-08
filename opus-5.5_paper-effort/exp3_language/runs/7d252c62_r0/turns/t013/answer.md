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
def rsv(M,k):
    U,S,Vt=np.linalg.svd(M,full_matrices=False); return Vt[:k].T
X=np.vstack([H[0][1][0],H[0][1][1],H[0][5][0],H[0][5][1]])
U,S,Vt=np.linalg.svd(X,full_matrices=False)
for npos in [20,30]:
    P=Vt[:npos].T; proj=np.eye(384)-P@P.T
    mats={'OV1':OV[(0,1)]@proj,'OV5':OV[(0,5)]@proj,'OV2':OV[(0,2)]@proj,'Vk':None}
    # induction QK maps (input side = key-token through OV5)
    for nm,(l,j) in [('L3H5',(3,5)),('L2H1',(2,1)),('L3H1',(3,1))]:
        Wq,Wk,Wv,Wo=H[l][j]
        mats[nm+'_key']=(Wk@OV[(0,5)])@proj     # rows 64, cols residual (key token dir)
        mats[nm+'_qry']=Wq@proj
    for k in [24,28,32]:
        A=rsv(mats['OV1'],k)
        for other in ['OV5','L3H5_key','L2H1_key','L3H1_key']:
            B=rsv(mats[other],k)
            c=np.linalg.svd(A.T@B,compute_uv=False)
            print(npos,k,'OV1 vs',other,c.round(2))
    print()
```