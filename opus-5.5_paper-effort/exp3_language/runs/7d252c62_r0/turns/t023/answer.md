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
Qo=np.linalg.qr(O)[0]
Wfc=W['W_fc.0']*W['ln2_w.0']
L1q=np.vstack([H[1][j][0] for j in range(6)]+[H[1][j][1] for j in range(6)])
def rs(M,k): return np.linalg.svd(M,full_matrices=False)[2][:k].T
for npos in [0,10,20,30,40,50,70]:
    P=Vt[:npos].T
    projP=np.eye(384)-P@P.T
    A_=OV[(0,1)]@projP; B_=OV[(0,5)]@projP
    Bsp=np.linalg.qr(np.hstack([Qo,P]))[0] if npos>0 else Qo
    projB=np.eye(384)-Bsp@Bsp.T
    k=34
    T1=rs(A_,k); T5=rs(B_,k); F=rs(Wfc@projB,k); Lq=rs(L1q@projB,k)
    c15=np.linalg.svd(T1.T@T5,compute_uv=False); c1F=np.linalg.svd(T1.T@F,compute_uv=False); c5F=np.linalg.svd(T5.T@F,compute_uv=False)
    c1L=np.linalg.svd(T1.T@Lq,compute_uv=False); cFL=np.linalg.svd(F.T@Lq,compute_uv=False)
    cnt=lambda c,t: int((c>t).sum())
    print(npos,'#cos>0.5: OV1-OV5',cnt(c15,.5),' OV1-Wfc',cnt(c1F,.5),' OV5-Wfc',cnt(c5F,.5),' OV1-L1qk',cnt(c1L,.5),' Wfc-L1qk',cnt(cFL,.5),
          '| >0.7:',cnt(c15,.7),cnt(c1F,.7),cnt(c5F,.7),cnt(c1L,.7),cnt(cFL,.7))
    if npos in (20,40):
        print('   OV1-Wfc',c1F[18:32].round(2)); print('   OV1-L1qk',c1L[18:32].round(2)); print('   Wfc-L1qk',cFL[18:32].round(2))
```