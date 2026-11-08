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
def rsv(M,k):
    U,S,Vt=np.linalg.svd(M,full_matrices=False); return Vt[:k].T
def ang(A,B):
    return np.linalg.svd(A.T@B,compute_uv=False)
# token subspace estimate: right sv of OV of H1,H5 (top 22)
T1=rsv(H[0][1][3]@H[0][1][2],22); T5=rsv(H[0][5][3]@H[0][5][2],22)
print('OV-in H1 vs H5 cosines', ang(T1,T5).round(2))
K1=rsv(H[0][1][1],10);K5=rsv(H[0][5][1],10);Q1=rsv(H[0][1][0],10);Q5=rsv(H[0][5][0],10)
print('K1 vs K5',ang(K1,K5).round(2)); print('Q1 vs Q5',ang(Q1,Q5).round(2))
T=np.linalg.qr(np.hstack([T1,T5]))[0]
Tb=rsv(np.vstack([H[0][1][3]@H[0][1][2],H[0][5][3]@H[0][5][2]]),24)
for nm,X in [('K1',K1),('K5',K5),('Q1',Q1),('Q5',Q5)]:
    print(nm,'vs Tb',ang(X,Tb).round(2))
# fraction of Frobenius norm of each L0 head's Wq, Wk, Wv in Tb
for j in range(6):
    Wq,Wk,Wv,Wo=H[0][j]
    f=lambda M: np.linalg.norm(M@Tb)**2/np.linalg.norm(M)**2
    print(f'L0H{j} frac in Tb (24d): q {f(Wq):.2f} k {f(Wk):.2f} v {f(Wv):.2f} OV {f(Wo@Wv):.2f}')
# L0H4 OV input dirs
R4=rsv(H[0][4][3]@H[0][4][2],3); R0=rsv(H[0][0][3]@H[0][0][2],3)
print('L0H4 in vs L0H0 in',ang(R4,R0).round(2))
print('L0H4 in vs Tb',ang(R4,Tb).round(2),' L0H0 in vs Tb',ang(R0,Tb).round(2))
print('L0H4 in vs K1',ang(R4,K1).round(2),'vs K5',ang(R4,K5).round(2))
```