```python
import numpy as np
np.set_printoptions(linewidth=220, precision=2, suppress=True)
def head(l,h):
    s=slice(64*h,64*h+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def hb(l,h):
    s=slice(64*h,64*h+64); b=W[f'ln1_b.{l}']
    return W[f'W_q.{l}'][s]@b, W[f'W_k.{l}'][s]@b
def orth(M,k): U,s,Vt=np.linalg.svd(M,full_matrices=False); return U[:,:k]
OVb=sum(head(0,h)[3]@head(0,h)[2] for h in [0,1,3,4,5])
Bs=orth(OVb,4)
Ub,sb,_=np.linalg.svd(OVb); print("summed broad OV sv:",sb[:8])
q,k,v,o=head(0,2); A=q.T@k
U0,s0,Vt0=np.linalg.svd(A); Ppos=np.linalg.qr(np.hstack([U0[:,:18],Vt0[:18].T]))[0][:,:24]
U2,s2,Vt2=np.linalg.svd(o@v); T=Vt2[:28].T; T=T-Ppos@(Ppos.T@T); T=np.linalg.qr(T)[0]
P0=orth(o@v,24)
for l in [1,2,3]:
    for h in range(6):
        q,k,v,o=head(l,h); qb,kb=hb(l,h); A=q.T@k
        cqb=np.linalg.norm(qb@k@Bs)          # const query x key b-out
        Mbb=Bs.T@A@Bs; evb=np.linalg.eigvals(Mbb)
        cTB=np.linalg.norm(T.T@A@Bs); cPB=np.linalg.norm(P0.T@A@Bs)
        print(f"L{l}H{h}: qbias*k_bout {cqb:5.2f} | bout-bout {np.linalg.norm(Mbb):5.2f} eig {np.round(evb.real,2)} | tok->bout {cTB:.2f} prevOut->bout {cPB:.2f} | qb {np.linalg.norm(qb):.1f}")
```