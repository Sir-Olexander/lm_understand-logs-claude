```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
from scipy.linalg import polar
def hm(nm,l,j):
    M=W[f"{nm}.{l}"]
    return M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
w0=W["ln1_w.0"]
Q3=hm("W_q",0,3)*w0; K3=hm("W_k",0,3)*w0
for kP in [12,16,20]:
    _,_,Vt=np.linalg.svd(np.concatenate([Q3,K3],0),full_matrices=False); P=Vt[:kP].T
    B3=P.T@Q3.T@K3@P
    R,S=polar(B3)   # B3 = R S
    es=np.linalg.eigvalsh((S+S.T)/2)
    print(f"kP={kP}: S eig (scalar-ness)", es[::-1][:8], "... min", es.min())
    ang=np.degrees(np.angle(np.linalg.eigvals(R))); print("   R eig angles", np.sort(np.abs(ang))[::2])
    Rp=[np.linalg.matrix_power(R,d) if d>=0 else np.linalg.matrix_power(R.T,-d) for d in range(-3,8)]
    for j in [0,3,1,2,4]:
        Bj=P.T@(hm("W_q",0,j)*w0).T@(hm("W_k",0,j)*w0)@P
        X=np.array([m.ravel() for m in Rp]).T
        coef,res,_,_=np.linalg.lstsq(X,Bj.ravel(),rcond=None)
        fit=1-np.linalg.norm(X@coef-Bj.ravel())**2/np.linalg.norm(Bj)**2
        print(f"   L0H{j}: R^d coefs d=-3..7:", coef.round(3), f"R2={fit:.2f}")
```