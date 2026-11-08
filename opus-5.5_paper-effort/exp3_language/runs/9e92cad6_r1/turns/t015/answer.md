```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def hm(nm,l,j):
    M=W[f"{nm}.{l}"]
    return M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
w0=W["ln1_w.0"]
E=lambda nm,j: hm(nm,0,j)*w0
Pread=np.concatenate([E("W_q",3),E("W_k",3),E("W_k",0),E("W_v",5),E("W_v",1),E("W_v",2),E("W_v",4)],0)
U,s,Vt=np.linalg.svd(Pread,full_matrices=False)
print("P readers sv", s[:60])
for kP in [20,30,40]:
    P=Vt[:kP].T
    Proj=np.eye(384)-P@P.T
    for nm,M in [("V3",E("W_v",3)),("V0",E("W_v",0)),("V3+V0",np.concatenate([E("W_v",3),E("W_v",0)],0))]:
        s2=np.linalg.svd(M@Proj,compute_uv=False)
        print(f"kP={kP} {nm} after removing P:", s2[:36])
```