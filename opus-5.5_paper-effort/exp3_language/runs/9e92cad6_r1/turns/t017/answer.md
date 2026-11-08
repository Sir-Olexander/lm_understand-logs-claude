```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def hm(nm,l,j):
    M=W[f"{nm}.{l}"]
    return M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
def OV(l,j): return hm("W_o",l,j)@(hm("W_v",l,j)*W[f"ln1_w.{l}"])
w0=W["ln1_w.0"]; E=lambda nm,j: hm(nm,0,j)*w0
Pread=np.concatenate([E("W_q",3),E("W_k",3),E("W_k",0),E("W_v",5),E("W_v",1),E("W_v",2),E("W_v",4)],0)
_,_,VtP=np.linalg.svd(Pread,full_matrices=False); P=VtP[:30].T
Proj=np.eye(384)-P@P.T
_,sT,VtT=np.linalg.svd(np.concatenate([E("W_v",3),E("W_v",0)],0)@Proj,full_matrices=False)
U,s,Vt=np.linalg.svd(W['W_proj.3'],full_matrices=False)
O=OV(3,3)
for kT in [24,28,32,40]:
    T=VtT[:kT].T
    for kL in [24,28]:
        C=U[:,:kL].T@O@T
        print(f"kT={kT} kL={kL} copy-map sv", np.linalg.svd(C,compute_uv=False)[:kT])
# full: OV33 projected onto logit space, its input-side spectrum
C=U[:,:28].T@O
print("U28^T OV33 sv", np.linalg.svd(C,compute_uv=False))
```