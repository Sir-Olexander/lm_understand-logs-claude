```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def hm(nm,l,j):
    M=W[f"{nm}.{l}"]
    return M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
def OV(l,j): return hm("W_o",l,j)@(hm("W_v",l,j)*W[f"ln1_w.{l}"])
out=[21,39,58,99,184]
def lsub(M,k): U,s,_=np.linalg.svd(M,full_matrices=False); return U[:,:k]
cols=[lsub(OV(0,j),k) for j,k in zip([3,0,1,2,4,5],[25,15,3,3,3,3])]
E=np.zeros((384,5)); E[out,range(5)]=1; cols.append(E)
A=np.linalg.qr(np.concatenate(cols,1))[0]; Proj=np.eye(384)-A@A.T
FC=(W["W_fc.0"]*W["ln2_w.0"])
_,s,Vt=np.linalg.svd(FC@Proj,full_matrices=False)
U3,_,_=np.linalg.svd(W['W_proj.3'],full_matrices=False)
for kT in [24,26,28,30]:
    T=Vt[:kT].T
    print(f"kT={kT} OV33@T sv:", np.linalg.svd(OV(3,3)@T,compute_uv=False))
    print(f"      U28^T OV33 T sv:", np.linalg.svd(U3[:,:28].T@OV(3,3)@T,compute_uv=False))
    print(f"      OV03@T sv:", np.linalg.svd(OV(0,3)@T,compute_uv=False))
```