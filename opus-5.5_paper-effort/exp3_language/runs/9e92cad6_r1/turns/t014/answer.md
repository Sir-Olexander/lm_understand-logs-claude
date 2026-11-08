```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def hm(nm,l,j):
    M=W[f"{nm}.{l}"]
    return M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
def OV(l,j): return hm("W_o",l,j)@(hm("W_v",l,j)*W[f"ln1_w.{l}"])
w0=W["ln1_w.0"]
U,s,Vt=np.linalg.svd(np.concatenate([hm("W_v",0,3)*w0, hm("W_v",0,0)*w0],0),full_matrices=False)
T=Vt[:22].T
w=W["ln1_w.1"]
axes=[]
for j in range(6):
    Q=hm("W_q",1,j)*w; K=hm("W_k",1,j)*w
    A=T.T@Q.T@K@T
    sym=(A+A.T)/2; e,V=np.linalg.eigh(sym)
    print(f"L1H{j} same sym eig", e[:5], "...", e[-3:])
    axes.append(V[:,0])
    sA=np.linalg.svd(A,compute_uv=False); print("     svals", sA[:8])
X=np.array(axes); print("overlap of most-negative axes across L1 heads\n", np.abs(X@X.T))
```