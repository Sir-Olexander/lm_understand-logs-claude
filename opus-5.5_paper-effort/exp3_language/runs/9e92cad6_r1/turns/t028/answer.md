```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def hm(nm,l,j):
    M=W[f"{nm}.{l}"]
    return M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
def OV(l,j): return hm("W_o",l,j)@(hm("W_v",l,j)*W[f"ln1_w.{l}"])
out=[21,39,58,99,184]
def lsub(M,k): U,s,_=np.linalg.svd(M,full_matrices=False); return U[:,:k]
cols=[lsub(OV(0,3),30),lsub(OV(0,0),20)]+[lsub(OV(0,j),4) for j in [1,2,4,5]]
E=np.zeros((384,5)); E[out,range(5)]=1; cols.append(E)
A=np.linalg.qr(np.concatenate(cols,1))[0]; Proj=np.eye(384)-A@A.T
_,_,Vt=np.linalg.svd((W["W_fc.0"]*W["ln2_w.0"])@Proj,full_matrices=False); T=Vt[:26].T
U3,_,_=np.linalg.svd(W['W_proj.3'],full_matrices=False); L=U3[:,1:25]   # token logit dirs 2-24 (approx)
C=L.T@OV(3,3)@T                     # copy map T->logit
B0=L.T@W['W_proj.0']@(W['W_fc.0']*W['ln2_w.0'])@T   # crude linearized MLP0 T->logit
print("C sv", np.linalg.svd(C,compute_uv=False))
print("B0 sv", np.linalg.svd(B0,compute_uv=False))
# how aligned are the output spaces of C and B0
Uc=np.linalg.svd(C)[0][:,:20]; Ub=np.linalg.svd(B0)[0][:,:20]
print("C-out vs B0-out cos", np.linalg.svd(Uc.T@Ub,compute_uv=False))
M=np.linalg.pinv(C,rcond=0.05)@B0
ev=np.linalg.eigvals(M); ev=ev[np.argsort(-np.abs(ev))]
print("succ-map eig", ev[:14].round(2))
print("trace", np.trace(M).round(2))
```