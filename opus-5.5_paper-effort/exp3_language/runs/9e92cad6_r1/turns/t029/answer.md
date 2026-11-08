```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def hm(nm,l,j):
    M=W[f"{nm}.{l}"]
    return M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
def OV(l,j): return hm("W_o",l,j)@(hm("W_v",l,j)*W[f"ln1_w.{l}"])
out=[21,39,58,99,184]
def lsub(M,k): U,s,_=np.linalg.svd(M,full_matrices=False); return U[:,:k]
w0=W["ln1_w.0"]
Tr=np.concatenate([hm("W_v",0,3)*w0,hm("W_v",0,0)*w0],0)
Pr=np.concatenate([hm("W_q",0,3)*w0,hm("W_k",0,3)*w0,hm("W_v",0,5)*w0],0)
def frac(M,v): M=M/np.linalg.norm(M); return (np.linalg.norm(M@v)**2)*384
cols=[lsub(OV(0,3),30),lsub(OV(0,0),20)]+[lsub(OV(0,j),4) for j in [1,2,4,5]]+[lsub(W['W_proj.0'],60)]+[lsub(OV(1,j),12) for j in range(6)]
E=np.zeros((384,5)); E[out,range(5)]=1; cols.append(E)
A=np.linalg.qr(np.concatenate(cols,1))[0]; Proj=np.eye(384)-A@A.T
print("removed dims", A.shape[1])
for nm,M in [("FC1",W["W_fc.1"]*W["ln2_w.1"]),("L1qkv",np.concatenate([hm(n,1,j)*W["ln1_w.1"] for n in ["W_q","W_k","W_v"] for j in range(6)],0))]:
    _,s,Vt=np.linalg.svd(M@Proj,full_matrices=False)
    eT=np.array([frac(Tr,Vt[k]) for k in range(34)]); eP=np.array([frac(Pr,Vt[k]) for k in range(34)])
    print(nm,"sv",s[:34]); print("   T-enr",eT); print("   P-enr",eP)
```