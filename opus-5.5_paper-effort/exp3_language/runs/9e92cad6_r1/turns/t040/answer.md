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
def frac(M,v): M=M/np.linalg.norm(M); return (np.linalg.norm(M@v)**2)*384
def fracL(M,v): M=M/np.linalg.norm(M); return (np.linalg.norm(M.T@v)**2)*384   # output-side
QKall=np.concatenate([hm(n,0,j)*w0 for n in ["W_q","W_k"] for j in range(6)],0)
V03=np.concatenate([hm("W_v",0,3)*w0,hm("W_v",0,0)*w0],0)
Wo0=W["W_o.0"]   # all head outputs (raw W_o columns span)
ks=[25,15,3,3,3,3]
cols=[lsub(OV(0,j),k) for j,k in zip([3,0,1,2,4,5],ks)]
E=np.zeros((384,5)); E[out,range(5)]=1; cols.append(E)
A=np.linalg.qr(np.concatenate(cols,1))[0]; Proj=np.eye(384)-A@A.T
_,s,Vt=np.linalg.svd((W["W_fc.0"]*W["ln2_w.0"])@Proj,full_matrices=False)
print("dir  sv   V03  QKall  Wo0-out  OVall-out")
OVall=np.concatenate([OV(0,j) for j in range(6)],1)
for k in range(20,40):
    v=Vt[k]
    print(f"{k:3d} {s[k]:5.2f} {frac(V03,v):5.2f} {frac(QKall,v):5.2f} {fracL(Wo0,v):6.2f} {fracL(OVall,v):6.2f}")
```