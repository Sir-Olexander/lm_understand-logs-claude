```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def hm(nm,l,j):
    M=W[f"{nm}.{l}"]
    return M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
def OV(l,j): return hm("W_o",l,j)@(hm("W_v",l,j)*W[f"ln1_w.{l}"])
out=[21,39,58,99,184]
cols=[]
for j,k in [(3,30),(0,20),(1,4),(2,4),(4,4),(5,4)]:
    U,s,_=np.linalg.svd(OV(0,j)); cols.append(U[:,:k])
E=np.zeros((384,len(out))); E[out,range(len(out))]=1; cols.append(E)
A=np.linalg.qr(np.concatenate(cols,1))[0]; Proj=np.eye(384)-A@A.T
w0=W["ln1_w.0"]
FC=(W["W_fc.0"]*W["ln2_w.0"])@Proj
_,s,Vt=np.linalg.svd(FC,full_matrices=False)
Pr=np.concatenate([hm("W_q",0,3)*w0,hm("W_k",0,3)*w0,hm("W_k",0,0)*w0,hm("W_q",0,0)*w0,hm("W_v",0,5)*w0,hm("W_v",0,1)*w0,hm("W_v",0,2)*w0,hm("W_v",0,4)*w0],0)
Tr=np.concatenate([hm("W_v",0,3)*w0,hm("W_v",0,0)*w0],0)
def frac(M,v): # energy of M along v relative to average direction
    M=M/np.linalg.norm(M); return (np.linalg.norm(M@v)**2)*384
for k in range(32):
    v=Vt[k]
    print(f"FC dir {k:2d} sv {s[k]:5.2f}  P-reader enr {frac(Pr,v):5.2f}  T-reader enr {frac(Tr,v):5.2f}  L0H3QK enr {frac(np.concatenate([hm('W_q',0,3)*w0,hm('W_k',0,3)*w0]),v):5.2f}")
```