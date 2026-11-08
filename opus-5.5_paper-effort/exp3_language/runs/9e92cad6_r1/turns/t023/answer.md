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
A=np.linalg.qr(np.concatenate(cols,1))[0]
Proj=np.eye(384)-A@A.T
FC=(W["W_fc.0"]*W["ln2_w.0"])@Proj
s=np.linalg.svd(FC,compute_uv=False); print("FC0 reading outside attn-outputs, sv:", s[:50])
w0=W["ln1_w.0"]
V=np.concatenate([hm("W_v",0,3)*w0,hm("W_v",0,0)*w0],0)@Proj
s=np.linalg.svd(V,compute_uv=False); print("V0/V3 outside:", s[:40])
_,_,Vt1=np.linalg.svd(FC,full_matrices=False); _,_,Vt2=np.linalg.svd(V,full_matrices=False)
for a,b in [(24,40),(30,40),(30,60)]:
    print(a,b, np.linalg.svd(Vt2[:a]@Vt1[:b].T,compute_uv=False)[16:])
```