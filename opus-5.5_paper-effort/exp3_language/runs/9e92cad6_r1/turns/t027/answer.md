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
# T26 from FC0
cols=[lsub(OV(0,3),30),lsub(OV(0,0),20)]+[lsub(OV(0,j),4) for j in [1,2,4,5]]
E=np.zeros((384,5)); E[out,range(5)]=1; cols.append(E)
A=np.linalg.qr(np.concatenate(cols,1))[0]; Proj=np.eye(384)-A@A.T
_,_,Vt=np.linalg.svd((W["W_fc.0"]*W["ln2_w.0"])@Proj,full_matrices=False); T26=Vt[:26].T
S={"T26":T26,"O03":lsub(OV(0,3),20),"O00":lsub(OV(0,0),10),"out":E,"MLP0":lsub(W["W_proj.0"],40),"MLP1":lsub(W["W_proj.1"],40),"MLP2":lsub(W["W_proj.2"],40)}
for j in range(6): S[f"O1{j}"]=lsub(OV(1,j),12)
for j in range(6): S[f"O2{j}"]=lsub(OV(2,j),12)
names=list(S)
print("     "+" ".join(f"{n:>5}" for n in names))
for (l,j) in [(3,0),(3,5),(3,3),(3,2),(3,1),(3,4),(2,4),(2,1)]:
    w=W[f"ln1_w.{l}"]; Q=hm("W_q",l,j)*w; K=hm("W_k",l,j)*w
    M=Q.T@K; Sym=(M+M.T)/2
    e,V=np.linalg.eigh(Sym)
    Vp=V[:,-8:]  # top positive eigvecs
    row=[]
    for n in names:
        U=S[n]; row.append(np.linalg.norm(U.T@Vp)**2/8/(U.shape[1]/384))
    print(f"L{l}H{j} +eigs {e[-4:]} | "+" ".join(f"{x:5.1f}" for x in row))
```