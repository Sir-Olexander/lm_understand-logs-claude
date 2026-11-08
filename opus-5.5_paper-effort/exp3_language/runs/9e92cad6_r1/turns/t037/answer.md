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
_,sF,Vt=np.linalg.svd((W["W_fc.0"]*W["ln2_w.0"])@Proj,full_matrices=False); T=Vt[:26].T
kbs=[]
for l in [1,2]:
    w=W[f"ln1_w.{l}"]; b=W[f"ln1_b.{l}"]
    for j in range(6):
        K=hm("W_k",l,j)*w; qb=hm("W_q",l,j)@b
        kbT=T.T@(K.T@qb); kbs.append(kbT)
KB=np.array(kbs); n=np.linalg.norm(KB,axis=1)
print("cos matrix of key-bias T vectors (L1H0-5, L2H0-5):\n", (KB/n[:,None])@(KB/n[:,None]).T)
print("sv of stacked:", np.linalg.svd(KB,compute_uv=False))
# coordinates in FC0-T basis (dir 0 = top FC0 token direction)
print("L1 key-bias T vecs in FC0 basis (first 10 coords):\n", KB[:6,:10])
```