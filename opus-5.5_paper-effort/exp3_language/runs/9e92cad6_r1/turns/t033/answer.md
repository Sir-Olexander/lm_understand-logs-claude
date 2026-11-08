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
_,s,Vt=np.linalg.svd(FC@Proj,full_matrices=False); T=Vt[:26].T
w0=W["ln1_w.0"]
# key token-direction of segment heads: K^T (Q h) dominant -> use top right sing vec of K restricted to T
for j in [1,2,4,5,0,3]:
    K=hm("W_k",0,j)*w0; Q=hm("W_q",0,j)*w0
    M=Q.T@K   # 384x384 bilinear
    # key-side direction: top right singular vector of M projected to T
    U_,S_,Vt_=np.linalg.svd(M)
    kT=T.T@Vt_[0]; qT=T.T@U_[:,0]
    print(f"L0H{j}: QK top sv {S_[:3]}, key-dir |in T| {np.linalg.norm(kT):.2f}, query-dir |in T| {np.linalg.norm(qT):.2f}")
    c=np.abs(Vt[:6]@Vt_[0]); print("     key-dir overlap with FC0 T-dirs 0..5:", c)
# top FC0 T-direction vs key dirs across heads
kd=[]
for j in [1,2,4]:
    K=hm("W_k",0,j)*w0; Q=hm("W_q",0,j)*w0; U_,S_,Vt_=np.linalg.svd(Q.T@K); v=T@(T.T@Vt_[0]); kd.append(v/np.linalg.norm(v))
kd=np.array(kd); print("T-part key dirs cos among H1,H2,H4:\n", kd@kd.T)
```