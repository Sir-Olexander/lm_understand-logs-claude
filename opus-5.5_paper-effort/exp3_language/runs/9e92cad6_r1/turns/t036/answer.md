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
_,_,Vt=np.linalg.svd((W["W_fc.0"]*W["ln2_w.0"])@Proj,full_matrices=False); T=Vt[:26].T
for l in range(4):
    w=W[f"ln1_w.{l}"]; b=W[f"ln1_b.{l}"]
    for j in range(6):
        Q=hm("W_q",l,j)*w; K=hm("W_k",l,j)*w
        qb=hm("W_q",l,j)@b
        kb=K.T@qb                      # key preference vector (residual-space) from constant query
        kbT=T.T@kb
        # query-dependent token->token part strength
        M=T.T@Q.T@K@T
        print(f"L{l}H{j}: |keybias|={np.linalg.norm(kb):6.2f} |keybias in T|={np.linalg.norm(kbT):5.2f} (frac {np.linalg.norm(kbT)**2/np.linalg.norm(kb)**2:.3f})  |QK T-T|={np.linalg.norm(M):5.2f} top sv {np.linalg.svd(M,compute_uv=False)[:3]}")
```