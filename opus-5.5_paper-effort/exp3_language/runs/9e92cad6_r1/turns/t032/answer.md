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
X=FC@T; nrm=np.linalg.norm(X,axis=1)
fracT=nrm**2/np.linalg.norm(FC@Proj,axis=1)**2
idx=np.where(nrm>1.0)[0]; print("n strong-T neurons", len(idx), " their median fracT", np.median(fracT[idx]))
Y=X[idx]/nrm[idx,None]
C=Y@Y.T
th=0.85
used=np.zeros(len(Y),bool); cents=[]; sizes=[]
order=np.argsort(-(C>th).sum(1))
for i in order:
    if used[i]: continue
    nb=np.where((C[i]>th)&~used)[0]
    used[nb]=True; used[i]=True
    c=Y[nb].mean(0) if len(nb)>0 else Y[i]; cents.append(c/np.linalg.norm(c)); sizes.append(len(nb))
cents=np.array(cents); sizes=np.array(sizes)
print("clusters:", len(cents), "sizes", sizes[:60])
big=cents[sizes>=3]
G=big@big.T; print("n big", len(big))
print("max off-diag |cos| per big cluster", np.sort(np.abs(G-np.eye(len(big))).max(1)))
# effective rank of big cluster centroids
print("sv of big centroids", np.linalg.svd(big,compute_uv=False))
```