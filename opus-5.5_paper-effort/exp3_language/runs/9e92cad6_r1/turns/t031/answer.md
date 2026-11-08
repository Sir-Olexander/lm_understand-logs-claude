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
X=FC@T                       # neuron weights in T coords
nrm=np.linalg.norm(X,axis=1); full=np.linalg.norm(FC,axis=1)
print("neuron T-norm quantiles", np.quantile(nrm,[0.1,0.5,0.9,0.99]), " frac of weight in T (median)", np.median(nrm**2/full**2))
idx=np.argsort(-nrm)[:600]
Y=X[idx]/nrm[idx,None]
C=Y@Y.T
# greedy clustering: count directions with many neighbours
np.fill_diagonal(C,0)
for th in [0.8,0.9]:
    used=np.zeros(len(Y),bool); clusters=[]
    order=np.argsort(-(np.abs(C)>th).sum(1))
    for i in order:
        if used[i]: continue
        nb=np.where((np.abs(C[i])>th)&~used)[0]
        if len(nb)>=2:
            clusters.append(len(nb)+1); used[nb]=True; used[i]=True
    print(f"th={th}: n clusters(size>=3)={len(clusters)} sizes={sorted(clusters,reverse=True)[:40]}")
```