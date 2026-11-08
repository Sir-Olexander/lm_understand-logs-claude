```python
def mats(l,j):
    w=W[f'ln1_w.{l}']
    s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def topright(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
Q1,K1,V1,O1=mats(0,1); Q5,K5,V5,O5=mats(0,5)
OV5=O5@V5; OV1=O1@V1
A=topright(OV1,30); B=topright(OV5,30)
P=(A@A.T+B@B.T)/2
ev,evec=np.linalg.eigh(P); idx=np.argsort(-ev); T=evec[:,idx[:26]]
F=W['W_fc.0']*W['ln2_w.0']
live=np.linalg.norm(W['W_fc.0'],axis=1)>1.0
F=F[live]
a=F@T; b=F@OV5@T; c=F@OV1@T
def cosr(x,y): return np.sum(x*y,1)/np.linalg.norm(x,axis=1)/np.linalg.norm(y,axis=1)
cab=cosr(a,b); cac=cosr(a,c); cbc=cosr(b,c)
print("cos(a,b) hist", np.histogram(cab,bins=np.linspace(-1,1,11))[0])
print("cos(a,c) hist", np.histogram(cac,bins=np.linspace(-1,1,11))[0])
print("cos(b,c) hist", np.histogram(cbc,bins=np.linspace(-1,1,11))[0])
na=np.linalg.norm(a,axis=1); nb=np.linalg.norm(b,axis=1); nc=np.linalg.norm(c,axis=1)
sel=(cab>0.6)&(cac>0.6)
print("neurons with a~b~c:", sel.sum(), "norms", np.round(na[sel],2), np.round(nb[sel],2), np.round(nc[sel],2))
if sel.sum()>1:
    X=a[sel]/na[sel,None]; print("mutual cos among them", np.round(X@X.T,2))
# baseline: shuffle neurons
rng=np.random.default_rng(0)
perm=rng.permutation(len(a))
print("baseline cos(a,b_perm) hist", np.histogram(cosr(a,b[perm]),bins=np.linspace(-1,1,11))[0])
```