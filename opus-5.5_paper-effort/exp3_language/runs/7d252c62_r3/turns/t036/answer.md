```python
from sklearn.cluster import KMeans
def mats(l,j):
    w=W[f'ln1_w.{l}']
    s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def topright(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
Q1,K1,V1,O1=mats(0,1); Q5,K5,V5,O5=mats(0,5)
OV5=O5@V5; OV1=O1@V1
A=topright(OV1,26); B=topright(OV5,26)
P=(A@A.T+B@B.T)/2
ev,evec=np.linalg.eigh(P); idx=np.argsort(-ev); T=evec[:,idx[:28]]
F=W['W_fc.0']*W['ln2_w.0']
live=np.linalg.norm(W['W_fc.0'],axis=1)>1.0
F=F[live]
a=F@T
na=np.linalg.norm(a,axis=1)
X=a/na[:,None]
# pairwise cosine distribution
Cm=X@X.T
iu=np.triu_indices(len(X),1)
print("cos hist", np.histogram(Cm[iu],bins=np.linspace(-1,1,21))[0])
for k in (10,15,20,24,28,32,40):
    km=KMeans(n_clusters=k,n_init=5,random_state=0).fit(X)
    print(k, round(km.inertia_,1))
# also PCA of a (unnormalized)
s=np.linalg.svd(a,compute_uv=False); print("sv of a", np.round(s/s[0],2))
```