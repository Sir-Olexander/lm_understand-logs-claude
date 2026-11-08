```python
def mats(l,j):
    w=W[f'ln1_w.{l}']
    s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def rs(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
V={j:W['W_v.0'][64*j:64*j+64] for j in range(6)}
Qr=np.vstack([W['W_q.0'][64:128],W['W_k.0'][64:128],W['W_q.0'][320:384],W['W_k.0'][320:384]])
Pp=rs(Qr,30); R=np.eye(384)-Pp@Pp.T
A=rs(V[1]@R,36); B=rs(V[5]@R,36)
U,cs,Vt=np.linalg.svd(A.T@B); T=A@U[:,:27]
Q1,K1,V1,O1=mats(0,1); Q5,K5,V5,O5=mats(0,5)
OV5=O5@V5; OV1=O1@V1
Fall=W['W_fc.0']*W['ln2_w.0']
live=np.where(np.linalg.norm(W['W_fc.0'],axis=1)>1.0)[0]
F=Fall[live]
a=F@T; b=F@OV5@T
na=np.linalg.norm(a,axis=1)
X=a/na[:,None]
Cm=X@X.T
# greedy clustering by density
remaining=set(range(len(X)))
clusters=[]
deg=(Cm>0.75).sum(1)
order=np.argsort(-deg)
assigned=np.full(len(X),-1)
for i in order:
    if assigned[i]>=0: continue
    members=[k for k in range(len(X)) if assigned[k]<0 and Cm[i,k]>0.75]
    if len(members)>=3:
        for k in members: assigned[k]=len(clusters)
        clusters.append(members)
print("cluster sizes", [len(c) for c in clusters], "unassigned", np.sum(assigned<0))
cent=np.array([X[c].mean(0)/np.linalg.norm(X[c].mean(0)) for c in clusters])
print("centroid mutual cos:\n", np.round(cent@cent.T,1))
# for each cluster, mean cos(a,b) of members (current same as prev?)
def cosr(x,y): return np.sum(x*y,1)/np.linalg.norm(x,axis=1)/np.linalg.norm(y,axis=1)
cab=cosr(a,b)
print("mean cos(a,b) per cluster", [round(float(np.mean(cab[c])),2) for c in clusters])
np.save('/tmp/x.npy',cent)
```