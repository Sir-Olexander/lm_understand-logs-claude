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
# need T in the w-scaled normalized coordinates: V (scaled) = Wv*w ; residual raw dir t -> normalized input t (LN acts per-dim, so a raw direction t enters as t then multiplied by w inside mats). OK use T directly.
F=W['W_fc.0']*W['ln2_w.0']
live=np.linalg.norm(W['W_fc.0'],axis=1)>1.0
F=F[live]
a=F@T; b=F@OV5@T; c=F@OV1@T
def cosr(x,y): return np.sum(x*y,1)/np.linalg.norm(x,axis=1)/np.linalg.norm(y,axis=1)
cab=cosr(a,b)
rng=np.random.default_rng(0)
print("cos(a,b) hist", np.histogram(cab,bins=np.linspace(-1,1,11))[0], "perm", np.histogram(cosr(a,b[rng.permutation(len(a))]),bins=np.linspace(-1,1,11))[0])
sel=np.where(cab>0.5)[0]
print("n sel",len(sel))
X=a[sel]/np.linalg.norm(a[sel],axis=1,keepdims=True)
Cm=X@X.T
np.set_printoptions(linewidth=250)
print(np.round(Cm,1))
print("neuron |a| for sel", np.round(np.linalg.norm(a[sel],axis=1),2))
```