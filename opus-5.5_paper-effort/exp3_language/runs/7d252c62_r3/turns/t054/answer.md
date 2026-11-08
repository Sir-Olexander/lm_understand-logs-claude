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
Q1,K1,V1,O1=mats(0,1); Q5,K5,V5,O5=mats(0,5); Q3,K3,V3,O3=mats(0,3)
OV5=O5@V5; OV1=O1@V1
Fall=W['W_fc.0']*W['ln2_w.0']
live=np.where(np.linalg.norm(W['W_fc.0'],axis=1)>1.0)[0]
F=Fall[live]
a=F@T; b=F@OV5@T; c=F@OV1@T
def cosr(x,y): return np.sum(x*y,1)/np.linalg.norm(x,axis=1)/np.linalg.norm(y,axis=1)
cab=cosr(a,b); cac=cosr(a,c); cbc=cosr(b,c)
sel=np.where(cab>0.5)[0]
clus=sel[[0,6,8,14,20,21]]
u=np.mean(a[clus]/np.linalg.norm(a[clus],axis=1,keepdims=True),0); u/=np.linalg.norm(u)
print("cluster cos(a,b)",np.round(cab[clus],2)," cos(a,c)",np.round(cac[clus],2), " cos(b,c)", np.round(cbc[clus],2))
print("|a|,|b|,|c|", np.round(np.linalg.norm(a[clus],axis=1),2), np.round(np.linalg.norm(b[clus],axis=1),2), np.round(np.linalg.norm(c[clus],axis=1),2))
print("proj norms", np.round(np.linalg.norm(W['W_proj.0'][:,live[clus]],axis=0),2), "median live proj", np.round(np.median(np.linalg.norm(W['W_proj.0'][:,live],axis=0)),2))
# how many live neurons have a-direction close to u (any b)?
ca_u=(a@u)/np.linalg.norm(a,axis=1)
print("neurons with cos(a,u)>0.7:", np.sum(ca_u>0.7), " < -0.7:", np.sum(ca_u<-0.7))
# compare u (in T coords) with L0 key-token dirs: compute in raw space
uraw=T@u
for j in range(6):
    Q,K,V,O=mats(0,j); M=Q.T@K
    Uu,S,Vt=np.linalg.svd(M@T); kd=T@Vt[0]
    print(j,"cos(u, L0H%d key-tok dir)"%j, abs(uraw@kd).round(2))
# where does u sit in prev-token OV importance
print("|OV5 u|",np.linalg.norm(OV5@uraw).round(2), "avg", (np.linalg.norm(OV5@T)/np.sqrt(27)).round(2))
```