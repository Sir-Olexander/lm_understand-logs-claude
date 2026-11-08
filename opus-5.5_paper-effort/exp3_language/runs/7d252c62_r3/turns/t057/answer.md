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
a=F@T; b=F@OV5@T; c=F@OV1@T
def nrm(x): return x/np.linalg.norm(x,axis=1,keepdims=True)
Xa=nrm(a); Xb=nrm(b); Xc=nrm(c)
cab=np.sum(Xa*Xb,1)
sel=np.where(cab>0.5)[0]; clus=sel[[0,6,8,14,20,21]]
u=Xa[clus].mean(0); u/=np.linalg.norm(u)
# Better: define u as centroid of the densest a-peak
Cm=Xa@Xa.T; deg=(Cm>0.8).sum(1); p=np.argmax(deg)
u2=Xa[Cm[p]>0.8].mean(0); u2/=np.linalg.norm(u2)
print("cos(u,u2)", (u@u2).round(2))
ua=Xa@u2; ub=Xb@u2; uc=Xc@u2
grp=np.where(ua>0.7)[0]
print("n neurons current~u:", len(grp))
print("their prev-read cos with u:", np.round(np.sort(ub[grp]),2))
print("their prev2-read cos with u:", np.round(np.sort(uc[grp]),2))
# neurons whose prev-read ~u (prev = u)
grpb=np.where(ub>0.7)[0]; print("n neurons prev~u:", len(grpb), "their current-read cos with u:", np.round(np.sort(ua[grpb]),2))
grpc=np.where(uc>0.7)[0]; print("n neurons prev2~u:", len(grpc), "their current cos u", np.round(np.sort(ua[grpc]),2), "prev cos u", np.round(np.sort(ub[grpc]),2))
# relative strength of reads for current~u group
print("norms a,b,c for group:", np.round(np.linalg.norm(a[grp],axis=1),2), np.round(np.linalg.norm(b[grp],axis=1),2))
```