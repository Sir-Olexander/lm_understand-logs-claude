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
F=Fall[live]; a=F@T; b=F@OV5@T; c=F@OV1@T
def nrm(x): return x/np.linalg.norm(x,axis=1,keepdims=True)
Xa=nrm(a); Xb=nrm(b); Xc=nrm(c)
Cm=Xa@Xa.T; deg=(Cm>0.8).sum(1)
peaks=[]
for i in np.argsort(-deg):
    if all(Cm[i,p]<0.6 for p in peaks): peaks.append(i)
    if len(peaks)>=20: break
for p in peaks:
    d=Xa[Cm[p]>0.8].mean(0); d/=np.linalg.norm(d)
    ga=np.where(Xa@d>0.7)[0]
    pb=Xb[ga]@d; pc=Xc[ga]@d
    nb=np.sum(Xb@d>0.7); 
    print(f"peak size {deg[p]:2d}: n(a~d)={len(ga):2d}  among them prev~d(>0.5): {np.sum(pb>0.5):2d}  prev<-0.5: {np.sum(pb<-0.5):2d} | prev2~d: {np.sum(pc>0.5):2d} | total n(b~d>0.7)={nb}")
```