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
def cosr(x,y): return np.sum(nrm(x)*nrm(y),1)
cab=cosr(a,b)
# doubled cluster from before
sel=np.where(cab>0.5)[0]; clus=sel[[0,6,8,14,20,21]]
u=Xa[clus].mean(0); u/=np.linalg.norm(u)
# local density peaks of a-directions: for each neuron, count neighbors cos>0.8
Cm=Xa@Xa.T; deg=(Cm>0.8).sum(1)
peaks=[]
for i in np.argsort(-deg):
    if all(Cm[i,p]<0.6 for p in peaks): peaks.append(i)
    if len(peaks)>=30: break
print("peak degrees", [int(deg[p]) for p in peaks])
print("cos(peak dir, u):", [round(float(Xa[p]@u),2) for p in peaks])
# Also look at b (prev-token reads): density peaks
Cb=Xb@Xb.T; degb=(Cb>0.8).sum(1)
peaksb=[]
for i in np.argsort(-degb):
    if all(Cb[i,p]<0.6 for p in peaksb): peaksb.append(i)
    if len(peaksb)>=15: break
print("prev-read peak degrees", [int(degb[p]) for p in peaksb])
print("cos(prev-peak dir, u):", [round(float(Xb[p]@u),2) for p in peaksb])
```