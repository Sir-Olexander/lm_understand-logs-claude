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
Fall=W['W_fc.0']*W['ln2_w.0']
live=np.where(np.linalg.norm(W['W_fc.0'],axis=1)>1.0)[0]
F=Fall[live]; a=F@T
Xa=a/np.linalg.norm(a,axis=1,keepdims=True)
Cm=Xa@Xa.T; deg=(Cm>0.8).sum(1)
peaks=[]
for i in np.argsort(-deg):
    if all(Cm[i,p]<0.6 for p in peaks): peaks.append(i)
    if len(peaks)>=6: break
pdirs=[Xa[Cm[p]>0.8].mean(0) for p in peaks]; pdirs=[d/np.linalg.norm(d) for d in pdirs]
print("peak sizes", [int(deg[p]) for p in peaks])
# key token directions of L0 heads in T coords (dual: functional on token coords)
for j in range(6):
    Q,K,V_,O=mats(0,j); M=Q.T@K
    Uu,S,Vt=np.linalg.svd(M@T)
    kd=Vt[0]  # in T coords
    print(f"L0H{j} key-tok sv {S[:3].round(2)} cos with MLP0 peaks:", [round(float(abs(kd@d)),2) for d in pdirs])
# also L0H4 OV token functional: which token directions does the uniform head average?
for j in (4,0,3):
    Q,K,V_,O=mats(0,j)
    Uu,S,Vt=np.linalg.svd(O@V_@T)
    print(f"L0H{j} OV token-part sv {S[:3].round(2)} top dir cos with peaks:", [round(float(abs(Vt[0]@d)),2) for d in pdirs])
```