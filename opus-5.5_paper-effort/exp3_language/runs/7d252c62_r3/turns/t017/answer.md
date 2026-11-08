```python
def mats(l,j):
    w=W[f'ln1_w.{l}']
    s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def topright(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
def topleft(M,k):
    U,S,Vt=np.linalg.svd(M); return U[:,:k]
Q1,K1,V1,O1=mats(0,1); Q5,K5,V5,O5=mats(0,5)
A=topright(O1@V1,26); B=topright(O5@V5,26)
P=(A@A.T+B@B.T)/2
ev,evec=np.linalg.eigh(P); idx=np.argsort(-ev); T=evec[:,idx[:20]]
subs={'T':T}
for j in range(6):
    Q,K,V,O=mats(0,j)
    subs[f'o0.{j}']=np.linalg.qr(O@V@T)[0] if j in (1,5) else topleft(O@V,2)
Qs=[]
for j in (1,5):
    Q,K,V,O=mats(0,j); Qs.append(topright(Q,30)); Qs.append(topright(K,30))
Pp=sum(x@x.T for x in Qs)/4
ev2,evec2=np.linalg.eigh(Pp); idx2=np.argsort(-ev2); Ppos=evec2[:,idx2[:24]]
Ppos=np.linalg.qr(Ppos-T@(T.T@Ppos))[0]; subs['pos']=Ppos
def fr(Mat,X):
    return (np.linalg.norm(Mat@X)**2/np.linalg.norm(Mat)**2)/(X.shape[1]/384)
for l in range(4):
    F=W[f'W_fc.{l}']*W[f'ln2_w.{l}']
    print(f"MLP{l} read:", " ".join(f"{nm}:{fr(F,X):.1f}" for nm,X in subs.items()))
    Pr=W[f'W_proj.{l}']
    print(f"MLP{l} write:", " ".join(f"{nm}:{fr(Pr.T,X):.1f}" for nm,X in subs.items()))
# Also: fraction of W_fc.0 energy explained by T+o0.1+o0.5+pos+o0.4 total
F=W['W_fc.0']*W['ln2_w.0']
Ball=np.linalg.qr(np.hstack([subs[k] for k in subs]))[0]
print("dims",Ball.shape, "frac energy W_fc.0 in union", np.linalg.norm(F@Ball)**2/np.linalg.norm(F)**2)
```