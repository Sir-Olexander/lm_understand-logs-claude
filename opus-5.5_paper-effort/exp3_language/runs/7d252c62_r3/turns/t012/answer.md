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
Qs=[]
for j in (1,5):
    Q,K,V,O=mats(0,j); Qs.append(topright(Q,30)); Qs.append(topright(K,30))
Pp=sum(x@x.T for x in Qs)/4
ev2,evec2=np.linalg.eigh(Pp); idx2=np.argsort(-ev2); Ppos=evec2[:,idx2[:24]]
# make Ppos orthogonal to T
Ppos=np.linalg.qr(Ppos-T@(T.T@Ppos))[0]
subs={'T':T,'pos':Ppos}
for j in range(6):
    Q,K,V,O=mats(0,j)
    subs[f'o0.{j}']=np.linalg.qr(O@V@T)[0] if j in (1,5) else topleft(O@V,2)
def enrich(M,X,Y):
    return (np.linalg.norm(X.T@M@Y)**2/np.linalg.norm(M)**2)/(X.shape[1]*Y.shape[1]/384**2)
def frac(M,X,Y): return np.linalg.norm(X.T@M@Y)**2/np.linalg.norm(M)**2
names=['T','pos','o0.1','o0.5','o0.4','o0.0','o0.2','o0.3']
for l in range(1,4):
  for j in range(6):
    Q,K,V,O=mats(l,j); M=Q.T@K
    row=[]
    for qn in names:
        for kn in names:
            f=frac(M,subs[qn],subs[kn])
            if f>0.03: row.append(f"q{qn}/k{kn}:{f:.2f}")
    # key pref
    b=W[f'ln1_b.{l}']; s=slice(64*j,64*j+64)
    kp=K.T@(W[f'W_q.{l}'][s]@b)
    kpr=" ".join(f"{n}:{np.linalg.norm(subs[n].T@kp)/np.linalg.norm(kp):.2f}" for n in names)
    print(f"L{l}H{j}", " ".join(row), "| keypref", f"{np.linalg.norm(kp)/8:.2f}", kpr)
```