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
    k=20 if j in (1,5) else 3
    subs[f'o0.{j}']=np.linalg.qr(O@V@T if j in (1,5) else topleft(O@V,k))[0]
for j in range(6):
    Q,K,V,O=mats(1,j)
    subs[f'o1.{j}']=topleft(O@V,20)
np.save('/tmp/T.npy',T)
def enrich(M,X,Y):
    return (np.linalg.norm(X.T@M@Y)**2/np.linalg.norm(M)**2)/(X.shape[1]*Y.shape[1]/384**2)
for l in range(1,4):
    for j in range(6):
        Q,K,V,O=mats(l,j); M=Q.T@K
        out=[]
        for qn in ['T','o0.1','o0.5']:
            for kn in ['T','o0.1','o0.5','o0.4','o0.0']+([f'o1.{i}' for i in range(6)] if l>1 else []):
                e=enrich(M,subs[qn],subs[kn])
                if e>3: out.append(f"q{qn}/k{kn}:{e:.1f}")
        print(f"L{l}H{j}", " ".join(out))
```