```python
def mats(l,j):
    w=W[f'ln1_w.{l}']
    s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def topright(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
Q1,K1,V1,O1=mats(0,1); Q5,K5,V5,O5=mats(0,5)
A=topright(O1@V1,26); B=topright(O5@V5,26)
P=(A@A.T+B@B.T)/2
ev,evec=np.linalg.eigh(P); idx=np.argsort(-ev); T=evec[:,idx[:20]]
U1,S1,_=np.linalg.svd(O1@V1@T); o1=U1[:,:12]
U5,S5,_=np.linalg.svd(O5@V5@T); o5=U5[:,:12]
Tt=T[:,:12] if False else np.linalg.svd(T)[0][:,:20]
subs={'T':T,'o1':o1,'o5':o5}
def enrich(M,X,Y):
    return (np.linalg.norm(X.T@M@Y)**2/np.linalg.norm(M)**2)/(X.shape[1]*Y.shape[1]/384**2)
print("    "+" ".join(f"{a}/{b:3s}" for a in subs for b in subs))
for l in range(1,4):
    for j in range(6):
        Q,K,V,O=mats(l,j); M=Q.T@K
        print(f"L{l}H{j} "+" ".join(f"{enrich(M,subs[a],subs[b]):5.1f}" for a in subs for b in subs))
```