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
def enrich(M,X,Y):
    return (np.linalg.norm(X.T@M@Y)**2/np.linalg.norm(M)**2)/(X.shape[1]*Y.shape[1]/384**2)
for k in (8,16):
    U1,S1,_=np.linalg.svd(O1@V1@T); o1=U1[:,:k]
    U5,S5,_=np.linalg.svd(O5@V5@T); o5=U5[:,:k]
    Tk=T[:,:k] if k<20 else T
    # use T's principal dirs matched: T basis ordered by eig of P already
    subs={'T':T[:,:k],'o5':o5,'o1':o1}
    print("k",k, "o1-o5 overlap max", np.round(np.linalg.svd(o1.T@o5,compute_uv=False)[:3],2), "o5-T", np.round(np.linalg.svd(o5.T@T,compute_uv=False)[:3],2), "o1-T", np.round(np.linalg.svd(o1.T@T,compute_uv=False)[:3],2))
    print("     q:T>k:o5  q:o5>k:T | q:o5>k:o1  q:o1>k:o5 | q:T>k:o1 q:o1>k:T | diag T, o5, o1")
    for l in range(1,4):
        for j in range(6):
            Q,K,V,O=mats(l,j); M=Q.T@K
            e=lambda a,b: enrich(M,subs[a],subs[b])
            print(f" L{l}H{j}  {e('T','o5'):5.1f} {e('o5','T'):5.1f} | {e('o5','o1'):5.1f} {e('o1','o5'):5.1f} | {e('T','o1'):5.1f} {e('o1','T'):5.1f} | {e('T','T'):5.1f} {e('o5','o5'):5.1f} {e('o1','o1'):5.1f}")
```