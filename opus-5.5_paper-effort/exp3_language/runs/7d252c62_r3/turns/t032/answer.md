```python
def mats(l,j):
    w=W[f'ln1_w.{l}']
    s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def topright(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
Q1,K1,V1,O1=mats(0,1); Q5,K5,V5,O5=mats(0,5)
OV5=O5@V5; OV1=O1@V1
A=topright(OV1,40); B=topright(OV5,40)
P=(A@A.T+B@B.T)/2
ev,evec=np.linalg.eigh(P); idx=np.argsort(-ev); T=evec[:,idx[:40]]
for (l,j,name) in [(2,1,'qT,k5'),(3,1,'qT,k5'),(3,5,'qT,k5'),(3,5,'q5,k5'),(3,5,'q1,k1'),(2,1,'q5,k1'),(3,5,'q5,k1'),(2,2,'q5,kT')]:
    Q,K,V,O=mats(l,j); M=Q.T@K
    if name=='qT,k5': G=T.T@M@OV5@T
    if name=='q5,k5': G=T.T@OV5.T@M@OV5@T
    if name=='q1,k1': G=T.T@OV1.T@M@OV1@T
    if name=='q5,k1': G=T.T@OV5.T@M@OV1@T
    if name=='q5,kT': G=T.T@OV5.T@M@T
    S=(G+G.T)/2
    es=np.sort(np.linalg.eigvalsh(S))[::-1]
    print(f"L{l}H{j} {name} sym eig:", np.round(es/es[0],2))
```