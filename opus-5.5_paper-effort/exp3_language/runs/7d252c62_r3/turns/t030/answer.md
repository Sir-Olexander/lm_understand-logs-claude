```python
def mats(l,j):
    w=W[f'ln1_w.{l}']
    s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def topright(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
Q1,K1,V1,O1=mats(0,1); Q5,K5,V5,O5=mats(0,5)
A=topright(O1@V1,36); B=topright(O5@V5,36)
P=(A@A.T+B@B.T)/2
ev,evec=np.linalg.eigh(P); idx=np.argsort(-ev); T=evec[:,idx[:36]]
print("proj eig", np.round(ev[idx[:40]],2))
OV5=O5@V5; OV1=O1@V1
for l in (1,2,3):
  for j in range(6):
    Q,K,V,O=mats(l,j); M=Q.T@K
    Bm=T.T@OV5.T@M@T   # query reads prev token (via OV5), key reads current token
    s=np.linalg.svd(Bm,compute_uv=False)
    e=np.linalg.eigvals(Bm)
    print(f"L{l}H{j} sv", np.round(s[:32]/s[0],2), " trace/sum|sv|", round(np.sum(e.real)/np.sum(s),2))
```