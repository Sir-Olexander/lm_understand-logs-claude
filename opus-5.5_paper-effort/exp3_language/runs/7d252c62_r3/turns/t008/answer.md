```python
def mats(l,j):
    w=W[f'ln1_w.{l}']
    s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def topright(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
Q1,K1,V1,O1=mats(0,1); Q5,K5,V5,O5=mats(0,5)
A=topright(O1@V1,24); B=topright(O5@V5,24)
print("cos principal angles OV1 vs OV5 input:", np.round(np.linalg.svd(A.T@B,compute_uv=False),2))
# random baseline expected ~ sqrt(24/384)
# QK input subspaces
for j in range(6):
    Q,K,V,O=mats(0,j)
    U,S,Vt=np.linalg.svd(Q); qs=Vt[:20].T
    U,S,Vt=np.linalg.svd(K); ks=Vt[:20].T
    T=np.linalg.qr(np.hstack([A,B]))[0]
    print(j,"Q in tokspace:",np.round(np.linalg.svd(T.T@qs,compute_uv=False)[:10],2))
    print(j,"K in tokspace:",np.round(np.linalg.svd(T.T@ks,compute_uv=False)[:10],2))
    print(j,"OV in in tokspace:",np.round(np.linalg.svd(T.T@topright(O@V,4),compute_uv=False),2))
```