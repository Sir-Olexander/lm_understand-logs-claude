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
ev,evec=np.linalg.eigh(P); idx=np.argsort(-ev); T=evec[:,idx[:24]]
OV5=O5@V5; OV1=O1@V1
def stat(Bm):
    s=np.linalg.svd(Bm,compute_uv=False); e=np.linalg.eigvals(Bm)
    return np.sum(e.real)/np.sum(s), np.linalg.norm(Bm)
print("cols: [qT,kprev5] [qT,kprev1] [qprev5,kT] [qprev1,kT] [qprev5,kprev1] [qprev1,kprev5] [qT,kT] [q5,k5] [q1,k1] ; value: trace/sum|sv| (norm)")
for l in (1,2,3):
  for j in range(6):
    Q,K,V,O=mats(l,j); M=Q.T@K
    blocks=[T.T@M@OV5@T, T.T@M@OV1@T, T.T@OV5.T@M@T, T.T@OV1.T@M@T, T.T@OV5.T@M@OV1@T, T.T@OV1.T@M@OV5@T, T.T@M@T, T.T@OV5.T@M@OV5@T, T.T@OV1.T@M@OV1@T]
    print(f"L{l}H{j} "+" ".join(f"{stat(b)[0]:+.2f}({stat(b)[1]:.1f})" for b in blocks))
```