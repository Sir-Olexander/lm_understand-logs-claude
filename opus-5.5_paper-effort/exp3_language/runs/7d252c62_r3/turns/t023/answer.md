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
np.save('/tmp/T20.npy',T)
dirs={}
for j in range(6):
    Q,K,V,O=mats(0,j); M=Q.T@K
    Mk=M@T   # query any, key in T (20 dims)
    U,S,Vt=np.linalg.svd(Mk)
    b=W['ln1_b.0']; s=slice(64*j,64*j+64)
    kp=T.T@(K.T@(W['W_q.0'][s]@b))
    print(f"H{j} q-any/k-T sv", np.round(S[:6],3), "| keypref-in-T norm", np.round(np.linalg.norm(kp)/8,3))
    dirs[j]=Vt[0]
for a in range(6):
    print(a, [round(abs(dirs[a]@dirs[b]),2) for b in range(6)])
```