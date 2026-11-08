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
Pt=np.eye(384)-T@T.T
M1=Pt@(Q1.T@K1)@Pt; M5=Pt@(Q5.T@K5)@Pt
for nm,M in [('M1',M1),('M5',M5)]:
    print(nm,'sv',np.round(np.linalg.svd(M,compute_uv=False)[:30],2))
# sym/antisym
for nm,M in [('M1',M1),('M5',M5)]:
    S=(M+M.T)/2; Aa=(M-M.T)/2
    es=np.linalg.eigvalsh(S)
    print(nm,'|sym|',np.linalg.norm(S).round(2),'|anti|',np.linalg.norm(Aa).round(2),'sym eig extremes',np.round(es[:6],2),np.round(es[-6:],2))
```