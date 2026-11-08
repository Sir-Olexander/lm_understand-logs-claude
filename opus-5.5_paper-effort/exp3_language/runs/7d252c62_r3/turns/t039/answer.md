```python
def mats(l,j):
    w=W[f'ln1_w.{l}']
    s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def topright(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
import scipy.linalg as sl
Q1,K1,V1,O1=mats(0,1); Q5,K5,V5,O5=mats(0,5)
OV5=O5@V5; OV1=O1@V1
A=topright(OV1,26); B=topright(OV5,26)
P=(A@A.T+B@B.T)/2
ev,evec=np.linalg.eigh(P); idx=np.argsort(-ev); T=evec[:,idx[:26]]
forms={}
for (l,j) in [(2,1),(3,1),(3,5)]:
    Q,K,V,O=mats(l,j); M=Q.T@K
    G=T.T@M@OV5@T; forms[f'L{l}H{j}_T5']=(G+G.T)/2
Q,K,V,O=mats(3,5); M=Q.T@K
G=T.T@OV5.T@M@OV5@T; forms['L3H5_55']=(G+G.T)/2
G=T.T@OV1.T@M@OV1@T; forms['L3H5_11']=(G+G.T)/2
Q,K,V,O=mats(2,2); M=Q.T@K
G=T.T@OV5.T@M@T; forms['L2H2_5T']=(G+G.T)/2
for k,G in forms.items():
    print(k, np.round(np.sort(np.linalg.eigvalsh(G))[::-1][:26]/np.max(np.linalg.eigvalsh(G)),2))
```