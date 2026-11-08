```python
def mats(l,j):
    w=W[f'ln1_w.{l}']
    s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def topright(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
Q1,K1,V1,O1=mats(0,1); Q5,K5,V5,O5=mats(0,5)
OV5=O5@V5; OV1=O1@V1
A=topright(OV1,26); B=topright(OV5,26)
P=(A@A.T+B@B.T)/2
ev,evec=np.linalg.eigh(P); idx=np.argsort(-ev); T=evec[:,idx[:24]]
dirs={}
for j in range(6):
    Q,K,V,O=mats(0,j); M=Q.T@K
    U,S,Vt=np.linalg.svd(M@T); dirs[j]=(T@Vt[0], U[:,0], S[0])
uA=dirs[5][0]; uB=dirs[3][0]
# token-part of L0H4/H0 V input
for j in (4,0,2):
    Q,K,V,O=mats(0,j)
    OV=O@V
    print(j, "OV applied to uA", np.linalg.norm(OV@uA).round(3), "uB", np.linalg.norm(OV@uB).round(3), "avg T", (np.linalg.norm(OV@T)/np.sqrt(24)).round(3))
# For H3: what query-side vector pairs with uB key; is it positional slow plane?
Q4,K4,V4,O4=mats(0,4); in4=topright(O4@V4,2)
for j in (1,3,5,0,2):
    t,qv,s=dirs[j]
    print(j, "key tok dir . uA", abs(t@uA).round(2), "uB", abs(t@uB).round(2), "| query vec in H4-plane", np.linalg.norm(in4.T@qv).round(2), "in T", np.linalg.norm(T.T@qv).round(2), "sv", s.round(3))
```