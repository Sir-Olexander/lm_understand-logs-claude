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
Qs=[]
for j in (1,5):
    Q,K,V,O=mats(0,j); Qs.append(topright(Q,30)); Qs.append(topright(K,30))
Pp=sum(x@x.T for x in Qs)/4
ev2,evec2=np.linalg.eigh(Pp); idx2=np.argsort(-ev2); Ppos=evec2[:,idx2[:24]]
Ppos=np.linalg.qr(Ppos-T@(T.T@Ppos))[0]
w1=W['ln1_w.0']; w2=W['ln2_w.0']
dT=np.sum(T**2,1); dP=np.sum(Ppos**2,1)
print("corr dT,ln1w",np.corrcoef(dT,w1)[0,1],"dP,ln1w",np.corrcoef(dP,w1)[0,1],"dT,ln2w",np.corrcoef(dT,w2)[0,1],"dP,ln2w",np.corrcoef(dP,w2)[0,1])
# axis alignment: participation ratio
print("T diag top", np.round(np.sort(dT)[::-1][:20],2), "sum", dT.sum())
print("P diag top", np.round(np.sort(dP)[::-1][:20],2), "sum", dP.sum())
# Note: subspaces were computed with ln1_w scaling; do raw (unscaled) versions too
o=np.argsort(-w1)
print("top ln1_w dims: dT", np.round(dT[o[:30]],2))
print("top ln1_w dims: dP", np.round(dP[o[:30]],2))
```